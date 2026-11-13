# ============================================================
# COMPLETE OPTIMIZED PRODUCTION SCRIPT (RECTIFIED)
# ============================================================

import warnings
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
RSRP_THRESHOLD = -110.0
RADIUS_M = 1000.0
GH_CHUNK_SIZE = 1_000_000  # Balanced for memory and vectorized throughput

# ============================================================
# UTILS & READERS
# ============================================================
def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, encoding="utf-8", **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)

def get_utm_crs(lon, lat):
    """Calculates local UTM zone for accurate metric geometry."""
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

# ============================================================
# LOAD SITES & CREATE 6-SECTOR CLUSTERS (60 DEG EACH)
# ============================================================
def load_sites(path):
    df = safe_read_csv(path)
    df["Site_ID"] = df["Site_ID"].astype(str)
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Longitude, df.Latitude),
        crs="EPSG:4326"
    )

def create_site_clusters(site_row):
    lon, lat = site_row.geometry.x, site_row.geometry.y
    utm_crs = get_utm_crs(lon, lat)
    
    pt = gpd.GeoSeries([site_row.geometry], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
    sectors = []

    angles_per_sector = np.linspace(0, 60, 25)
    for i in range(6):
        deg = i * 60 + angles_per_sector
        rad = np.deg2rad(deg)
        arc_x = pt.x + RADIUS_M * np.sin(rad)
        arc_y = pt.y + RADIUS_M * np.cos(rad)
        poly_coords = [(pt.x, pt.y)] + list(zip(arc_x, arc_y))
        sectors.append(Polygon(poly_coords))

    gdf = gpd.GeoDataFrame(
        {
            "Site_ID": [str(site_row.Site_ID)] * 6,
            "Cluster_ID": list(range(1, 7)),
            "geometry": sectors
        },
        crs=utm_crs
    )
    return gdf.to_crs("EPSG:4326")

# ============================================================
# META SAMPLE TAGGING & TOP 3 SECTORS PER SITE
# ============================================================
def compute_top_clusters(meta_file, clusters):
    meta = safe_read_csv(meta_file)
    meta["TotalSamples"] = meta["A Samples"] + meta["J Samples"]

    cols = ["Site_ID", "TotalSamples"] if "Site_ID" in meta.columns else ["TotalSamples"]
    gdf = gpd.GeoDataFrame(
        meta[cols].copy(),
        geometry=gpd.points_from_xy(meta.Longitude, meta.Latitude),
        crs="EPSG:4326"
    )

    # Spatial join to associate samples with sector clusters
    joined = gpd.sjoin(gdf, clusters[["Site_ID", "Cluster_ID", "geometry"]], predicate="within")
    site_col = "Site_ID_right" if "Site_ID_right" in joined.columns else "Site_ID"

    summary = (
        joined.groupby([site_col, "Cluster_ID"])["TotalSamples"]
        .sum()
        .reset_index()
        .rename(columns={site_col: "Site_ID"})
    )
    summary["Site_ID"] = summary["Site_ID"].astype(str)

    # Sort descending by TotalSamples to ensure deterministic tie-breaking
    summary = summary.sort_values(["Site_ID", "TotalSamples"], ascending=[True, False])
    summary["Rank"] = (
        summary.groupby("Site_ID")["TotalSamples"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    return summary[summary.Rank <= 3]

# ============================================================
# FIXED GH RSRP PROCESSING (WEIGHTED SUM & COUNT CHUNKED)
# ============================================================
def compute_rsrp_segments_chunked(gh_file, clusters):
    col_map = {
        "Tile Latitude": "Tile Latitude",
        "Tile Longitude": "Tile Longitude",
        "Average RSRP/RSRQ (Serving)/Serving Cell Average RSRP-All MR (dBm)": "Avg_RSRP"
    }

    # Bounding box filter to skip points outside cluster extent
    minx, miny, maxx, maxy = clusters.total_bounds
    pad = 0.01
    minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad

    chunk_stats = []

    reader = pd.read_csv(
        gh_file,
        chunksize=GH_CHUNK_SIZE,
        usecols=list(col_map.keys()),
        encoding="latin1",
        engine="c",
        low_memory=True,
        dtype={
            "Tile Latitude": "float32",
            "Tile Longitude": "float32",
            list(col_map.keys())[2]: "float32"
        }
    )

    for chunk in reader:
        chunk = chunk.rename(columns=col_map)
        
        # Fast bounding box mask
        mask = (
            (chunk["Tile Longitude"] >= minx) &
            (chunk["Tile Longitude"] <= maxx) &
            (chunk["Tile Latitude"] >= miny) &
            (chunk["Tile Latitude"] <= maxy)
        )
        filtered = chunk[mask]
        if filtered.empty:
            continue

        gdf = gpd.GeoDataFrame(
            filtered[["Avg_RSRP"]],
            geometry=gpd.points_from_xy(filtered["Tile Longitude"], filtered["Tile Latitude"]),
            crs="EPSG:4326"
        )

        joined = gpd.sjoin(
            gdf,
            clusters[["Site_ID", "Cluster_ID", "geometry"]],
            how="inner",
            predicate="within"
        )

        if joined.empty:
            continue

        # Track cumulative sum and count per chunk (avoids "average of averages")
        stats = (
            joined.groupby(["Site_ID", "Cluster_ID"])["Avg_RSRP"]
            .agg(sum_rsrp="sum", count_rsrp="count")
            .reset_index()
        )
        chunk_stats.append(stats)

    if not chunk_stats:
        return pd.DataFrame(columns=["Site_ID", "Cluster_ID", "Avg_RSRP"])

    combined = pd.concat(chunk_stats, ignore_index=True)
    final_stats = (
        combined.groupby(["Site_ID", "Cluster_ID"])
        .agg(total_sum=("sum_rsrp", "sum"), total_count=("count_rsrp", "sum"))
        .reset_index()
    )
    final_stats["Avg_RSRP"] = final_stats["total_sum"] / final_stats["total_count"]
    final_stats["Site_ID"] = final_stats["Site_ID"].astype(str)

    del chunk_stats, combined
    gc.collect()

    return final_stats[["Site_ID", "Cluster_ID", "Avg_RSRP"]]

# ============================================================
# FINAL OUTPUT + BUSINESS LOGIC
# ============================================================
def build_outputs(top_meta, rsrp):
    rows, tickets = [], []

    for site, grp in top_meta.groupby("Site_ID"):
        grp = grp.sort_values("TotalSamples", ascending=False)

        row = {"Site_ID": site}
        bad_segments = []

        for i, (_, r) in enumerate(grp.iterrows(), start=1):
            cluster_id = int(r.Cluster_ID)
            total_samples = int(r.TotalSamples)

            row[f"Top_Samples_Segment_{i}"] = f"Segment_{cluster_id}"
            row[f"TotalSamples_{i}"] = total_samples

            rs = rsrp[
                (rsrp.Site_ID == site) &
                (rsrp.Cluster_ID == cluster_id)
            ]

            if not rs.empty and pd.notnull(rs.Avg_RSRP.iloc[0]):
                avg_rsrp = round(float(rs.Avg_RSRP.iloc[0]), 2)
                row[f"Top_RSRP_Segment_{i}"] = f"Segment_{cluster_id}"
                row[f"Top_Avg_Serving_RSRP_{i}"] = avg_rsrp

                if avg_rsrp < RSRP_THRESHOLD:
                    bad_segments.append(cluster_id)
                    tickets.append({
                        "Site_ID": site,
                        "Segment": f"Segment_{cluster_id}",
                        "Avg_RSRP": avg_rsrp,
                        "Issue": f"RSRP below {RSRP_THRESHOLD} dBm"
                    })
            else:
                row[f"Top_RSRP_Segment_{i}"] = f"Segment_{cluster_id}"
                row[f"Top_Avg_Serving_RSRP_{i}"] = np.nan
                bad_segments.append(cluster_id)
                tickets.append({
                    "Site_ID": site,
                    "Segment": f"Segment_{cluster_id}",
                    "Avg_RSRP": None,
                    "Issue": "No GH samples"
                })

        if len(grp) < 3:
            row["Correct Remark"] = "Segment missing"
        elif not bad_segments:
            row["Correct Remark"] = "All three Segments covered"
        else:
            row["Correct Remark"] = (
                "Segment_" + ",".join(map(str, bad_segments)) + " not covered"
            )

        row["Logic"] = (
            f"If any one or more High Meta Samples Segment RSRP < {int(RSRP_THRESHOLD)}"
            if bad_segments
            else f"If all High Meta Samples Segment RSRP > {int(RSRP_THRESHOLD)}"
        )

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(tickets)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading sites...")
    sites = load_sites("Sites.CSV")

    print("Generating metric sector clusters...")
    clusters = gpd.GeoDataFrame(
        pd.concat([create_site_clusters(r) for _, r in sites.iterrows()], ignore_index=True),
        crs="EPSG:4326"
    )

    print("Identifying Top 3 Meta sample clusters...")
    top_meta = compute_top_clusters("Meta_Samples.CSV", clusters)

    print("Computing exact GH RSRP averages...")
    rsrp = compute_rsrp_segments_chunked("GH_Samples.CSV", clusters)

    print("Compiling site summaries and weak sector tickets...")
    final_df, tickets_df = build_outputs(top_meta, rsrp)

    final_df.to_excel("Final_Site_Summary.xlsx", index=False)
    tickets_df.to_excel("Weak_Sector_Tickets.xlsx", index=False)

    print("\n COMPLETED SUCCESSFULLY")
    print("→ Final_Site_Summary.xlsx")
    print("→ Weak_Sector_Tickets.xlsx")

if __name__ == "__main__":
    main()