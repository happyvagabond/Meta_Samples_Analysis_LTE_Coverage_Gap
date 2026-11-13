import multiprocessing as mp
import warnings
import gc
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import folium
from folium.plugins import Search, Fullscreen

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================
RSRP_GOOD = -105.0
RSRP_POOR = -110.0
RADIUS_M = 1000.0
GH_CHUNK_SIZE = 50_000   # Keeps C-parser memory lightweight
N_WORKERS = max(1, mp.cpu_count() - 1)

OUTPUT_HTML = "Site_Cluster_Map3.html"

# Leaflet Hex Colors
RSRP_COLORS = {
    "Good":     "#00E676",  # Bright Green
    "Moderate": "#FFD600",  # Yellow
    "Poor":     "#FF1744",  # Red
    "No Data":  "#9E9E9E",  # Neutral Grey
}

# ============================================================
# UTILS & HELPERS
# ============================================================
def safe_read_csv(path, **kw):
    try:
        return pd.read_csv(path, encoding="utf-8", **kw)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kw)

def rsrp_class(v):
    if v is None or np.isnan(v):
        return "No Data"
    if v > RSRP_GOOD:
        return "Good"
    if v >= RSRP_POOR:
        return "Moderate"
    return "Poor"

def get_utm_crs(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return f"EPSG:{32600 + zone if lat >= 0 else 32700 + zone}"

# ============================================================
# LOAD SITES & CREATE SECTOR CLUSTERS
# ============================================================
def load_sites(path):
    df = safe_read_csv(path)
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

    # Vectorized sector angles
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
# TRAFFIC META-SAMPLE TAGGING (TOP 3 RANKING)
# ============================================================
def compute_top_meta(meta_file, clusters):
    meta = safe_read_csv(meta_file)
    meta["TotalSamples"] = meta["A Samples"] + meta["J Samples"]

    cols = ["Site_ID", "TotalSamples"] if "Site_ID" in meta.columns else ["TotalSamples"]
    gdf = gpd.GeoDataFrame(
        meta[cols].copy(),
        geometry=gpd.points_from_xy(meta.Longitude, meta.Latitude),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(gdf, clusters[["Site_ID", "Cluster_ID", "geometry"]], predicate="within")
    site_col = "Site_ID_right" if "Site_ID_right" in joined.columns else "Site_ID"
    
    summary = joined.groupby([site_col, "Cluster_ID"])["TotalSamples"].sum().reset_index()
    summary.rename(columns={site_col: "Site_ID"}, inplace=True)
    summary["Site_ID"] = summary["Site_ID"].astype(str)
    
    summary["Rank"] = summary.groupby("Site_ID")["TotalSamples"].rank(ascending=False, method="first").astype(int)

    return summary[summary["Rank"] <= 3]

# ============================================================
# PARALLEL RSRP AGGREGATION (STREAMING WITH LOW MEMORY)
# ============================================================
_SHARED_CLUSTERS = None
_BBOX = None

def init_worker(shared_clusters, bbox):
    global _SHARED_CLUSTERS, _BBOX
    _SHARED_CLUSTERS = shared_clusters
    _BBOX = bbox
    _ = _SHARED_CLUSTERS.sindex

def gh_worker(chunk):
    global _SHARED_CLUSTERS, _BBOX
    
    minx, miny, maxx, maxy = _BBOX
    mask = (
        (chunk["Tile Longitude"] >= minx) &
        (chunk["Tile Longitude"] <= maxx) &
        (chunk["Tile Latitude"] >= miny) &
        (chunk["Tile Latitude"] <= maxy)
    )
    filtered = chunk[mask]
    if filtered.empty:
        return None

    gdf = gpd.GeoDataFrame(
        filtered[["Avg_RSRP"]],
        geometry=gpd.points_from_xy(filtered["Tile Longitude"], filtered["Tile Latitude"]),
        crs="EPSG:4326"
    )

    joined = gpd.sjoin(gdf, _SHARED_CLUSTERS[["Site_ID", "Cluster_ID", "geometry"]], predicate="within")
    if joined.empty:
        return None

    stats = joined.groupby(["Site_ID", "Cluster_ID"])["Avg_RSRP"].agg(["sum", "count"]).reset_index()
    return stats

def compute_rsrp_parallel(gh_file, clusters):
    col_map = {
        "Tile Latitude": "Tile Latitude",
        "Tile Longitude": "Tile Longitude",
        "Average RSRP/RSRQ (Serving)/Serving Cell Average RSRP-All MR (dBm)": "Avg_RSRP"
    }

    total_bounds = clusters.total_bounds
    bbox = (
        total_bounds[0] - 0.01,
        total_bounds[1] - 0.01,
        total_bounds[2] + 0.01,
        total_bounds[3] + 0.01
    )

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

    def chunk_generator():
        for chunk in reader:
            yield chunk.rename(columns=col_map)

    aggregated_results = []

    with mp.Pool(processes=N_WORKERS, initializer=init_worker, initargs=(clusters, bbox)) as pool:
        for res in pool.imap(gh_worker, chunk_generator(), chunksize=1):
            if res is not None and not res.empty:
                aggregated_results.append(res)
            
            if len(aggregated_results) % 50 == 0:
                gc.collect()

    if not aggregated_results:
        return pd.DataFrame(columns=["Site_ID", "Cluster_ID", "Avg_RSRP"])

    combined = pd.concat(aggregated_results, ignore_index=True)
    grouped = combined.groupby(["Site_ID", "Cluster_ID"]).agg({"sum": "sum", "count": "sum"}).reset_index()
    grouped["Avg_RSRP"] = grouped["sum"] / grouped["count"]
    grouped["Site_ID"] = grouped["Site_ID"].astype(str)

    del aggregated_results, combined
    gc.collect()
    
    return grouped[["Site_ID", "Cluster_ID", "Avg_RSRP"]]

# ============================================================
# BUILD FAST HTML MAP (GOOGLE EARTH CLUTTER + KML/KMZ UPLOADER)
# ============================================================
def build_html_map(clusters, rsrp, top_meta):
    df = clusters.merge(rsrp, on=["Site_ID", "Cluster_ID"], how="left")
    df["RSRP_Class"] = df["Avg_RSRP"].apply(rsrp_class)
    df["Avg_RSRP_Display"] = df["Avg_RSRP"].apply(lambda v: f"{v:.2f} dBm" if pd.notnull(v) else "N/A")

    top_meta_clean = top_meta[["Site_ID", "Cluster_ID", "Rank"]].copy()
    top_meta_clean["Site_ID"] = top_meta_clean["Site_ID"].astype(str)
    df = df.merge(top_meta_clean, on=["Site_ID", "Cluster_ID"], how="left")
    
    df["Is_Top_Traffic"] = df["Rank"].notnull()
    df["Rank_Display"] = df["Rank"].apply(lambda r: f"Rank {int(r)}" if pd.notnull(r) else "Unranked")

    # Center map on centroid
    center_lat = clusters.geometry.centroid.y.mean()
    center_lon = clusters.geometry.centroid.x.mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
        prefer_canvas=True
    )

    # ------------------------------------------------------------
    # GOOGLE EARTH & CLUTTER BASE LAYERS
    # ------------------------------------------------------------
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Earth Clutter Imagery",
        name="Google Earth Imagery (Pure Clutter)",
        max_zoom=21,
        subdomains=["mt0", "mt1", "mt2", "mt3"]
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",
        attr="Google Earth Hybrid",
        name="Google Hybrid (Satellite + Overlay)",
        max_zoom=21,
        subdomains=["mt0", "mt1", "mt2", "mt3"],
        show=True
    ).add_to(m)

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
        attr="Google Terrain 3D",
        name="Google 3D Terrain & Relief",
        max_zoom=21,
        subdomains=["mt0", "mt1", "mt2", "mt3"],
        show=False
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri High-Res Clutter Imagery",
        max_zoom=19,
        show=False
    ).add_to(m)

    # ------------------------------------------------------------
    # HIGH-TRANSPARENCY VECTOR STYLING
    # ------------------------------------------------------------
    def style_fn(feature):
        props = feature["properties"]
        rsrp_cat = props.get("RSRP_Class", "No Data")
        is_top = props.get("Is_Top_Traffic", False)
        fill_color = RSRP_COLORS.get(rsrp_cat, "#9E9E9E")

        return {
            "fillColor": fill_color,
            "fillOpacity": 0.22 if is_top else 0.16,
            "color": "#FFFFFF" if is_top else fill_color,
            "weight": 2.5 if is_top else 1.0,
            "opacity": 0.95 if is_top else 0.65,
            "dashArray": None if is_top else "3, 4"
        }

    def highlight_fn(_):
        return {
            "fillOpacity": 0.45,
            "weight": 3.5,
            "color": "#00FFFF",
            "dashArray": None
        }

    geojson_layer = folium.GeoJson(
        data=df.__geo_interface__,
        name="Translucent Sector Clusters",
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["Site_ID", "Cluster_ID", "Rank_Display", "Avg_RSRP_Display", "RSRP_Class"],
            aliases=["Site ID:", "Sector ID:", "Traffic Rank:", "Avg RSRP:", "Quality:"],
            localize=True,
            sticky=True,
            style="background-color: rgba(18, 24, 38, 0.90); color: #FFF; font-family: sans-serif; font-size: 12px; padding: 10px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.2); box-shadow: 0 4px 12px rgba(0,0,0,0.5);"
        ),
        popup=folium.GeoJsonPopup(
            fields=["Site_ID", "Cluster_ID", "Rank_Display", "Avg_RSRP_Display", "RSRP_Class"],
            aliases=["Site ID:", "Sector ID:", "Traffic Rank:", "Avg RSRP:", "Quality:"]
        )
    ).add_to(m)

    # ------------------------------------------------------------
    # SITE PIN MARKERS
    # ------------------------------------------------------------
    site_points_group = folium.FeatureGroup(name="Site Center Beacons", show=True)
    site_roots = df.groupby("Site_ID")["geometry"].apply(lambda geoms: geoms.iloc[0].exterior.coords[0]).reset_index()

    for _, row in site_roots.iterrows():
        s_id = str(row["Site_ID"])
        root_lon, root_lat = row["geometry"]
        folium.CircleMarker(
            location=[root_lat, root_lon],
            radius=4,
            color="#FFFFFF",
            weight=1.5,
            fill=True,
            fill_color="#FF0055",
            fill_opacity=1.0,
            tooltip=f"<b>Site ID:</b> {s_id}",
            popup=f"<b>Site ID:</b> {s_id}<br><b>Coordinates:</b> {root_lat:.5f}, {root_lon:.5f}"
        ).add_to(site_points_group)
    
    site_points_group.add_to(m)

    # ============================================================
    # SEARCH BAR: DYNAMIC ZOOM-IN TO SITE_ID
    # ============================================================
    Search(
        layer=geojson_layer,
        geom_type="Polygon",
        search_label="Site_ID",
        search_zoom=16,
        auto_collapse=True,
        auto_type=True,
        animate_location=True,
        placeholder="Search Site ID...",
        collapsed=False,
        position="topleft"
    ).add_to(m)

    # UI Controls
    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    Fullscreen(position="topright").add_to(m)

    # Legend UI
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        background: rgba(18, 24, 38, 0.88);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 14px 18px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        line-height: 1.6;
        color: #E0E0E0;
    ">
        <b style="font-size: 13px; color: #FFFFFF; letter-spacing: 0.5px;">RSRP Coverage (Clutter View)</b><br>
        <span style="display:inline-block;width:14px;height:14px;background:{RSRP_COLORS['Good']};opacity:0.75;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Good (&gt; {RSRP_GOOD} dBm)<br>
        <span style="display:inline-block;width:14px;height:14px;background:{RSRP_COLORS['Moderate']};opacity:0.75;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Moderate ({RSRP_POOR} to {RSRP_GOOD} dBm)<br>
        <span style="display:inline-block;width:14px;height:14px;background:{RSRP_COLORS['Poor']};opacity:0.75;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>Poor (&lt; {RSRP_POOR} dBm)<br>
        <span style="display:inline-block;width:14px;height:14px;background:{RSRP_COLORS['No Data']};opacity:0.75;border-radius:2px;margin-right:6px;vertical-align:middle;"></span>No Data<br>
        <hr style="margin: 8px 0; border: none; border-top: 1px solid rgba(255,255,255,0.15);">
        <span style="display:inline-block;width:14px;height:3px;background:#FFFFFF;margin-right:6px;vertical-align:middle;"></span><b>White Solid:</b> Top 3 Traffic Sector<br>
        <span style="display:inline-block;width:14px;height:2px;border-top:2px dashed #AAA;margin-right:6px;vertical-align:middle;"></span><b>Dashed:</b> Unranked Sectors<br>
        <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#FF0055;border:1px solid #FFF;margin-right:6px;vertical-align:middle;"></span><b>Dot:</b> Site Center Node<br>
        <small style="color: #9E9E9E; display:block; margin-top:4px;"><i>Hover sectors to inspect traffic and clutter</i></small>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Inject JSZip via CDN for zero-overhead client-side KMZ decompression
    m.get_root().header.add_child(
        folium.Element('<script src="https://cdnjs.cloudflare.com/ajax/libs/jszip/3.10.1/jszip.min.js"></script>')
    )

    # ============================================================
    # FAST STREAMING CLIENT-SIDE KML & KMZ UPLOADER WIDGET
    # ============================================================
    spatial_uploader_html = r"""
    <div id="spatial-upload-box" style="
        position: fixed;
        bottom: 30px;
        left: 20px;
        z-index: 1000;
        background: rgba(18, 24, 38, 0.90);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        padding: 12px 16px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 18px rgba(0,0,0,0.45);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: #FFFFFF;
        width: 250px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
            <b style="font-size: 13px; letter-spacing: 0.3px;">KML / KMZ Overlay</b>
            <span id="spatial-count" style="font-size: 11px; color: #00E676;"></span>
        </div>
        
        <input type="file" id="spatial-file-input" accept=".kml,.kmz" style="display: none;" onchange="handleSpatialUpload(event)" />
        
        <div style="display: flex; gap: 8px;">
            <button onclick="document.getElementById('spatial-file-input').click()" style="
                flex: 1;
                background: #00E5FF;
                color: #000;
                font-weight: 600;
                border: none;
                padding: 6px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
                transition: background 0.2s;
            " onmouseover="this.style.background='#80D8FF'" onmouseout="this.style.background='#00E5FF'">
                Load KML / KMZ
            </button>
            <button onclick="clearSpatialLayer()" style="
                background: rgba(255, 255, 255, 0.15);
                color: #FFF;
                border: 1px solid rgba(255, 255, 255, 0.25);
                padding: 6px 10px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 11px;
            ">
                Clear
            </button>
        </div>
        <div id="spatial-status" style="font-size: 11px; color: #BBB; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
            No file loaded
        </div>
    </div>

    <script>
    var spatialLayerGroup = null;

    function getLeafletMapInstance() {
        for (var key in window) {
            if (key.startsWith("map_") && window[key] instanceof L.Map) {
                return window[key];
            }
        }
        return null;
    }

    function clearSpatialLayer() {
        var map = getLeafletMapInstance();
        if (map && spatialLayerGroup) {
            spatialLayerGroup.clearLayers();
            map.removeLayer(spatialLayerGroup);
            spatialLayerGroup = null;
        }
        document.getElementById('spatial-count').innerText = "";
        document.getElementById('spatial-status').innerText = "Layer cleared";
        document.getElementById('spatial-file-input').value = "";
    }

    function parseKMLString(kmlText, fileName) {
        var statusEl = document.getElementById('spatial-status');
        var countEl = document.getElementById('spatial-count');
        var map = getLeafletMapInstance();

        if (!map) {
            statusEl.innerText = "Error: Map not found";
            return;
        }

        if (!spatialLayerGroup) {
            spatialLayerGroup = L.layerGroup().addTo(map);
        } else {
            spatialLayerGroup.clearLayers();
        }

        var parser = new DOMParser();
        var xmlDoc = parser.parseFromString(kmlText, "text/xml");
        var placemarks = xmlDoc.getElementsByTagName("Placemark");
        var bounds = L.latLngBounds([]);
        var featureCount = 0;

        var pointMarkerOpts = {
            radius: 5,
            fillColor: "#00E5FF",
            color: "#FFFFFF",
            weight: 1.5,
            opacity: 0.9,
            fillOpacity: 0.8
        };

        for (var i = 0; i < placemarks.length; i++) {
            var pm = placemarks[i];
            var nameEl = pm.getElementsByTagName("name")[0];
            var name = nameEl ? nameEl.textContent : ("Feature " + (i + 1));
            var descEl = pm.getElementsByTagName("description")[0];
            var desc = descEl ? descEl.textContent : "";
            var popupContent = "<b>" + name + "</b>" + (desc ? ("<br><small>" + desc + "</small>") : "");

            // 1. Point Geometries
            var pointEl = pm.getElementsByTagName("Point")[0];
            if (pointEl) {
                var coordEl = pointEl.getElementsByTagName("coordinates")[0];
                if (coordEl) {
                    var raw = coordEl.textContent.trim().split(/[\s,]+/);
                    if (raw.length >= 2) {
                        var lon = parseFloat(raw[0]);
                        var lat = parseFloat(raw[1]);
                        if (!isNaN(lat) && !isNaN(lon)) {
                            var latlng = [lat, lon];
                            bounds.extend(latlng);
                            var marker = L.circleMarker(latlng, pointMarkerOpts)
                                .bindTooltip(name, { sticky: true })
                                .bindPopup(popupContent);
                            spatialLayerGroup.addLayer(marker);
                            featureCount++;
                        }
                    }
                }
            }

            // 2. LineString Geometries
            var lineEl = pm.getElementsByTagName("LineString")[0];
            if (lineEl) {
                var lineCoordsEl = lineEl.getElementsByTagName("coordinates")[0];
                if (lineCoordsEl) {
                    var entries = lineCoordsEl.textContent.trim().split(/\s+/);
                    var lineCoords = [];
                    for (var j = 0; j < entries.length; j++) {
                        var parts = entries[j].split(",");
                        if (parts.length >= 2) {
                            var l_lon = parseFloat(parts[0]);
                            var l_lat = parseFloat(parts[1]);
                            if (!isNaN(l_lat) && !isNaN(l_lon)) {
                                lineCoords.push([l_lat, l_lon]);
                                bounds.extend([l_lat, l_lon]);
                            }
                        }
                    }
                    if (lineCoords.length > 0) {
                        var polyline = L.polyline(lineCoords, { color: "#FFEB3B", weight: 2.5, opacity: 0.85 })
                            .bindTooltip(name, { sticky: true })
                            .bindPopup(popupContent);
                        spatialLayerGroup.addLayer(polyline);
                        featureCount++;
                    }
                }
            }

            // 3. Polygon Geometries
            var polyEl = pm.getElementsByTagName("Polygon")[0];
            if (polyEl) {
                var polyCoordsEl = polyEl.getElementsByTagName("coordinates")[0];
                if (polyCoordsEl) {
                    var polyEntries = polyCoordsEl.textContent.trim().split(/\s+/);
                    var polyCoords = [];
                    for (var k = 0; k < polyEntries.length; k++) {
                        var pParts = polyEntries[k].split(",");
                        if (pParts.length >= 2) {
                            var p_lon = parseFloat(pParts[0]);
                            var p_lat = parseFloat(pParts[1]);
                            if (!isNaN(p_lat) && !isNaN(p_lon)) {
                                polyCoords.push([p_lat, p_lon]);
                                bounds.extend([p_lat, p_lon]);
                            }
                        }
                    }
                    if (polyCoords.length > 0) {
                        var polygon = L.polygon(polyCoords, { color: "#00E676", fillColor: "#00E676", fillOpacity: 0.25, weight: 1.5 })
                            .bindTooltip(name, { sticky: true })
                            .bindPopup(popupContent);
                        spatialLayerGroup.addLayer(polygon);
                        featureCount++;
                    }
                }
            }
        }

        if (featureCount > 0) {
            statusEl.innerText = fileName;
            countEl.innerText = featureCount + " features";
            if (bounds.isValid()) {
                map.flyToBounds(bounds, { maxZoom: 16, padding: [40, 40] });
            }
        } else {
            statusEl.innerText = "No features found";
            countEl.innerText = "0 items";
        }
    }

    function handleSpatialUpload(event) {
        var file = event.target.files[0];
        if (!file) return;

        var statusEl = document.getElementById('spatial-status');
        var lowerName = file.name.toLowerCase();
        statusEl.innerText = "Reading " + file.name + "...";

        if (lowerName.endsWith(".kmz")) {
            // Process zipped KMZ archive using JSZip in-memory
            var reader = new FileReader();
            reader.onload = function(e) {
                JSZip.loadAsync(e.target.result).then(function(zip) {
                    // Find the primary .kml file inside archive (doc.kml or first encountered .kml)
                    var kmlFile = null;
                    zip.forEach(function(relativePath, entry) {
                        if (relativePath.toLowerCase().endsWith(".kml") && !kmlFile) {
                            kmlFile = entry;
                        }
                    });

                    if (kmlFile) {
                        return kmlFile.async("string");
                    } else {
                        throw new Error("No KML file found inside KMZ package");
                    }
                }).then(function(kmlText) {
                    parseKMLString(kmlText, file.name);
                }).catch(function(err) {
                    console.error(err);
                    statusEl.innerText = "Error parsing KMZ package";
                });
            };
            reader.readAsArrayBuffer(file);

        } else if (lowerName.endsWith(".kml")) {
            // Process standard uncompressed KML text
            var reader = new FileReader();
            reader.onload = function(e) {
                try {
                    parseKMLString(e.target.result, file.name);
                } catch(err) {
                    console.error(err);
                    statusEl.innerText = "Error parsing KML";
                }
            };
            reader.readAsText(file);
        } else {
            statusEl.innerText = "Unsupported format (must be .kml or .kmz)";
        }
    }
    </script>
    """
    m.get_root().html.add_child(folium.Element(spatial_uploader_html))

    m.save(OUTPUT_HTML)

# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading sites...")
    sites = load_sites("Sites.CSV")

    print("Creating sector clusters...")
    clusters_list = [create_site_clusters(r) for _, r in sites.iterrows()]
    clusters = gpd.GeoDataFrame(pd.concat(clusters_list, ignore_index=True), crs="EPSG:4326")

    print("Computing top meta clusters...")
    top_meta = compute_top_meta("Meta_Samples.CSV", clusters)

    print("Processing GH RSRP samples in parallel (Streaming Mode)...")
    rsrp = compute_rsrp_parallel("GH_Samples.CSV", clusters)

    print("Generating HTML map with Google Earth Clutter, Transparent Clusters, and Fast KML/KMZ Upload...")
    build_html_map(clusters, rsrp, top_meta)

    print(f"\n Interactive HTML Map Ready: {OUTPUT_HTML}")

if __name__ == "__main__":
    main()