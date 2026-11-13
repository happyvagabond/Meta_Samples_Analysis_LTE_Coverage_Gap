Code snippet
# Meta Samples Coverage Analysis & Optimization Project

## 📖 Executive Summary
This automated Python-based workflow identifies coverage gaps in high-Meta sample clusters using 1 km site buffers segmented into 60-degree azimuths. The tool enables targeted RF improvements—such as hygiene corrections or antenna reorientations—**without requiring new CAPEX**. 

By prioritizing population-dense areas with high traffic but poor RF conditions, this project supports revenue optimization, reduces churn, and provides actionable insights for network planning leads.

---

## 🎯 Business Impact
*   **Revenue Optimization:** Identifies sites for physical optimization, resulting in the opportunity to onboard/retain populations in revenue-critical zones.
*   **CAPEX Savings:** Minimizes CAPEX leakage by focusing on physical optimization (azimuth/hygiene) rather than deploying new sites.
*   **Churn Reduction:** Cumulative churn at optimized sites is expected to drop significantly (industry parallels show 15-33% reductions).

---

## 🧠 Methodology & Logic (From PDF)
1.  **Buffer Creation:** Creates a **1 km radius** buffer around each site.
2.  **Segmentation:** Divides each buffer into six 60-degree segments:
    *   `Segment_1`: 0°-60° (Center: 30°)
    *   `Segment_2`: 60°-120° (Center: 90°)
    *   `Segment_3`: 120°-180° (Center: 150°)
    *   `Segment_4`: 180°-240° (Center: 210°)
    *   `Segment_5`: 240°-300° (Center: 270°)
    *   `Segment_6`: 300°-360° (Center: 330°)
3.  **Meta Mapping:** Maps 4G Meta Samples to these segments across all sites.
4.  **Prioritization:** Identifies the **Top 3 segments** per site based on total Meta sample tiles.
5.  **RF Overlay (GH):** Overlays GH RSRP sample tiles onto these segments.
6.  **Gap Identification:** Flags a top segment as "Uncovered" if the average RSRP falls **below -110 dBm**.

### 🛠️ Action Protocol
*   **Hygiene Correction:** If an existing sector points toward the uncovered segment (±30 degrees), prioritize physical hygiene checks/corrections.
*   **Reorientation:** If no sector aligns within ±30 degrees and 4G traffic permits, reorient the nearest sector's azimuth to cover the high-traffic area.

---

## ⚙️ Prerequisites
Ensure you have Python 3.8+ installed. Install the required libraries using:

```bash
pip install pandas numpy geopandas shapely folium openpyxl
```

🚀 Data Pipeline & Usage Workflow
The workflow is broken down into four consecutive scripts. Follow these steps to generate the final summaries and maps.

Step 1: Combine Meta Raw Data
We start with hundreds of downloaded raw CSV Meta files.

Place all raw Meta CSV files in a single directory.

Run the combiner script to merge them and remove duplicates.

```Bash
python combineCSV.py
```
Output: combined_output.csv

Step 2: Post-Process Meta Coordinates
The combined data needs coordinate splitting and column renaming for the downstream geospatial analysis.

Rename combined_output.csv to input.csv (or modify the script).

Run the post-processing script:

```Bash
python CSVPostP.py
```

Output: output.csv.Note: Rename this output.csv to Meta_Samples.CSV for the next steps.

Step 3: Run Summarization & Ticket Generation
This step applies the core business logic (1km buffers, 60-degree sectors, top 3 ranking, RSRP aggregation).

Ensure the following files are in your working directory:

Sites.CSV (Network site database)

Meta_Samples.CSV (Generated in Step 2)

GH_Samples.CSV (GH RSRP data)

Execute the summarization script:

```Bash
python Summarization_Sites_Meta_GH_Final.py
```

Outputs:

Final_Site_Summary.xlsx: Detailed site-by-site breakdown of top segments, total samples, and coverage remarks.

Weak_Sector_Tickets.xlsx: Actionable tickets for segments with high traffic but RSRP < -110 dBm.

Step 4: Generate Interactive HTML Map
Visualize the data to plan physical site visits and verify GH sample clusters against Google Earth Clutter.

Execute the visualization script:

```Bash
python Visualization_Sites_Meta_GH_HTMLFinal.py
```
Output: Site_Cluster_Map3.html

🗺️ Interactive Map Features (Site_Cluster_Map3.html)
Google Earth Clutter Imagery: High-resolution base maps for urban/terrain context.

Translucent Sector Clusters:

Solid White Borders: Top 3 Traffic Sectors.

Dashed Borders: Unranked/Low Traffic Sectors.

Color-Coded RSRP:

🟢 Good (> -105 dBm)

🟡 Moderate (-110 to -105 dBm)

🔴 Poor (< -110 dBm)

Dynamic Search: Search for a specific Site_ID to automatically fly/zoom to that location.

Client-Side Spatial Uploader: Drag-and-drop KML/KMZ files directly onto the browser map to overlay external datasets instantly (zero server overhead).

📝 Notes for Future Development
Scalability: The scripts utilize chunking (GH_CHUNK_SIZE) and Python multiprocessing (mp.Pool) to keep memory footprint lightweight when parsing millions of rows of GH data.

Next Steps: After concluding trials, retune the model to include Small Cells & ULS data across larger geographical territories.