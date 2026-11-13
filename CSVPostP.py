import pandas as pd

# Read input CSV
df = pd.read_csv("input.csv")

# Split 'Tile Center' by whitespace into Longitude and Latitude
# Tile Center format: "<Longitude> <Latitude>"
tile_coords = df["Tile Center"].astype(str).str.strip().str.split(n=1, expand=True)

# Build output dataframe with required mapping and column headers
output_df = pd.DataFrame(
    {
        "TlesID": df["location"],
        "Longitude": tile_coords[0],
        "Latitude": tile_coords[1],
        "A RSRP (dBm)": df["Signal strength (dBm)"],
        "A Samples": df["Samples"],
        "J RSRP (dBm)": df["Reliance Jio"],
        "J Samples": df["Reliance Jio Samples"],
        "Avg RSRP": df["All Cellular Networks Signal strength (dBm)"],
    }
)

# Export to output CSV
output_df.to_csv("output.csv", index=False)