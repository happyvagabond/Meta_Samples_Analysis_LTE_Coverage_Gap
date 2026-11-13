import pandas as pd
import glob

# Specify the path to your CSV files
csv_files = glob.glob("*.csv")  # Adjust the path if needed

# Initialize an empty DataFrame to store combined data
combined_data = pd.DataFrame()

for file in csv_files:
    # Read the CSV file, treating the second row as the header
    data = pd.read_csv(file, header=1)
    
    # Append the data to the combined DataFrame
    combined_data = pd.concat([combined_data, data], ignore_index=True)

# Remove duplicate rows
combined_data = combined_data.drop_duplicates()

# Save the combined data to a new CSV file
combined_data.to_csv("combined_output.csv", index=False)

print("CSV files combined successfully and duplicates removed!")
