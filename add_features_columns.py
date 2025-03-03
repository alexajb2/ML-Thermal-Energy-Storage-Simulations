import pandas as pd
import glob
import os

# Define the folder containing the CSV files
folder_path = "data/testing"  # Ensure this folder path is correct

# Use glob to get all CSV files in the folder
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# Loop through each CSV file
for file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file)
    
    # Add the constant columns with the specified values
    df["input_type"] = "temperature"
    df["ambient_temperature"] = 300
    df["convection_coefficient"] = 20
    df["shape"] = "cylindrical"
    df["length"] = 20
    df["height"] = 3
    df["diameter_of_hx"] = 3
    df["density"] = 200
    df["thermal_conductivity"] = 2
    df["specific_heat"] = 400
    
    # Save the updated DataFrame back to CSV (overwriting the original file)
    df.to_csv(file, index=False)
    
    print(f"Updated file: {file}")
