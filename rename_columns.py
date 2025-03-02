import os
import pandas as pd

# Define the folder containing the train CSV files.
train_folder = r"data/train"  # Update this path if necessary.

# Loop through all CSV files in the train folder.
for file_name in os.listdir(train_folder):
    if file_name.endswith(".csv"):
        file_path = os.path.join(train_folder, file_name)
        df = pd.read_csv(file_path)
        
        # Check if the last column is "Thermal Input (C)".
        if df.columns[-1] == "Thermal Input (C)":
            df.rename(columns={"Thermal Input (C)": "Thermal_Input_C"}, inplace=True)
            df.to_csv(file_path, index=False)
            print(f"Renamed column in: {file_name}")
        else:
            print(f"No renaming needed in: {file_name}")
