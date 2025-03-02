import os
import pandas as pd
import numpy as np

# Define the training folder and the desired fixed length.
train_folder = r"data/train"  # Update path if necessary.
desired_length = 100

# Iterate over all CSV files in the training folder.
for filename in os.listdir(train_folder):
    if not filename.endswith(".csv"):
        continue
    file_path = os.path.join(train_folder, filename)
    df = pd.read_csv(file_path)
    original_length = len(df)
    
    # Check if the file already has the desired number of rows.
    if original_length == desired_length:
        print(f"{filename}: No fix needed. Rows = {original_length}")
        continue
    if original_length < 2:
        print(f"{filename}: Not enough rows to interpolate (rows = {original_length}). Skipping.")
        continue
    
    # Interpolation: Create a new time vector with exactly desired_length points.
    time_original = df["Time (s)"].values
    new_time = np.linspace(time_original[0], time_original[-1], desired_length)
    
    # Build a new DataFrame starting with the new time vector.
    df_fixed = pd.DataFrame({"Time (s)": new_time})
    
    # Interpolate each column (other than Time) at the new time points.
    for col in df.columns:
        if col == "Time (s)":
            continue
        df_fixed[col] = np.interp(new_time, time_original, df[col].values)
    
    # Overwrite the original file with the fixed DataFrame.
    df_fixed.to_csv(file_path, index=False)
    print(f"{filename}: Fixed. Original rows: {original_length}, New rows: {desired_length}")