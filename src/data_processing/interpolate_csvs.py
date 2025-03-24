import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import glob
import argparse

def interpolate_csv(input_file, output_file):
    """
    Interpolate a CSV file to have 100 rows with uniform time steps from 0 to 14,400 seconds.
    
    Parameters:
    - input_file (str): Path to the input CSV file.
    - output_file (str): Path where the interpolated CSV will be saved.
    """
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Verify required columns
    required_columns = ['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input (C)']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {input_file}")
    
    # Sort by time to ensure ascending order
    df = df.sort_values('Time (s)')
    
    # Extract original data
    original_time = df['Time (s)'].values
    t_min = df['T_min (C)'].values
    t_max = df['T_max (C)'].values
    t_ave = df['T_ave (C)'].values
    thermal_input = df['Thermal_Input (C)'].values
    
    # Define new time array: 100 points from 0 to 14,400
    new_time = np.linspace(0, 14400, 100)
    
    # Create interpolation functions (linear)
    interp_t_min = interp1d(original_time, t_min, kind='linear')
    interp_t_max = interp1d(original_time, t_max, kind='linear')
    interp_t_ave = interp1d(original_time, t_ave, kind='linear')
    interp_thermal = interp1d(original_time, thermal_input, kind='linear')
    
    # Interpolate temperature values at new time points
    new_t_min = interp_t_min(new_time)
    new_t_max = interp_t_max(new_time)
    new_t_ave = interp_t_ave(new_time)
    new_thermal = interp_thermal(new_time)
    
    # Create new DataFrame
    new_df = pd.DataFrame({
        'Time (s)': new_time,
        'T_min (C)': new_t_min,
        'T_max (C)': new_t_max,
        'T_ave (C)': new_t_ave,
        'Thermal_Input (C)': new_thermal
    })
    
    # Save to CSV without the index column
    new_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Interpolate thermal data CSV files to 100 rows with uniform time steps.")
    parser.add_argument("--input", "-i", default=".", help="Input directory containing CSV files (default: current directory)")
    parser.add_argument("--output", "-o", default="interpolated_csvs", help="Output directory for interpolated CSV files (default: interpolated_csvs)")
    
    args = parser.parse_args()
    
    # Convert paths to absolute to avoid relative path issues
    input_dir = os.path.abspath(args.input)
    output_dir = os.path.abspath(args.output)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all CSV files in the input directory
    pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(pattern)
    
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        print(f"Looking for pattern: {pattern}")
    else:
        print(f"Found {len(csv_files)} CSV files")
        for csv_file in csv_files:
            # Construct output file path with the same file name
            file_name = os.path.basename(csv_file)
            output_file = os.path.join(output_dir, file_name)
            print(f"Processing {csv_file} -> {output_file}")
            try:
                interpolate_csv(csv_file, output_file)
                print(f"  Successfully interpolated to {output_file}")
            except Exception as e:
                print(f"  Error processing {csv_file}: {str(e)}")