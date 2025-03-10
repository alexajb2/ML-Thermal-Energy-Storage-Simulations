import os
import pandas as pd
import matplotlib.pyplot as plt

# Specify the folder path (change this if your CSV files are located elsewhere)
folder_path = 'data/train'  # current directory

# Loop through all files in the folder
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        try:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file}: {e}")
            continue

        # Check if the required columns exist in the file
        required_columns = ['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input (C)']
        if not all(col in df.columns for col in required_columns):
            print(f"File {file} does not contain all the required columns. Skipping.")
            continue

        # Create a new figure for each file
        plt.figure(figsize=(10, 6))
        
        # Plot each of the required columns against Time (s)
        plt.plot(df['Time (s)'], df['T_min (C)'], label='T_min (C)')
        plt.plot(df['Time (s)'], df['T_max (C)'], label='T_max (C)')
        plt.plot(df['Time (s)'], df['T_ave (C)'], label='T_ave (C)')
        plt.plot(df['Time (s)'], df['Thermal_Input (C)'], label='Thermal_Input (C)')
        
        # Labeling the plot
        plt.xlabel('Time (s)')
        plt.ylabel('Temperature (C)')
        plt.title(f'Temperature vs Time for {file}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Display the plot
        plt.show()
