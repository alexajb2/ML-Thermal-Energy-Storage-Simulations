import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_file(file_path):
    """Reads a CSV file and plots the training data."""
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    # Expected column names. Adjust these if your CSV uses different headers.
    expected_columns = ["Time (s)", "T_min (C)", "T_max (C)", "T_ave (C)", "Thermal_Input_C"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        print(f"Skipping {os.path.basename(file_path)} because missing columns: {missing}")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["Time (s)"], df["T_min (C)"], label="T_min (C)")
    plt.plot(df["Time (s)"], df["T_max (C)"], label="T_max (C)")
    plt.plot(df["Time (s)"], df["T_ave (C)"], label="T_ave (C)")
    plt.plot(df["Time (s)"], df["Thermal_Input_C"], label="Thermal Input (C)", linestyle="--", color="black")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (°C)")
    plt.title(f"Training Data Plot: {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_folder(folder_path):
    """Iterates over all CSV files in the given folder and plots them."""
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a directory.")
        return

    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".csv")]
    if not csv_files:
        print("No CSV files found in the folder.")
        return

    for csv_file in csv_files:
        print(f"Plotting {csv_file} ...")
        plot_csv_file(csv_file)

if __name__ == "__main__":

    folder_path = f"C:\\Users\\lolly\\OneDrive\\Desktop\\Projects\\ml_research_thermal_storage\\data\\train"
    plot_folder(folder_path)
