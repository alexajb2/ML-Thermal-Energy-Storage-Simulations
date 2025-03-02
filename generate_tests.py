import pandas as pd
import numpy as np

# File paths (adjust as needed)
excel_file = r"C:\Users\rcbee\OneDrive\Desktop\Projects\ml_research_thermal_storage\Testing cases for temperature condition.xlsx"
sample_csv = r"C:\Users\rcbee\OneDrive\Desktop\Projects\ml_research_thermal_storage\data\train\Case1_TrialData_Sample1.csv"

# -----------------------------------------------------------
# Step 1: Read the sample CSV to get the master time vector.
sample_df = pd.read_csv(sample_csv)
time_column_name = "Time (s)"
time_vector = sample_df[time_column_name].values

# -----------------------------------------------------------
# Step 2: Read the Excel file.
# In this approach we use header=0 so that the first row becomes the column names.
df = pd.read_excel(excel_file, header=0)

# For debugging, print the column names:
print("Excel columns:", df.columns.tolist())

# -----------------------------------------------------------
# Step 3: Group columns into test cases.
# Based on your printed output, we assume each test case uses two columns:
# one for time and one for temperature.
# It appears that the file might include a blank column between cases.
# Here we assume groups of 3 columns: [Time, Temperature, (blank)].
# We will loop over these groups.
cols = df.columns.tolist()
num_groups = len(cols) // 3  # adjust if your file structure differs

for i in range(num_groups):
    # Get indices for the time and temperature columns.
    time_idx = i * 3      # first column in the group
    temp_idx = i * 3 + 1  # second column in the group

    # Extract the two columns.
    case_data = df.iloc[:, [time_idx, temp_idx]].copy()

    # Sometimes the first row of data is a duplicate header.
    # Remove the row where the time column equals the header name.
    header_label = case_data.columns[0]  # this should be "Time (sec)"
    case_data = case_data[case_data[header_label] != header_label]

    # Rename the columns to standard names.
    case_data.columns = ["Time (sec)", "Temperature (C)"]

    # Convert columns to numeric (coercing errors to NaN).
    case_data["Time (sec)"] = pd.to_numeric(case_data["Time (sec)"], errors="coerce")
    case_data["Temperature (C)"] = pd.to_numeric(case_data["Temperature (C)"], errors="coerce")

    # Drop any rows where Time is NaN.
    case_data = case_data.dropna(subset=["Time (sec)"])

    # Extract time and temperature arrays.
    case_time = case_data["Time (sec)"].values
    case_temp = case_data["Temperature (C)"].values

    # Perform linear interpolation on temperature values.
    interp_temp = np.interp(time_vector, case_time, case_temp)

    # Create a DataFrame with the master time vector and the interpolated thermal input.
    output_df = pd.DataFrame({
        "Time (s)": time_vector,
        "Thermal Input C": interp_temp
    })

    # Create an output filename for this test case.
    out_file_name = f"Case{i+1}.csv"
    output_df.to_csv(out_file_name, index=False)
    print(f"Created file: {out_file_name}")
