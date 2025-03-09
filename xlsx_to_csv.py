import os
import pandas as pd
import glob
from pathlib import Path

def extract_thermal_data(input_dir='.', output_dir='cleaned_csv_output'):
	"""
	Extract thermal storage data from Excel files with complex structures.
	
	Args:
		input_dir (str): Directory containing Excel files
		output_dir (str): Directory to save cleaned CSV files
	"""
	# Convert to absolute paths
	input_dir = os.path.abspath(input_dir)
	output_dir = os.path.abspath(output_dir)
	
	# Create output directory if it doesn't exist
	os.makedirs(output_dir, exist_ok=True)
	
	# Find all xlsx files in the input directory
	pattern = os.path.join(input_dir, "*.xlsx")
	xlsx_files = glob.glob(pattern)
	
	if not xlsx_files:
		print(f"No Excel files found in {input_dir}")
		print(f"Looking for pattern: {pattern}")
		return
	
	print(f"Found {len(xlsx_files)} Excel files")
	
	for xlsx_file in xlsx_files:
		file_name = Path(xlsx_file).stem
		print(f"Processing file: {xlsx_file}")
		
		# Load the Excel file
		try:
			xl = pd.ExcelFile(xlsx_file)
		except Exception as e:
			print(f"  Error opening file: {str(e)}")
			continue
		
		# Process each sheet
		for sheet_name in xl.sheet_names:
			print(f"  Processing sheet: {sheet_name}")
			
			try:
				# Read the entire sheet without headers first to search for data
				df_raw = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
				
				# Look for the "Time (s)" or similar column
				time_row_idx = None
				time_col_idx = None
				
				# Search for common headers in the time column
				search_terms = ["Time (s)", "Time(s)", "Time (sec)", "Time(sec)"]
				
				for i in range(len(df_raw)):
					for j in range(len(df_raw.columns)):
						cell_value = str(df_raw.iloc[i, j]).strip() if not pd.isna(df_raw.iloc[i, j]) else ""
						if any(term.lower() in cell_value.lower() for term in search_terms):
							time_row_idx = i
							time_col_idx = j
							break
					if time_row_idx is not None:
						break
				
				# If we couldn't find the time column, try an alternative approach
				if time_row_idx is None:
					# Look for numeric time values in the beginning of the data section
					for i in range(len(df_raw)):
						for j in range(len(df_raw.columns)):
							try:
								# Check if this cell and the next few cells look like times
								if (pd.to_numeric(df_raw.iloc[i, j], errors='coerce') > 0 and
									pd.to_numeric(df_raw.iloc[i+1, j], errors='coerce') > 0 and
									pd.to_numeric(df_raw.iloc[i+2, j], errors='coerce') > 0):
									
									# Check if column to the right has temperatures (usually around 100°C)
									if (pd.to_numeric(df_raw.iloc[i, j+1], errors='coerce') >= 90 and
										pd.to_numeric(df_raw.iloc[i, j+1], errors='coerce') <= 200):
										
										time_row_idx = i-1  # Use row above as header
										time_col_idx = j
										break
							except (IndexError, ValueError):
								continue
						if time_row_idx is not None:
							break
				
				if time_row_idx is None:
					print(f"    Warning: Could not find time data in sheet {sheet_name}. Skipping.")
					continue
				
				# Now read the data with the identified header row
				# If the header row is -1, we'll create our own headers
				if time_row_idx == -1:
					df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
					df.columns = ['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)'] + [f'Unnamed_{i}' for i in range(4, len(df.columns))]
				else:
					df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=time_row_idx)
				
				# Clean column names
				df.columns = [str(col).strip() for col in df.columns]
				
				# Find our target columns
				time_col = None
				tmin_col = None
				tmax_col = None
				tave_col = None
				
				for col in df.columns:
					col_lower = str(col).lower()
					if 'time' in col_lower and any(unit in col_lower for unit in ['s)', 'sec)', 'second']):
						time_col = col
					elif any(term in col_lower for term in ['t_min', 'tmin', 'min']) and '(' in col_lower:
						tmin_col = col
					elif any(term in col_lower for term in ['t_max', 'tmax', 'max']) and '(' in col_lower:
						tmax_col = col
					elif any(term in col_lower for term in ['t_ave', 'tave', 'ave', 'average']) and '(' in col_lower:
						tave_col = col
				
				# If we still can't find the columns, use positional approach based on the time column
				if time_col is not None and (tmin_col is None or tmax_col is None or tave_col is None):
					time_col_idx = df.columns.get_loc(time_col)
					
					# Assume the temperature columns are next to the time column
					if time_col_idx + 1 < len(df.columns) and tmin_col is None:
						tmin_col = df.columns[time_col_idx + 1]
					if time_col_idx + 2 < len(df.columns) and tmax_col is None:
						tmax_col = df.columns[time_col_idx + 2]
					if time_col_idx + 3 < len(df.columns) and tave_col is None:
						tave_col = df.columns[time_col_idx + 3]
				
				# If we still can't find columns, check for column indexes from the raw data
				if time_col is None and time_col_idx is not None:
					# Create a new dataframe with just the columns we need
					col_data = []
					
					# Start from the row after the header
					data_start_row = time_row_idx + 1
					
					# Extract time column
					time_data = []
					for i in range(data_start_row, len(df_raw)):
						val = df_raw.iloc[i, time_col_idx]
						if pd.isna(val):
							break
						try:
							time_data.append(float(val))
						except (ValueError, TypeError):
							break
					
					# Extract temperature columns (assuming they are the next 3 columns)
					tmin_data = []
					tmax_data = []
					tave_data = []
					
					for i in range(data_start_row, data_start_row + len(time_data)):
						# T_min
						if time_col_idx + 1 < len(df_raw.columns):
							try:
								tmin_data.append(float(df_raw.iloc[i, time_col_idx + 1]))
							except (ValueError, TypeError):
								tmin_data.append(None)
						
						# T_max
						if time_col_idx + 2 < len(df_raw.columns):
							try:
								tmax_data.append(float(df_raw.iloc[i, time_col_idx + 2]))
							except (ValueError, TypeError):
								tmax_data.append(None)
						
						# T_ave
						if time_col_idx + 3 < len(df_raw.columns):
							try:
								tave_data.append(float(df_raw.iloc[i, time_col_idx + 3]))
							except (ValueError, TypeError):
								tave_data.append(None)
					
					# Create the cleaned dataframe
					cleaned_df = pd.DataFrame({
						'Time (s)': time_data,
						'T_min (C)': tmin_data if tmin_data else [None] * len(time_data),
						'T_max (C)': tmax_data if tmax_data else [None] * len(time_data),
						'T_ave (C)': tave_data if tave_data else [None] * len(time_data)
					})
					
					# Add Thermal input column
					cleaned_df['Thermal_Input_C'] = cleaned_df['T_max (C)']
					
				else:
					if not all([time_col, tmin_col, tmax_col, tave_col]):
						print(f"    Warning: Could not find all required columns in sheet {sheet_name}.")
						print(f"    Found: Time={time_col}, T_min={tmin_col}, T_max={tmax_col}, T_ave={tave_col}")
						print(f"    Available columns: {list(df.columns)}")
						continue
					
					# Extract relevant data
					# Skip any rows that don't have numeric time values
					df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
					df = df.dropna(subset=[time_col])
					
					# Convert temperature values to numeric
					df[tmin_col] = pd.to_numeric(df[tmin_col], errors='coerce')
					df[tmax_col] = pd.to_numeric(df[tmax_col], errors='coerce')
					df[tave_col] = pd.to_numeric(df[tave_col], errors='coerce')
					
					# Create the cleaned dataframe
					cleaned_df = pd.DataFrame({
						'Time (s)': df[time_col],
						'T_min (C)': df[tmin_col],
						'T_max (C)': df[tmax_col],
						'T_ave (C)': df[tave_col],
						'Thermal_Input_C': df[tmax_col]  # Use T_max as Thermal input
					})
				
				# Drop any remaining NaN rows
				cleaned_df = cleaned_df.dropna()
				
				# Sort by time
				cleaned_df = cleaned_df.sort_values('Time (s)')
				
				# Check if we have data
				if len(cleaned_df) == 0:
					print(f"    Warning: No valid data found in sheet {sheet_name} after processing.")
					continue
				
				# Create clean output filename
				clean_sheet_name = ''.join(c if c.isalnum() or c in ' _-' else '_' for c in sheet_name)
				output_file = f"{file_name}_{clean_sheet_name}_cleaned.csv"
				output_path = os.path.join(output_dir, output_file)
				
				# Write to CSV
				cleaned_df.to_csv(output_path, index=False)
				print(f"    Successfully extracted {len(cleaned_df)} data points to: {output_path}")
				
			except Exception as e:
				print(f"    Error processing sheet {sheet_name}: {str(e)}")
				import traceback
				traceback.print_exc()
	
	print("Extraction complete!")

if __name__ == "__main__":
	import argparse
	
	parser = argparse.ArgumentParser(description="Extract thermal storage data from Excel files")
	parser.add_argument("--input", "-i", default=".", help="Input directory containing Excel files")
	parser.add_argument("--output", "-o", default="cleaned_csv_output", help="Output directory for cleaned CSV files")
	
	args = parser.parse_args()
	
	extract_thermal_data(args.input, args.output)