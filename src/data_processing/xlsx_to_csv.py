import os
import pandas as pd
import glob
from pathlib import Path

def extract_thermal_data(input_dir='.', output_dir='cleaned_csv_output'):
	"""
	Extract thermal storage data from Excel files with complex structures, adding an initial row at time=0.
	
	Args:
		input_dir (str): Directory containing Excel files
		output_dir (str): Directory to save cleaned CSV files
	"""
	# Convert to absolute paths
	input_dir = os.path.abspath(input_dir)
	output_dir = os.path.abspath(output_dir)
	
	# Create output directory if it doesn’t exist
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
		
		try:
			xl = pd.ExcelFile(xlsx_file)
		except Exception as e:
			print(f"  Error opening file: {str(e)}")
			continue
		
		for sheet_name in xl.sheet_names:
			print(f"  Processing sheet: {sheet_name}")
			
			try:
				# Read the entire sheet without headers to search for data
				df_raw = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
				
				# --- Step 1: Extract Initial Temperature ---
				initial_temp = None
				# Limit search to first 10 rows, assuming Thermal input is near the top
				for i in range(min(10, len(df_raw))):
					for j in range(len(df_raw.columns)):
						cell_value = str(df_raw.iloc[i, j]).strip().lower()
						# Look for "Time (sec)" or similar
						if 'time' in cell_value and ('sec' in cell_value or 's)' in cell_value):
							thermal_time_col = j
							# Search for row where time=0
							for k in range(i + 1, len(df_raw)):
								try:
									time_value = float(df_raw.iloc[k, thermal_time_col])
									if time_value == 0:
										# Find "Temp (C)" column in the header row
										for m in range(len(df_raw.columns)):
											header = str(df_raw.iloc[i, m]).strip().lower()
											if 'temp' in header and '(c)' in header:
												initial_temp = float(df_raw.iloc[k, m])
												break
										break  # Exit once time=0 row is found
								except (ValueError, TypeError):
									continue
							if initial_temp is not None:
								break
					if initial_temp is not None:
						break
				
				if initial_temp is None:
					print(f"    Warning: Initial temperature not found in sheet {sheet_name}. Proceeding without it.")
				
				# --- Existing Code: Find Main Data Table ---
				time_row_idx = None
				time_col_idx = None
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
				
				if time_row_idx is None:
					# Alternative approach for numeric time values
					for i in range(len(df_raw)):
						for j in range(len(df_raw.columns)):
							try:
								if (pd.to_numeric(df_raw.iloc[i, j], errors='coerce') > 0 and
									pd.to_numeric(df_raw.iloc[i+1, j], errors='coerce') > 0 and
									pd.to_numeric(df_raw.iloc[i+2, j], errors='coerce') > 0):
									if (pd.to_numeric(df_raw.iloc[i, j+1], errors='coerce') >= 90 and
										pd.to_numeric(df_raw.iloc[i, j+1], errors='coerce') <= 200):
										time_row_idx = i - 1
										time_col_idx = j
										break
							except (IndexError, ValueError):
								continue
						if time_row_idx is not None:
							break
				
				if time_row_idx is None:
					print(f"    Warning: Could not find time data in sheet {sheet_name}. Skipping.")
					continue
				
				if time_row_idx == -1:
					df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=None)
					df.columns = ['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)'] + [f'Unnamed_{i}' for i in range(4, len(df.columns))]
				else:
					df = pd.read_excel(xlsx_file, sheet_name=sheet_name, header=time_row_idx)
				
				df.columns = [str(col).strip() for col in df.columns]
				
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
				
				if time_col is not None and (tmin_col is None or tmax_col is None or tave_col is None):
					time_col_idx = df.columns.get_loc(time_col)
					if time_col_idx + 1 < len(df.columns) and tmin_col is None:
						tmin_col = df.columns[time_col_idx + 1]
					if time_col_idx + 2 < len(df.columns) and tmax_col is None:
						tmax_col = df.columns[time_col_idx + 2]
					if time_col_idx + 3 < len(df.columns) and tave_col is None:
						tave_col = df.columns[time_col_idx + 3]
				
				if time_col is None and time_col_idx is not None:
					data_start_row = time_row_idx + 1
					time_data = []
					for i in range(data_start_row, len(df_raw)):
						val = df_raw.iloc[i, time_col_idx]
						if pd.isna(val):
							break
						try:
							time_data.append(float(val))
						except (ValueError, TypeError):
							break
					
					tmin_data = []
					tmax_data = []
					tave_data = []
					for i in range(data_start_row, data_start_row + len(time_data)):
						if time_col_idx + 1 < len(df_raw.columns):
							try:
								tmin_data.append(float(df_raw.iloc[i, time_col_idx + 1]))
							except (ValueError, TypeError):
								tmin_data.append(None)
						if time_col_idx + 2 < len(df_raw.columns):
							try:
								tmax_data.append(float(df_raw.iloc[i, time_col_idx + 2]))
							except (ValueError, TypeError):
								tmax_data.append(None)
						if time_col_idx + 3 < len(df_raw.columns):
							try:
								tave_data.append(float(df_raw.iloc[i, time_col_idx + 3]))
							except (ValueError, TypeError):
								tave_data.append(None)
					
					cleaned_df = pd.DataFrame({
						'Time (s)': time_data,
						'T_min (C)': tmin_data if tmin_data else [None] * len(time_data),
						'T_max (C)': tmax_data if tmax_data else [None] * len(time_data),
						'T_ave (C)': tave_data if tave_data else [None] * len(time_data)
					})
					cleaned_df['Thermal_Input (C)'] = cleaned_df['T_max (C)']
				else:
					if not all([time_col, tmin_col, tmax_col, tave_col]):
						print(f"    Warning: Could not find all required columns in sheet {sheet_name}.")
						continue
					
					df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
					df = df.dropna(subset=[time_col])
					df[tmin_col] = pd.to_numeric(df[tmin_col], errors='coerce')
					df[tmax_col] = pd.to_numeric(df[tmax_col], errors='coerce')
					df[tave_col] = pd.to_numeric(df[tave_col], errors='coerce')
					
					cleaned_df = pd.DataFrame({
						'Time (s)': df[time_col],
						'T_min (C)': df[tmin_col],
						'T_max (C)': df[tmax_col],
						'T_ave (C)': df[tave_col],
						'Thermal_Input (C)': df[tmax_col]  # Will be overwritten for initial row
					})
				
				# --- Step 2: Add Initial Row ---
				if initial_temp is not None:
					initial_row = pd.DataFrame({
						'Time (s)': [0],
						'T_min (C)': [initial_temp],
						'T_max (C)': [initial_temp],
						'T_ave (C)': [initial_temp],
						'Thermal_Input (C)': [initial_temp]
					})
					cleaned_df = pd.concat([initial_row, cleaned_df], ignore_index=True)
				
				# Sort by time to ensure time=0 is first
				cleaned_df = cleaned_df.sort_values('Time (s)')
				
				# Drop any NaN rows (initial row should be safe as all values are set)
				cleaned_df = cleaned_df.dropna()
				
				if len(cleaned_df) == 0:
					print(f"    Warning: No valid data found in sheet {sheet_name} after processing.")
					continue
				
				# Create output filename
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