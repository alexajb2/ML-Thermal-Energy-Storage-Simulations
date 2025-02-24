import os

folder_path = r"C:\Users\lolly\OneDrive\Desktop\Projects\ThermalAI\data\train"

# List all filenames ending with .csv
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

bruh = []

print("CSV files in the folder:")
for filename in csv_files:
    print(filename)  # Just to see what the filenames are

    # Remove the .csv extension
    base_name = filename[:-4]  # everything except last 4 chars ('.csv')

    # Check if the last 3 chars are digits
    if len(base_name) >= 3 and base_name[-3:].isdigit():
        bruh.append(int(base_name[-3:]))
    # If not, check if the last 2 chars are digits
    elif len(base_name) >= 2 and base_name[-2:].isdigit():
        bruh.append(int(base_name[-2:]))
    else:
        # If neither condition is met, we do nothing ("move on").
        pass

# Sort if you like
bruh_sorted = sorted(bruh)

print("Extracted digits:", bruh)
print("Sorted digits:", bruh_sorted)

missing_digits = []

if bruh_sorted:  # Make sure the list isn't empty
    min_digit = bruh_sorted[0]
    max_digit = bruh_sorted[-1]
    
    for num in range(min_digit, max_digit + 1):
        if num not in bruh_sorted:
            missing_digits.append(num)


print("Missing file numbers:", missing_digits)

print("Number of missing data files:",len(missing_digits))