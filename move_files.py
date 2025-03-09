import os
import shutil
import random

# Define the source and target directories
source_dir = 'data/csvs_interpolated'
training_dir = 'data/train'
validation_dir = 'data/validation'

# Create target directories if they don't exist
os.makedirs(training_dir, exist_ok=True)
os.makedirs(validation_dir, exist_ok=True)

# Get a list of all files in the source directory
files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Calculate the number of files for training (80%) and validation (20%)
total_files = len(files)
train_size = int(0.8 * total_files)
val_size = total_files - train_size

# Randomly shuffle the list of files
random.shuffle(files)

# Split the files into training and validation sets
train_files = files[:train_size]
val_files = files[train_size:]

# Copy files to the training directory
for file in train_files:
    shutil.copy(os.path.join(source_dir, file), training_dir)

# Copy files to the validation directory
for file in val_files:
    shutil.copy(os.path.join(source_dir, file), validation_dir)