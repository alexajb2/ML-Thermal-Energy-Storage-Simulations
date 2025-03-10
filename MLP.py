import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from sklearn.metrics import mean_absolute_percentage_error, r2_score

#########################################
# Dataset Definition
#########################################

class ThermalRegressionDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        
        # Columns to scale (including features and extra columns for inverse transform)
        columns_for_scaling = ["Time (s)", "T_min (C)", "T_max (C)", "T_ave (C)", "Thermal_Input (C)"]
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler
        # Scale the dataframe
        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])
        
        # Create independent samples for regression:
        # Use each row (except the last) as input and the next row's T_min and T_ave as target.
        self.X = df[["Time (s)", "T_min (C)", "T_ave (C)", "Thermal_Input (C)"]].values[:-1]
        self.Y = df[["T_min (C)", "T_ave (C)"]].values[1:]
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        
        # Store the original (unscaled) time values corresponding to the target rows for plotting.
        df_original = pd.read_csv(csv_file)
        # For targets, use rows 1 to end
        self.full_time_original = df_original["Time (s)"].values[1:]
        # Also store the original thermal input for the input rows (row 0 to n-1)
        self.full_time_input_original = df_original["Time (s)"].values[:-1]
        self.full_thermal_input_original = df_original["Thermal_Input (C)"].values[:-1]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#########################################
# Model Definition
#########################################

class ThermalRegressionModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(ThermalRegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.fc(x)

#########################################
# Training Function
#########################################

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_paths = glob.glob(os.path.join(script_dir, "data", "train", "*.csv"))
    val_paths = glob.glob(os.path.join(script_dir, "data", "validation", "*.csv"))
    test_paths = glob.glob(os.path.join(script_dir, "data", "testing", "*.csv"))
    
    # Fit the scaler on all training files.
    train_dfs = [pd.read_csv(path) for path in train_paths]
    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    columns_for_scaling = ["Time (s)", "T_min (C)", "T_max (C)", "T_ave (C)", "Thermal_Input (C)"]
    scaler = MinMaxScaler()
    scaler.fit(combined_train_df[columns_for_scaling])
    
    train_datasets = [ThermalRegressionDataset(path, scaler=scaler) for path in train_paths]
    val_datasets = [ThermalRegressionDataset(path, scaler=scaler) for path in val_paths]
    test_datasets = [ThermalRegressionDataset(path, scaler=scaler) for path in test_paths]
    
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)
    
    train_dataloader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(combined_val_dataset, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalRegressionModel(input_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * inputs.size(0)
        avg_train_loss = total_train_loss / len(combined_train_dataset)
        
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = model(inputs)
                loss = criterion(predictions, targets)
                total_val_loss += loss.item() * inputs.size(0)
        avg_val_loss = total_val_loss / len(combined_val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, "best_regression_model.pth"))
            
    model.load_state_dict(torch.load(os.path.join(script_dir, "best_regression_model.pth")))
    return model, test_datasets, scaler

#########################################
# Testing Function with Plot (Including Thermal Input)
#########################################

def test_model(model, test_datasets, scaler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # The scaler was fitted on 5 columns in the following order:
    # [Time (s), T_min (C), T_max (C), T_ave (C), Thermal_Input (C)]
    # We'll invert scaling for T_min and T_ave using the stored data_min_ and data_max_.
    t_min_range = scaler.data_max_[1] - scaler.data_min_[1]
    t_min_min = scaler.data_min_[1]
    t_ave_range = scaler.data_max_[3] - scaler.data_min_[3]
    t_ave_min = scaler.data_min_[3]
    
    for test_dataset in test_datasets:
        dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        predictions_list = []
        actual_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions_list.append(outputs.cpu().numpy())
                actual_list.append(targets.cpu().numpy())
        predictions = np.concatenate(predictions_list, axis=0)
        actual = np.concatenate(actual_list, axis=0)
        
        # Invert scaling for T_min and T_ave
        pred_t_min_original = predictions[:, 0] * t_min_range + t_min_min
        pred_t_ave_original = predictions[:, 1] * t_ave_range + t_ave_min
        actual_t_min_original = actual[:, 0] * t_min_range + t_min_min
        actual_t_ave_original = actual[:, 1] * t_ave_range + t_ave_min
        
        # Retrieve the original time values for targets (for T_min and T_ave)
        time_original = test_dataset.full_time_original
        
        # Compute metrics for T_min and T_ave
        mape_t_min = mean_absolute_percentage_error(actual_t_min_original, pred_t_min_original)
        r2_t_min = r2_score(actual_t_min_original, pred_t_min_original)
        mape_t_ave = mean_absolute_percentage_error(actual_t_ave_original, pred_t_ave_original)
        r2_t_ave = r2_score(actual_t_ave_original, pred_t_ave_original)
        
        print(f"Results for {test_dataset.file_name}:")
        print(f"  T_min:   MAPE = {mape_t_min:.4f}, R² = {r2_t_min:.4f}")
        print(f"  T_ave:   MAPE = {mape_t_ave:.4f}, R² = {r2_t_ave:.4f}")
        
        # Create a single plot with two y-axes:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Plot temperatures on the left y-axis
        ax1.plot(time_original, actual_t_min_original, label="T_min (Actual)", color="blue")
        ax1.plot(time_original, pred_t_min_original, label="T_min (Predicted)", linestyle="dashed", color="blue")
        ax1.plot(time_original, actual_t_ave_original, label="T_ave (Actual)", color="green")
        ax1.plot(time_original, pred_t_ave_original, label="T_ave (Predicted)", linestyle="dashed", color="green")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Temperature (C)")
        
        # Create a second y-axis for thermal input
        ax2 = ax1.twinx()
        ax2.plot(test_dataset.full_time_input_original, test_dataset.full_thermal_input_original, 
                 label="Thermal Input", color="purple", alpha=0.7)
        ax2.set_ylabel("Thermal Input (C)")
        
        # Combine legends from both y-axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        
        plt.title(f"Regression Results and Thermal Input for {test_dataset.file_name}")
        plt.show()

#########################################
# Main
#########################################

if __name__ == "__main__":
    trained_model, test_datasets, scaler = train_model()
    test_model(trained_model, test_datasets, scaler)
