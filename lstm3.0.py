import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_percentage_error, r2_score

# =============================================================================
# ThermalDataset (for training):
# Loads CSV data with columns:
#   Time (s), T_min (C), T_max (C), T_ave (C), Thermal_Input_C
# Scales the temperature columns and creates a single sequence.
# The target Y is the shifted [T_min, T_ave] sequence.
# =============================================================================
class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()
        # Scale the four temperature-related columns.
        self.scaler.fit(df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']])
        df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']] = self.scaler.transform(
            df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']]
        )
        data = df[['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']].values
        self.X = [data[:-1]]
        target = df[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values[1:]
        self.Y = [target[:, [0, 2]]]  # T_min and T_ave targets.
        self.time_values = [df['Time (s)'].values[1:]]
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        print(f"Loaded {len(self.X)} training sequence from {csv_file}. Input shape: {self.X[0].shape}, Target shape: {self.Y[0].shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]

# =============================================================================
# TestThermalDataset (for testing):
# Loads a test CSV with only two columns: Time (s) and Thermal Input C.
# Since the model expects 5 input features, we create dummy features by copying
# the Thermal Input into the missing T_min, T_max, T_ave columns.
#
# A dummy scaler is created (by fitting on the Thermal Input repeated) so that
# we can reverse-transform predictions for plotting.
# =============================================================================
class TestThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Create a scaler by faking 4 columns from Thermal Input.
        self.scaler = MinMaxScaler()
        dummy = np.column_stack([df["Thermal Input C"]]*4)
        self.scaler.fit(dummy)
        # Transform the Thermal Input column by creating a 4-column array and taking the first column.
        thermal_scaled = self.scaler.transform(np.column_stack([df["Thermal Input C"]]*4))[:, 0]
        df["Thermal Input C"] = thermal_scaled
        
        self.time_values = df["Time (s)"].values
        thermal = df["Thermal Input C"].values
        # Create input: [Time, T_min, T_max, T_ave, Thermal_Input_C]
        # Fill the missing features with the thermal input.
        data = np.column_stack((self.time_values, thermal, thermal, thermal, thermal))
        # The entire file is one continuous sequence.
        self.X = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, 5)
        # Save the actual (scaled) thermal input for plotting.
        self.actual = thermal
        print(f"Loaded test file {csv_file}. Sequence length: {self.X.shape[1]}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Return X, no target, plus time values and actual thermal input.
        return self.X, None, self.time_values, self.actual


# =============================================================================
# LSTMModel:
# A simplified LSTM that predicts two values (T_min and T_ave) per time step.
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=48, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# =============================================================================
# weighted_loss:
# Custom L1 loss on predictions (only for T_min and T_ave).
# =============================================================================
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0])):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets)
    return torch.mean(loss * weights)

# =============================================================================
# train_model:
# Now loads all CSV files in the "data/train" folder for training.
# =============================================================================
def train_model():
    import os
    from torch.utils.data import TensorDataset, DataLoader

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "data", "train")
    # List all CSV files in the train folder.
    train_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    train_paths = [os.path.join(train_dir, f) for f in train_files]
    print(f"Training on {len(train_paths)} files.")

    # Instead of using ConcatDataset, we load each file (which is a distinct sequence)
    # and then build a dataset where each file is one training example.
    all_X = []
    all_Y = []
    all_time = []
    for path in train_paths:
        ds = ThermalDataset(path)
        # Each ThermalDataset is built to hold one training sequence.
        X, Y, time_vals = ds[0]  # Extract the single training example.
        all_X.append(X)
        all_Y.append(Y)
        all_time.append(time_vals)
    
    # Stack all the training examples into tensors.
    all_X = torch.stack(all_X)         # Shape: (num_files, seq_len, 5)
    all_Y = torch.stack(all_Y)         # Shape: (num_files, seq_len, 2)
    all_time = torch.tensor(np.stack(all_time))  # Shape: (num_files, seq_len)
    
    # Create a dataset where each file is a distinct example.
    train_dataset = TensorDataset(all_X, all_Y, all_time)
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, pin_memory=True)
    # train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 5000
    patience = 300
    best_loss = float("inf")
    counter = 0
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = weighted_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        # scheduler.step(epoch_loss)
    
        # if epoch_loss < best_loss:
        #     best_loss = epoch_loss
        #     counter = 0
        # else:
        #     counter += 1
        #     if counter >= patience:
        #         print(f"Early stopping at epoch {epoch+1}. No improvement in {patience} epochs.")
        #         break
    
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return model

# =============================================================================
# test_model:
# Loads test files (Case1.csv to Case5.csv) from the "data/testing" folder.
# For each file, obtains model predictions (T_min and T_ave) and plots them
# alongside the actual Thermal Input (from the test file) for comparison.
# =============================================================================
def test_model(model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "data", "testing")
    # Test files: Case1.csv to Case5.csv
    test_files = [f"Case{i}.csv" for i in range(1, 6)]
    test_paths = [os.path.join(test_dir, f) for f in test_files]
    test_datasets = [TestThermalDataset(path) for path in test_paths]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_figures = []

    for i, test_dataset in enumerate(test_datasets):
        dataset_name = test_files[i]
        print(f"Testing on dataset: {dataset_name}")
        try:
            # For the test dataset, we get: X, None, time_values, and actual thermal input.
            test_input, _, time_values, actual = test_dataset[0]
            test_input = test_input.to(device)  # shape: (1, seq_len, 5)
            with torch.no_grad():
                predicted_sequence = model(test_input).cpu().numpy()  # shape: (1, seq_len, 2)
            t_min_pred = predicted_sequence[0, :, 0]
            t_ave_pred = predicted_sequence[0, :, 1]
            # For T_max, we simply use the dummy input (column index 2)
            t_max_pred = test_input[0, :, 2].cpu().numpy()

            # Inverse scaling using the test dataset's scaler.
            scaler = test_dataset.scaler
            t_min_pred_original = scaler.inverse_transform(
                np.column_stack((t_min_pred, t_min_pred, t_min_pred, t_min_pred))
            )[:, 0]
            t_ave_pred_original = scaler.inverse_transform(
                np.column_stack((t_ave_pred, t_ave_pred, t_ave_pred, t_ave_pred))
            )[:, 2]
            t_max_pred_original = scaler.inverse_transform(
                np.column_stack((t_max_pred, t_max_pred, t_max_pred, t_max_pred))
            )[:, 1]
            actual_original = scaler.inverse_transform(
                np.column_stack((actual, actual, actual, actual))
            )[:, 0]

            # Plot Thermal Input (actual) and predicted T_min and T_ave.
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_values, actual_original, label="Thermal Input (Actual)", color="black")
            ax.plot(time_values, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")
            ax.plot(time_values, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title(f"Test Case: {dataset_name}")
            ax.legend()
            all_figures.append(fig)
        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")

    for fig in all_figures:
        plt.figure(fig.number)
    plt.show()

# =============================================================================
# Main execution: Train the model and then evaluate on test datasets.
# =============================================================================
if __name__ == "__main__":
    model = train_model()
    test_model(model)
