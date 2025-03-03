import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_absolute_percentage_error, r2_score


# =============================================================================
# ThermalDataset (for training)
# =============================================================================
class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Scale the four columns used: T_min, T_max, T_ave, and Thermal_Input.
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']])
        df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']] = self.scaler.transform(
            df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']]
        )
        # Reorder the columns so that the input is: [Thermal_Input, T_min, T_max, T_ave]
        input_data = df[['Thermal_Input_C', 'T_min (C)', 'T_max (C)', 'T_ave (C)']].values
        # Targets: next time step's temperatures (all three)
        target_data = df[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values

        # Shift by one time step for training
        input_seq = input_data[:-1]   # shape: (seq_len-1, 4)
        target_seq = target_data[1:]    # shape: (seq_len-1, 3)
        time_seq = df['Time (s)'].values[1:]  # shape: (seq_len-1,)

        # Convert to torch tensors.
        self.X = torch.tensor(input_seq, dtype=torch.float32)
        self.Y = torch.tensor(target_seq, dtype=torch.float32)
        self.time_values = torch.tensor(time_seq, dtype=torch.float32)

        print(f"Loaded training sequence from {csv_file}. "
              f"Input shape: {self.X.shape}, Target shape: {self.Y.shape}")

    # Since each file is one sequence, we return 1.
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.X, self.Y, self.time_values


# =============================================================================
# TestThermalDataset (for testing)
# =============================================================================
class TestThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # For testing, ensure the column name matches.
        if "Thermal Input C" in df.columns:
            df = df.rename(columns={"Thermal Input C": "Thermal_Input_C"})
        # Create a dummy scaler by fitting on the thermal input repeated 4 times.
        self.scaler = MinMaxScaler()
        dummy = np.column_stack([df["Thermal_Input_C"]]*4)
        self.scaler.fit(dummy)
        # Transform the thermal input.
        thermal = self.scaler.transform(np.column_stack([df["Thermal_Input_C"]]*4))[:, 0]
        df["Thermal_Input_C"] = thermal

        self.time_values = df["Time (s)"].values
        # For recursive forecasting, initialize all features with the thermal input.
        data = np.column_stack((thermal, thermal, thermal, thermal))  # order: [Thermal_Input, T_min, T_max, T_ave]
        # The entire file is one continuous sequence.
        self.X = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, 4)
        self.thermal = thermal
        print(f"Loaded test file {csv_file}. Sequence length: {self.X.shape[1]}")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Return X, no target, plus time values and thermal input.
        return self.X, None, self.time_values, self.thermal


# =============================================================================
# LSTM Model
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=48, num_layers=2, output_size=3):
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
# weighted_loss: Custom L1 loss
# =============================================================================
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0, 1.0])):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets)
    return torch.mean(loss * weights)


# =============================================================================
# Training Function
# =============================================================================
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(script_dir, "data", "train")
    train_files = [f for f in os.listdir(train_dir) if f.endswith(".csv")]
    train_paths = [os.path.join(train_dir, f) for f in train_files]
    print(f"Training on {len(train_paths)} files.")

    all_X = []
    all_Y = []
    all_time = []
    for path in train_paths:
        ds = ThermalDataset(path)
        X, Y, time_vals = ds[0]
        all_X.append(X)
        all_Y.append(Y)
        all_time.append(time_vals)

    # Now stack the tensors. Each X has shape (seq_len, 4).
    all_X = torch.stack(all_X)         # Shape: (num_files, seq_len, 4)
    all_Y = torch.stack(all_Y)         # Shape: (num_files, seq_len, 3)
    all_time = torch.stack(all_time)     # Shape: (num_files, seq_len)

    train_dataset = TensorDataset(all_X, all_Y, all_time)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 500

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = weighted_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model


# =============================================================================
# Testing Function (Recursive Prediction)
# =============================================================================
def test_model(model):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(script_dir, "data", "testing")
    test_files = [f"Case{i}.csv" for i in range(1, 7)]
    test_paths = [os.path.join(test_dir, f) for f in test_files]
    test_datasets = [TestThermalDataset(path) for path in test_paths]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_figures = []

    for i, test_dataset in enumerate(test_datasets):
        dataset_name = test_files[i]
        print(f"Testing on dataset: {dataset_name}")
        try:
            test_input, _, time_values, thermal = test_dataset[0]
            seq_len = test_input.shape[1]
            predictions = []
            current_input = np.array([thermal[0], thermal[0], thermal[0], thermal[0]])
            for t in range(seq_len - 1):
                current_input[0] = thermal[t]
                inp_tensor = torch.tensor(current_input, dtype=torch.float32).view(1, 1, -1).to(device)
                with torch.no_grad():
                    pred = model(inp_tensor)
                pred = pred.cpu().numpy().flatten()
                predictions.append(pred)
                current_input = np.array([thermal[t+1], pred[0], pred[1], pred[2]])
            predictions = np.array(predictions)

            t_min_pred_original = test_dataset.scaler.inverse_transform(
                np.column_stack((predictions[:, 0], predictions[:, 0], predictions[:, 0], predictions[:, 0]))
            )[:, 0]
            t_max_pred_original = test_dataset.scaler.inverse_transform(
                np.column_stack((predictions[:, 1], predictions[:, 1], predictions[:, 1], predictions[:, 1]))
            )[:, 1]
            t_ave_pred_original = test_dataset.scaler.inverse_transform(
                np.column_stack((predictions[:, 2], predictions[:, 2], predictions[:, 2], predictions[:, 2]))
            )[:, 2]
            actual_original = test_dataset.scaler.inverse_transform(
                np.column_stack((thermal, thermal, thermal, thermal))
            )[:, 3]
            time_vals = time_values[1:]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(time_values, actual_original, label="Thermal Input (Actual)", color="black")
            ax.plot(time_vals, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")
            ax.plot(time_vals, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")
            ax.plot(time_vals, t_max_pred_original, label="T_max (Predicted)", linestyle="dashed", color="red")
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


if __name__ == "__main__":
    model = train_model()
    test_model(model)
