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
# ThermalDataset:
# Loads the CSV data, scales the temperature columns, and creates a single sequence.
# Input X consists of:
#   [Time (s), T_min (C), T_max (C), T_ave (C), Thermal_Input_C]
# Target Y is built by shifting the original data by one time step.
# Since T_max is assumed to remain unchanged from input, we only use T_min and T_ave
# for the learning target.
# =============================================================================
class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.scaler = MinMaxScaler()
        # Scale the temperature-related columns and the thermal input.
        self.scaler.fit(df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']])
        df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']] = self.scaler.transform(
            df[['T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']]
        )
        # Instead of grouping, treat the entire CSV as one continuous sequence.
        data = df[['Time (s)', 'T_min (C)', 'T_max (C)', 'T_ave (C)', 'Thermal_Input_C']].values
        # Input sequence: all rows except the last one.
        self.X = [data[:-1]]
        # Target sequence: shifted rows; drop T_max (column index 1) since it is taken from input.
        target = df[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values[1:]
        self.Y = [target[:, [0, 2]]]
        # Time values for plotting.
        self.time_values = [df['Time (s)'].values[1:]]
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        print(f"Loaded {len(self.X)} sequence. Input shape: {self.X[0].shape}, Target shape: {self.Y[0].shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]

# =============================================================================
# LSTMModel:
# A simplified LSTM that predicts only two values (T_min and T_ave) per time step.
# Since we assume that T_max is exactly as provided in the input, it is not predicted.
# =============================================================================
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=48, num_layers=2, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Use an LSTM with reduced layers/hidden size for faster training.
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
# A custom loss function that computes an L1 loss on the predictions.
# Since only T_min and T_ave are predicted, the weight vector has two entries.
# =============================================================================
def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0])):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets)
    return torch.mean(loss * weights)

# =============================================================================
# train_model:
# Loads training files, initializes the model, and trains on T_min and T_ave only.
# =============================================================================
def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Define training and testing file ranges.
    train_files = [f"Case1_TrialData_Sample{i}.csv" for i in range(15, 151)]
    train_files2 = [f"Case2_Linear Constant_Sample{i}.csv" for i in range(9, 149)]
    train_files3 = [f"Case2_Nonlinear_Sample{i}.csv" for i in range(7, 11)]
    train_files = train_files + train_files2 + train_files3
    test_files = [f"Test_{i}.csv" for i in range(1, 6)] + [f"Case2_Nonlinear_Sample{i}.csv" for i in range(4, 7)]
    train_paths = [os.path.join(script_dir, "data", "train", file) for file in train_files]
    test_paths = [os.path.join(script_dir, "data", "testing", file) for file in test_files]
    # Load and concatenate training datasets.
    train_datasets = [ThermalDataset(path) for path in train_paths]
    combined_train_dataset = ConcatDataset(train_datasets)
    # Load testing datasets.
    test_datasets = [ThermalDataset(path) for path in test_paths]
    train_dataloader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2000
    patience = 300  # Early stopping patience.
    best_loss = float("inf")
    counter = 0
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=100, factor=0.5)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # Model predicts only T_min and T_ave.
            outputs = model(inputs)
            loss = weighted_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dataloader)
        scheduler.step(epoch_loss)

        # Early stopping check.
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}. No improvement in {patience} epochs.")
                break

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return model, test_datasets

# =============================================================================
# test_model:
# For each test sample, the model's predictions for T_min and T_ave are computed.
# T_max is taken directly from the input sequence.
# The predictions and actual values (all scaled back to original units) are plotted,
# and overall accuracy metrics (MAPE and R-squared) are calculated.
# =============================================================================
def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_figures = []
    all_actual = []
    all_predicted = []

    for i, test_dataset in enumerate(test_datasets):
        dataset_name = f"Expanded_sample{i+10}.csv"
        print(f"Testing on dataset: {dataset_name}")
        try:
            num_examples = len(test_dataset)
            fig, axes = plt.subplots(num_examples, 1, figsize=(10, 5 * num_examples))
            if num_examples == 1:
                axes = [axes]

            for idx in range(num_examples):
                test_input, test_actual, time_values = test_dataset[idx]
                # test_input shape: (seq_len, 5); test_actual shape: (seq_len, 2) for T_min and T_ave.
                test_input = test_input.unsqueeze(0).to(device)  # Shape: (1, seq_len, 5)
                with torch.no_grad():
                    predicted_sequence = model(test_input).cpu().numpy()  # Shape: (1, seq_len, 2)
                t_min_pred = predicted_sequence[0, :, 0]
                t_ave_pred = predicted_sequence[0, :, 1]
                # For T_max, simply copy from the input.
                t_max_pred = test_input[0, :, 2].cpu().numpy()

                # Actual values for T_min and T_ave are in test_actual.
                t_min_actual = test_actual[:, 0].numpy()
                t_ave_actual = test_actual[:, 1].numpy()
                # For T_max actual, we assume it is identical to the (shifted) input.
                t_max_actual = test_input[0, :, 2].cpu().numpy()

                scaler = test_dataset.scaler
                # Reverse scaling: we create dummy 4-column arrays for the scaler.
                t_min_pred_original = scaler.inverse_transform(
                    np.column_stack((t_min_pred, t_min_pred, t_min_pred, t_min_pred))
                )[:, 0]
                t_ave_pred_original = scaler.inverse_transform(
                    np.column_stack((t_ave_pred, t_ave_pred, t_ave_pred, t_ave_pred))
                )[:, 2]
                t_max_pred_original = scaler.inverse_transform(
                    np.column_stack((t_max_pred, t_max_pred, t_max_pred, t_max_pred))
                )[:, 1]

                t_min_actual_original = scaler.inverse_transform(
                    np.column_stack((t_min_actual, t_min_actual, t_min_actual, t_min_actual))
                )[:, 0]
                t_ave_actual_original = scaler.inverse_transform(
                    np.column_stack((t_ave_actual, t_ave_actual, t_ave_actual, t_ave_actual))
                )[:, 2]
                t_max_actual_original = scaler.inverse_transform(
                    np.column_stack((t_max_actual, t_max_actual, t_max_actual, t_max_actual))
                )[:, 1]

                all_actual.append(np.stack([t_min_actual_original, t_max_actual_original, t_ave_actual_original], axis=1))
                all_predicted.append(np.stack([t_min_pred_original, t_max_pred_original, t_ave_pred_original], axis=1))

                ax = axes[idx]
                ax.plot(time_values, t_min_actual_original, label="T_min (Actual)", color="blue")
                ax.plot(time_values, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")
                ax.plot(time_values, t_max_actual_original, label="T_max (Actual)", color="red")
                ax.plot(time_values, t_max_pred_original, label="T_max (Predicted)", linestyle="dashed", color="red")
                ax.plot(time_values, t_ave_actual_original, label="T_ave (Actual)", color="green")
                ax.plot(time_values, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.set_title(f"Actual vs. Predicted Trends for {dataset_name} (Example {idx+1})")

            all_figures.append(fig)
        except Exception as e:
            print(f"Error testing on {dataset_name}: {e}")

    all_actual = np.concatenate(all_actual, axis=0)
    all_predicted = np.concatenate(all_predicted, axis=0)
    mape = mean_absolute_percentage_error(all_actual, all_predicted) * 100  # Percentage
    r2 = r2_score(all_actual, all_predicted)
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")

    for fig in all_figures:
        plt.figure(fig.number)
    plt.show()

# =============================================================================
# Main execution: Train the model and then evaluate on test datasets.
# =============================================================================
if __name__ == "__main__":
    trained_model, test_datasets = train_model()
    test_model(trained_model, test_datasets)
