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

class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name

        # Dictionary of default constant values for new features
        default_features = {
            "input_type": "temperature",
            "ambient_temperature": 300,
            "convection_coefficient": 20,
            "shape": "cylindrical",
            "length": 20,
            "height": 3,
            "diameter_of_hx": 3,
            "density": 200,
            "thermal_conductivity": 2,
            "specific_heat": 400,
        }
        # Add missing columns with their constant values
        for col, value in default_features.items():
            if col not in df.columns:
                df[col] = value

        # Map categorical features to numeric values
        df["input_type"] = df["input_type"].map({"temperature": 0, "convection": 1})
        df["shape"] = df["shape"].map({"cylindrical": 0, "square": 1})

        # Define full input columns (original 5 columns + 10 new = 15)
        input_columns = [
            "Time (s)", 
            "T_min (C)", 
            "T_max (C)", 
            "T_ave (C)", 
            "Thermal_Input_C",
            "input_type",
            "ambient_temperature",
            "convection_coefficient",
            "shape",
            "length",
            "height",
            "diameter_of_hx",
            "density",
            "thermal_conductivity",
            "specific_heat"
        ]
        self.input_columns = input_columns

        # Fit the scaler on all 15 input columns and transform the data
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[input_columns])
        df[input_columns] = self.scaler.transform(df[input_columns])

        # Group the data by file name and create sequences
        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []

        for _, group in grouped:
            # Build input sequence: all 15 features for all rows except the last one
            X_seq = group[input_columns].values[:-1]
            # Build target sequence from temperature columns; we predict T_min and T_ave (columns 1 and 3)
            Y_seq = group[["T_min (C)", "T_max (C)", "T_ave (C)"]].values[1:]
            Y_seq = Y_seq[:, [0, 2]]  # select only T_min and T_ave
            time_vals = group["Time (s)"].values[1:]
            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)

        # print(f"Loaded {len(self.X)} sequences from {self.file_name}.")
        # if len(self.X) > 0:
        #     print(f"  Input shape: {self.X[0].shape}, Target shape: {self.Y[0].shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=48, num_layers=4, output_size=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out


def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0])):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets)
    return torch.mean(loss * weights)


def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Specify file names for training, validation, and testing
    case_1 = [f"Case1_TrialData_Sample{i}.csv" for i in range(1, 13)]
    case2_nonlinear = [f"Case2_Nonlinear_Sample{i}.csv" for i in range(5, 7)]
    train_files = case_1 + case2_nonlinear
    val_files = [f"Validation_{i}.csv" for i in range(1, 4)]
    test_files = [f"Case{i}.csv" for i in range(1, 6)]

    train_paths = [os.path.join(script_dir, "data", "train", f) for f in train_files]
    val_paths   = [os.path.join(script_dir, "data", "validation", f) for f in val_files]
    test_paths  = [os.path.join(script_dir, "data", "testing", f) for f in test_files]

    train_datasets = [ThermalDataset(path) for path in train_paths]
    val_datasets   = [ThermalDataset(path) for path in val_paths]
    test_datasets  = [ThermalDataset(path) for path in test_paths]

    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset   = ConcatDataset(val_datasets)

    train_dataloader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
    val_dataloader   = DataLoader(combined_val_dataset,   batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(input_size=15).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    num_epochs = 10000
    patience = 300
    best_val_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        for inputs, targets, _ in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = weighted_loss(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets, _ in val_dataloader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)
                val_outputs = model(val_inputs)
                loss_val = weighted_loss(val_outputs, val_targets)
                total_val_loss += loss_val.item()
        epoch_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(epoch_val_loss)

        # Uncomment below if you want early stopping behavior
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     early_stop_counter = 0
        #     best_model_path = os.path.join(script_dir, "best_model.pth")
        #     torch.save(model.state_dict(), best_model_path)
        # else:
        #     early_stop_counter += 1
        #     if early_stop_counter >= patience:
        #         print(f"Early stopping triggered at epoch {epoch+1}.")
        #         break

        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    # If using early stopping, load the best model here:
    # model.load_state_dict(torch.load(os.path.join(script_dir, "best_model.pth")))
    return model, test_datasets


def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_figures = []
    all_actual = []
    all_predicted = []

    for i, test_dataset in enumerate(test_datasets):
        actual_name = os.path.basename(test_dataset.file_name)
        print(f"Testing on dataset: {actual_name}")
        try:
            num_examples = len(test_dataset)
            fig, axes = plt.subplots(num_examples, 1, figsize=(10, 5 * num_examples))
            if num_examples == 1:
                axes = [axes]
            for idx in range(num_examples):
                test_input, test_actual, time_values = test_dataset[idx]
                test_input_batch = test_input.unsqueeze(0).to(device)
                with torch.no_grad():
                    predicted_sequence = model(test_input_batch).cpu().numpy()
                t_min_pred = predicted_sequence[0, :, 0]
                t_ave_pred = predicted_sequence[0, :, 1]
                t_max_input = test_input[:, 2].cpu().numpy()
                t_min_actual = test_actual[:, 0].numpy()
                t_ave_actual = test_actual[:, 1].numpy()
                t_max_actual = t_max_input  # from the input

                scaler = test_dataset.scaler
                # Create dummy arrays with 15 columns for inverse scaling
                dummy_pred_min = np.zeros((len(t_min_pred), 15))
                dummy_pred_min[:, 1] = t_min_pred
                inv_min = scaler.inverse_transform(dummy_pred_min)
                t_min_pred_original = inv_min[:, 1]

                dummy_pred_ave = np.zeros((len(t_ave_pred), 15))
                dummy_pred_ave[:, 3] = t_ave_pred
                inv_ave = scaler.inverse_transform(dummy_pred_ave)
                t_ave_pred_original = inv_ave[:, 3]

                dummy_input_max = np.zeros((len(t_max_input), 15))
                dummy_input_max[:, 2] = t_max_input
                inv_max = scaler.inverse_transform(dummy_input_max)
                t_max_pred_original = inv_max[:, 2]

                dummy_act_min = np.zeros((len(t_min_actual), 15))
                dummy_act_min[:, 1] = t_min_actual
                inv_act_min = scaler.inverse_transform(dummy_act_min)
                t_min_actual_original = inv_act_min[:, 1]

                dummy_act_ave = np.zeros((len(t_ave_actual), 15))
                dummy_act_ave[:, 3] = t_ave_actual
                inv_act_ave = scaler.inverse_transform(dummy_act_ave)
                t_ave_actual_original = inv_act_ave[:, 3]

                dummy_act_max = np.zeros((len(t_max_actual), 15))
                dummy_act_max[:, 2] = t_max_actual
                inv_act_max = scaler.inverse_transform(dummy_act_max)
                t_max_actual_original = inv_act_max[:, 2]

                dummy_time = np.zeros((len(time_values), 15))
                dummy_time[:, 0] = time_values
                inv_time = scaler.inverse_transform(dummy_time)
                time_values_original = inv_time[:, 0]

                all_actual.append(np.stack([t_min_actual_original, t_max_actual_original, t_ave_actual_original], axis=1))
                all_predicted.append(np.stack([t_min_pred_original, t_max_pred_original, t_ave_pred_original], axis=1))

                ax = axes[idx]
                ax.plot(time_values_original, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed")
                ax.plot(time_values_original, t_max_actual_original, label="Thermal Input")
                ax.plot(time_values_original, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed")
                ax.set_xlabel("Time")
                ax.set_ylabel("Temperature (C)")
                ax.legend()
                ax.set_title(f"Actual vs. Predicted for {actual_name} (Example {idx+1})")

            all_figures.append(fig)
        except Exception as e:
            print(f"Error testing on {actual_name}: {e}")
    for fig in all_figures:
        plt.figure(fig.number)
    plt.show()


if __name__ == "__main__":
    trained_model, test_datasets = train_model()
    test_model(trained_model, test_datasets)
