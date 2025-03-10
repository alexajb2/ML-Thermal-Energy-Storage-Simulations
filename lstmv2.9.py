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

class ThermalDataset(Dataset):
    def __init__(self, csv_file, scaler=None):
        self.file_name = os.path.basename(csv_file)
        df = pd.read_csv(csv_file)
        df["FileName"] = self.file_name

        columns_for_scaling = ["Time (s)", "T_min (C)", "T_max (C)", "T_ave (C)", "Thermal_Input (C)"]
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(df[columns_for_scaling])
        else:
            self.scaler = scaler
        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])

        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []
        self.thermal_input_full = []
        self.full_time, self.full_t_min, self.full_t_max, self.full_t_ave = [], [], [], []

        for _, group in grouped:
            X_seq = group[["Time (s)", "T_min (C)", "T_ave (C)", "Thermal_Input (C)"]].values[:-1]
            Y_seq = group[["T_min (C)", "T_ave (C)"]].values[1:]
            time_vals = group["Time (s)"].values[1:]
            thermal_input = group["Thermal_Input (C)"].values[:-1]

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)
            self.thermal_input_full.append(thermal_input)
            self.full_time.append(group["Time (s)"].values)
            self.full_t_min.append(group["T_min (C)"].values)
            self.full_t_max.append(group["T_max (C)"].values)
            self.full_t_ave.append(group["T_ave (C)"].values)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)
        self.thermal_input_full = np.array(self.thermal_input_full)
        self.full_time = np.array(self.full_time)
        self.full_t_min = np.array(self.full_t_min)
        self.full_t_max = np.array(self.full_t_max)
        self.full_t_ave = np.array(self.full_t_ave)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.Y[idx],
            self.time_values[idx],
            self.thermal_input_full[idx],
            self.full_time[idx],
            self.full_t_min[idx],
            self.full_t_max[idx],
            self.full_t_ave[idx]
        )

class ThermalLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=48, output_size=2, num_layers=4):
        super(ThermalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h, c)

def weighted_loss(predictions, targets, weights=torch.tensor([1.0, 1.0]), time_weights=None):
    weights = weights.to(predictions.device)
    loss = torch.abs(predictions - targets) * weights
    if time_weights is not None:
        time_weights = time_weights.to(predictions.device)
        loss = loss * time_weights.unsqueeze(-1)  # Apply time-based weighting
    return torch.mean(loss)

def train_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_paths = glob.glob(os.path.join(script_dir, "data", "train", "*.csv"))
    val_paths = glob.glob(os.path.join(script_dir, "data", "validation", "*.csv"))
    test_paths = glob.glob(os.path.join(script_dir, "data", "testing", "*.csv"))

    train_dfs = [pd.read_csv(path) for path in train_paths]
    combined_train_df = pd.concat(train_dfs, ignore_index=True)
    columns_for_scaling = ["Time (s)", "T_min (C)", "T_max (C)", "T_ave (C)", "Thermal_Input (C)"]
    scaler = MinMaxScaler()
    scaler.fit(combined_train_df[columns_for_scaling])

    train_datasets = [ThermalDataset(path, scaler=scaler) for path in train_paths]
    val_datasets = [ThermalDataset(path, scaler=scaler) for path in val_paths]
    test_datasets = [ThermalDataset(path, scaler=scaler) for path in test_paths]

    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset = ConcatDataset(val_datasets)

    train_dataloader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(combined_val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThermalLSTM(input_size=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    num_epochs = 100
    patience = 300
    best_val_loss = float('inf')
    early_stop_counter = 0
    burn_in_steps = 5  # Warm-up period

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0
        teacher_forcing_ratio = max(0.0, 1.0 - (epoch / num_epochs))

        for batch in train_dataloader:
            inputs, targets, _, _, _, _, _, _ = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size, seq_len, _ = inputs.shape
            optimizer.zero_grad()
            hidden = model.init_hidden(batch_size)

            # Warm-up phase: Pass initial steps to initialize hidden state
            if seq_len > burn_in_steps:
                warm_up_input = inputs[:, :burn_in_steps, :]
                _, hidden = model(warm_up_input, hidden)

            batch_loss = 0.0
            current_t_min = inputs[:, 0, 1]
            current_t_ave = inputs[:, 0, 2]

            # Time-based weights: higher for initial steps, decaying linearly
            time_weights = torch.linspace(1.5, 1.0, seq_len, device=device)

            for t in range(seq_len):
                time_t = inputs[:, t, 0]
                thermal_input_t = inputs[:, t, 3]
                input_t = torch.stack([time_t, current_t_min, current_t_ave, thermal_input_t], dim=1)
                input_t = input_t.unsqueeze(1)
                output, hidden = model(input_t, hidden)
                output = output.squeeze(1)
                target_t = targets[:, t, :]
                loss_t = weighted_loss(output, target_t, time_weights=time_weights[t:t+1])
                batch_loss += loss_t
                if t < seq_len - 1:
                    use_teacher = (torch.rand(batch_size, device=device) < teacher_forcing_ratio).float()
                    ground_truth = inputs[:, t+1, 1:3]
                    current_t_min = use_teacher * ground_truth[:, 0] + (1 - use_teacher) * output[:, 0]
                    current_t_ave = use_teacher * ground_truth[:, 1] + (1 - use_teacher) * output[:, 1]

            loss = batch_loss / seq_len
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        epoch_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_inputs, val_targets, _, _, _, _, _, _ = val_batch
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)
                batch_size, seq_len, _ = val_inputs.shape
                val_hidden = model.init_hidden(batch_size)

                if seq_len > burn_in_steps:
                    warm_up_input = val_inputs[:, :burn_in_steps, :]
                    _, val_hidden = model(warm_up_input, val_hidden)

                batch_val_loss = 0.0
                current_t_min = val_inputs[:, 0, 1]
                current_t_ave = val_inputs[:, 0, 2]
                for t in range(seq_len):
                    time_t = val_inputs[:, t, 0]
                    thermal_input_t = val_inputs[:, t, 3]
                    input_t = torch.stack([time_t, current_t_min, current_t_ave, thermal_input_t], dim=1)
                    input_t = input_t.unsqueeze(1)
                    output, val_hidden = model(input_t, val_hidden)
                    output = output.squeeze(1)
                    target_t = val_targets[:, t, :]
                    loss_t = weighted_loss(output, target_t)
                    batch_val_loss += loss_t
                    if t < seq_len - 1:
                        current_t_min = val_inputs[:, t+1, 1]
                        current_t_ave = val_inputs[:, t+1, 2]
                total_val_loss += (batch_val_loss / seq_len).item()

        epoch_val_loss = total_val_loss / len(val_dataloader)
        scheduler.step(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(script_dir, "best_model.pth"))
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered.")
                break

        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(script_dir, "best_model.pth")))
    return model, test_datasets

def test_model(model, test_datasets):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_figures = []
    burn_in_steps = 5

    for test_dataset in test_datasets:
        actual_name = os.path.basename(test_dataset.file_name)
        print(f"Testing on dataset: {actual_name}")

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        test_input, test_actual, time_values, thermal_input_full, full_time, full_t_min, full_t_max, full_t_ave = test_dataset[0]
        test_input = test_input.to(device)
        thermal_input_full = torch.tensor(thermal_input_full, device=device)
        seq_len = test_input.shape[0]

        hidden = model.init_hidden(batch_size=1)
        if seq_len > burn_in_steps:
            warm_up_input = test_input[:burn_in_steps, :].unsqueeze(0)
            _, hidden = model(warm_up_input, hidden)

        current_t_min = test_input[0, 1]
        current_t_ave = test_input[0, 2]
        predictions = []

        for t in range(seq_len):
            input_x = torch.tensor([
                test_input[t, 0],
                current_t_min,
                current_t_ave,
                thermal_input_full[t]
            ], dtype=torch.float32).view(1, 1, 4).to(device)

            with torch.no_grad():
                output, hidden = model(input_x, hidden)
                prediction = output[0, 0]
                predictions.append(prediction.cpu().numpy())
                if t >= burn_in_steps - 1:
                    current_t_min = prediction[0]
                    current_t_ave = prediction[1]
                else:
                    current_t_min = test_input[t+1, 1]  # Use ground truth during burn-in
                    current_t_ave = test_input[t+1, 2]

        predicted_sequence = np.array(predictions)
        t_min_pred = predicted_sequence[:, 0]
        t_ave_pred = predicted_sequence[:, 1]

        scaler = test_dataset.scaler
        dummy = np.zeros((len(full_time), 5))
        dummy[:, 0] = full_time
        full_time_original = scaler.inverse_transform(dummy)[:, 0]
        dummy[:, 1] = full_t_min
        full_t_min_original = scaler.inverse_transform(dummy)[:, 1]
        dummy[:, 2] = full_t_max
        full_t_max_original = scaler.inverse_transform(dummy)[:, 2]
        dummy[:, 3] = full_t_ave
        full_t_ave_original = scaler.inverse_transform(dummy)[:, 3]

        dummy_pred = np.zeros((len(t_min_pred), 5))
        dummy_pred[:, 1] = t_min_pred
        dummy_pred[:, 3] = t_ave_pred
        inv_pred = scaler.inverse_transform(dummy_pred)
        t_min_pred_original = inv_pred[:, 1]
        t_ave_pred_original = inv_pred[:, 3]

        ax.plot(full_time_original, full_t_min_original, label="T_min (Actual)", color="blue")
        ax.plot(full_time_original[1:], t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")
        ax.plot(full_time_original, full_t_ave_original, label="T_ave (Actual)", color="green")
        ax.plot(full_time_original[1:], t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")
        ax.plot(full_time_original, full_t_max_original, label="T_max (Actual)", color="red")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (C)")
        ax.legend()
        ax.set_title(f"Actual vs. Predicted for {actual_name}")
        all_figures.append(fig)

    for fig in all_figures:
        plt.figure(fig.number)
    plt.show()

if __name__ == "__main__":
    trained_model, test_datasets = train_model()
    test_model(trained_model, test_datasets)