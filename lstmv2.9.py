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

        columns_for_scaling = [
            "Time (s)", 
            "T_min (C)", 
            "T_max (C)", 
            "T_ave (C)", 
            "Thermal_Input_C",
        ]
        self.scaler = MinMaxScaler()
        self.scaler.fit(df[columns_for_scaling])


        df[columns_for_scaling] = self.scaler.transform(df[columns_for_scaling])


        grouped = df.groupby("FileName")
        self.X, self.Y, self.time_values = [], [], []

        for _, group in grouped:

            X_seq = group[[
                "Time (s)",
                "T_min (C)",
                "T_max (C)",
                "T_ave (C)",
                "Thermal_Input_C"
            ]].values[:-1]


            Y_seq = group[["T_min (C)", "T_max (C)", "T_ave (C)"]].values[1:]
            Y_seq = Y_seq[:, [0, 2]]  # keep only T_min & T_ave

            time_vals = group["Time (s)"].values[1:]  # shifted by 1

            self.X.append(X_seq)
            self.Y.append(Y_seq)
            self.time_values.append(time_vals)

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)
        self.time_values = np.array(self.time_values)

        print(f"Loaded {len(self.X)} sequences from {self.file_name}.")
        print(f"  Input shape: {self.X[0].shape}, Target shape: {self.Y[0].shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]


# LSTM :]

class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=48, num_layers=4, output_size=2):
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

    train_files = [
        f"Case1_TrialData_Sample{i}.csv"
        for i in [
            15, 16, 17, 18, 19, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 38, 39, 48, 49, 58, 59, 60, 61, 62, 63, 64, 65, 66,
            67, 70, 71, 72, 73, 74, 75, 76, 77, 88, 89, 98, 99, 100,
            101, 102, 103, 104, 105, 106, 107, 110, 111, 112, 113, 114,
            115, 116, 117, 128, 129, 138, 139, 144, 145, 146, 147, 148,
            149, 150
        ] # Bro common data group L
    ]
    val_files = [f"Validation_{i}.csv" for i in range(1, 10)]
    test_files = [f"Test_{i}.csv" for i in range(1, 6)]

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
    model = LSTMModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=50, verbose=True
    )

    num_epochs = 10000
    patience = 300  # early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0


    # Train dat thang
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

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            early_stop_counter = 0
            best_model_path = os.path.join(script_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break


        if (epoch + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    model.load_state_dict(torch.load(os.path.join(script_dir, "best_model.pth")))
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


                t_max_input = test_input[ :, 2].cpu().numpy()


                t_min_actual = test_actual[:, 0].numpy()
                t_ave_actual = test_actual[:, 1].numpy()


                t_max_actual = t_max_input

                scaler = test_dataset.scaler


                dummy_pred_min = np.zeros((len(t_min_pred), 5))
                dummy_pred_min[:, 1] = t_min_pred  # col 1 is T_min
                inv_min = scaler.inverse_transform(dummy_pred_min)
                t_min_pred_original = inv_min[:, 1]  # read back column 1

                dummy_pred_ave = np.zeros((len(t_ave_pred), 5))
                dummy_pred_ave[:, 3] = t_ave_pred  # col 3 is T_ave
                inv_ave = scaler.inverse_transform(dummy_pred_ave)
                t_ave_pred_original = inv_ave[:, 3]


                dummy_input_max = np.zeros((len(t_max_input), 5))
                dummy_input_max[:, 2] = t_max_input 
                inv_max = scaler.inverse_transform(dummy_input_max)
                t_max_pred_original = inv_max[:, 2]  


                dummy_act_min = np.zeros((len(t_min_actual), 5))
                dummy_act_min[:, 1] = t_min_actual
                inv_act_min = scaler.inverse_transform(dummy_act_min)
                t_min_actual_original = inv_act_min[:, 1]

                dummy_act_ave = np.zeros((len(t_ave_actual), 5))
                dummy_act_ave[:, 3] = t_ave_actual
                inv_act_ave = scaler.inverse_transform(dummy_act_ave)
                t_ave_actual_original = inv_act_ave[:, 3]

                dummy_act_max = np.zeros((len(t_max_actual), 5))
                dummy_act_max[:, 2] = t_max_actual
                inv_act_max = scaler.inverse_transform(dummy_act_max)
                t_max_actual_original = inv_act_max[:, 2]

                all_actual.append(
                    np.stack([t_min_actual_original, t_max_actual_original, t_ave_actual_original], axis=1)
                )
                all_predicted.append(
                    np.stack([t_min_pred_original, t_max_pred_original, t_ave_pred_original], axis=1)
                )

                ax = axes[idx]
                ax.plot(time_values, t_min_actual_original, label="T_min (Actual)", color="blue")
                ax.plot(time_values, t_min_pred_original, label="T_min (Predicted)", linestyle="dashed", color="blue")

                ax.plot(time_values, t_max_actual_original, label="T_max (Actual)", color="red")

                ax.plot(time_values, t_max_pred_original, label="T_max (Provided)", linestyle="dashed", color="red")

                ax.plot(time_values, t_ave_actual_original, label="T_ave (Actual)", color="green")
                ax.plot(time_values, t_ave_pred_original, label="T_ave (Predicted)", linestyle="dashed", color="green")

                ax.set_xlabel("Time (s) [Scaled in training]")
                ax.set_ylabel("Temperature (°C)")
                ax.legend()
                ax.set_title(f"Actual vs. Predicted for {actual_name} (Example {idx+1})")

            all_figures.append(fig)

        except Exception as e:
            print(f"Error testing on {actual_name}: {e}")

    all_actual = np.concatenate(all_actual, axis=0)
    all_predicted = np.concatenate(all_predicted, axis=0)
    mape = mean_absolute_percentage_error(all_actual, all_predicted) * 100
    r2 = r2_score(all_actual, all_predicted)

    print(f"Overall MAPE: {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")


    for fig in all_figures:
        plt.figure(fig.number)
    plt.show()


if __name__ == "__main__":
    trained_model, test_datasets = train_model()
    test_model(trained_model, test_datasets)
