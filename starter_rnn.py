import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Train: Sample 1, 2, 7

# Load and Prepare Data
class ThermalDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        df[['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']] = df[['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp']].fillna(method='ffill')
        grouped = df.groupby(['Start_Time', 'Start_Temp', 'End_Time', 'End_Temp'])

        self.X, self.Y, self.time_values = [], [], []  # Store time separately

        for (start_time, start_temp, end_time, end_temp), group in grouped:
            if len(group) == 100:
                self.X.append([start_time, start_temp, end_time, end_temp])
                self.Y.append(group[['T_min (C)', 'T_max (C)', 'T_ave (C)']].values)  # ✅ Only temperatures
                self.time_values.append(group['Time (s)'].values)  # ✅ Store time separately

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)
        self.time_values = np.array(self.time_values)  # Convert to NumPy for easy plotting

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.time_values[idx]  # ✅ Return time values too

# Define LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=3, seq_length=100):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1).repeat(1, self.seq_length, 1)  # Expand to match sequence length
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# Training the Model
def train_model():
    multiple_paths = ["C:\\Users\\lolly\\OneDrive\\Documents\\ThermalAITest1.csv",  "C:\\Users\\lolly\\OneDrive\\Documents\\ThermalAITest2 .csv", 
                      "C:\\Users\\lolly\\OneDrive\\Documents\\ThermalAITest3.csv"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize Model
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 8000

    all_datasets = []

    for csv_file_path in multiple_paths:
        print(f"Training on dataset: {csv_file_path}")

        dataset = ThermalDataset(csv_file_path)
        all_datasets.append(dataset)  
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        best_loss = float("inf")
        patience = 50
        counter = 0

        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets, _ in dataloader:  # ✅ Ignore time values
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss /= len(dataloader)
            '''
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] on {csv_file_path}, Loss: {epoch_loss:.4f}")

            # Early stopping
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} for {csv_file_path}. No improvement in {patience} epochs.")
                    break
            '''
    return model, all_datasets




def test_model(model, combined_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Select the first dataset for testing
    sample_dataset = combined_dataset[0]  

    # Extract a test sample (first example)
    test_input, test_actual, time_values = sample_dataset[0]  # ✅ Get Time (s)
    test_input = test_input.unsqueeze(0).to(device)  # Add batch dimension

    # Generate predictions
    with torch.no_grad():
        predicted_sequence = model(test_input).cpu().numpy()

    # Extract predicted values
    t_min_pred = predicted_sequence[0, :, 0]
    t_max_pred = predicted_sequence[0, :, 1]
    t_ave_pred = predicted_sequence[0, :, 2]

    # Extract actual values
    t_min_actual = test_actual[:, 0].numpy()
    t_max_actual = test_actual[:, 1].numpy()
    t_ave_actual = test_actual[:, 2].numpy()

    # ✅ Now using Time (s) as X-axis
    plt.figure(figsize=(10, 5))
    plt.plot(time_values, t_min_actual, label="T_min (Actual)", color="blue")
    plt.plot(time_values, t_min_pred, label="T_min (Predicted)", linestyle="dashed", color="blue")

    plt.plot(time_values, t_max_actual, label="T_max (Actual)", color="red")
    plt.plot(time_values, t_max_pred, label="T_max (Predicted)", linestyle="dashed", color="red")

    plt.plot(time_values, t_ave_actual, label="T_ave (Actual)", color="green")
    plt.plot(time_values, t_ave_pred, label="T_ave (Predicted)", linestyle="dashed", color="green")

    plt.xlabel("Time (s)")  # ✅ Now correct
    plt.ylabel("Temperature (°C)")
    plt.legend()
    plt.title("Actual vs. Predicted Temperature Trends")
    plt.show()


# Run the Training and Prediction
trained_model, data = train_model()
test_model(trained_model, data)
