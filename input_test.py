import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from lstm import LSTMModel  # ensure lstm.py (modified above) is in your PYTHONPATH

def test_model_with_user_input(model, input_scaler=None):
    """
    This function interacts with the user, asking for Time and Thermal_Input_C.
    It then uses the trained LSTM model (which expects a sequence of two features)
    to forecast T_min, T_max, and T_ave up to time = 14400.
    
    A dummy input scaler is used if none is provided.
    """
    # Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # If no input scaler provided, create one for 2 features: [Time, Thermal_Input_C]
    if input_scaler is None:
        input_scaler = MinMaxScaler()
        dummy_data = np.array([[0, 0], [14400, 800]])
        input_scaler.fit(dummy_data)

    # Ask user for the number of data points (each with Time and Thermal_Input_C)
    N = int(input("How many data points will you input? "))
    times_user = []
    thermal_input_user = []
    print("\nPlease input each data point (Time(s) and Thermal_Input_C).")
    for i in range(N):
        print(f"\nData point {i+1}:")
        t_s = float(input("  Time (s): "))
        thermal_in = float(input("  Thermal_Input_C: "))
        times_user.append(t_s)
        thermal_input_user.append(thermal_in)
    
    times_user = np.array(times_user)
    thermal_input_user = np.array(thermal_input_user)
    
    # Sort user inputs by time
    sort_indices = np.argsort(times_user)
    times_user = times_user[sort_indices]
    thermal_input_user = thermal_input_user[sort_indices]
    
    # Build input array of shape (N, 2)
    user_input_array = np.column_stack((times_user, thermal_input_user))
    
    # Forecast from the last provided time to 14400 with fixed step (e.g., 100s)
    last_time = times_user[-1]
    forecast_times = []
    time_step = 100.0
    while last_time < 14400:
        last_time += time_step
        if last_time > 14400:
            last_time = 14400
        forecast_times.append(last_time)
        if last_time >= 14400:
            break
    forecast_times = np.array(forecast_times)
    
    if len(forecast_times) > 0:
        # Assume the thermal input remains constant (equal to the last provided value)
        constant_thermal = thermal_input_user[-1]
        forecast_input_array = np.column_stack((forecast_times, np.full(forecast_times.shape, constant_thermal)))
        # Combine user input and forecast inputs
        full_input_array = np.concatenate((user_input_array, forecast_input_array), axis=0)
    else:
        full_input_array = user_input_array
    
    # Scale the full input array
    full_input_scaled = input_scaler.transform(full_input_array)
    
    # Convert to tensor with shape (1, sequence_length, 2)
    full_input_tensor = torch.tensor(full_input_scaled, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Run the model to get predictions (shape: (1, seq_len, 3))
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(full_input_tensor)
        predictions_scaled = predictions_scaled.squeeze(0).cpu().numpy()
    
    # Create a dummy target scaler for inverse-transforming predictions
    # (Assuming temperature values are in the range [0,600])
    from sklearn.preprocessing import MinMaxScaler
    target_scaler = MinMaxScaler()
    dummy_target = np.array([[0, 0, 0], [600, 600, 600]])
    target_scaler.fit(dummy_target)
    predictions = target_scaler.inverse_transform(predictions_scaled)
    
    # The time axis (unscaled) is just the first column of full_input_array
    final_times = full_input_array[:, 0]
    t_min_pred = predictions[:, 0]
    t_max_pred = predictions[:, 1]
    t_ave_pred = predictions[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.plot(final_times, t_min_pred, label="T_min (Predicted)", color="blue", linestyle="dashed")
    plt.plot(final_times, t_max_pred, label="T_max (Predicted)", color="red", linestyle="dashed")
    plt.plot(final_times, t_ave_pred, label="T_ave (Predicted)", color="green", linestyle="dashed")
    plt.title("Forecasted Temperatures from User Input")
    plt.xlabel("Time (s)")
    plt.ylabel("Temperature (C)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\nDone! The plot shows the forecasted T_min, T_max, and T_ave.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Instantiate the model (make sure it matches the training configuration)
    model = LSTMModel()  
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    test_model_with_user_input(model, input_scaler=None)
