import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import serial.tools.list_ports
import csv
import time
import numpy as np
from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn

# Import your model class here
from ecg_model import *  # change this if your model class has a different name

# Load the trained model
model_path = 'best_model.pth'  # change if named differently
model = CNNLSTMWithAttention()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()


# ---------- Configuration ----------
BAUD_RATE = 115200
MAX_POINTS = 1000
CSV_FILENAME = f"ECG_{int(time.time())}.csv"
SAMPLE_RATE = 100  # Hz
NORMALIZE = True
FILTER_START_THRESHOLD = 150
# -----------------------------------

# Bandpass filter (0.5 Hz to 40 Hz)
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='band')

def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=SAMPLE_RATE, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    try:
        y = filtfilt(b, a, data)
        return np.nan_to_num(y)
    except Exception as e:
        print("[Filter] Skipped filtering due to:", e)
        return np.array(data)

# Find the correct serial port (Arduino)
def find_arduino_uno_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "Arduino_UNO" in p.description or "CH340" in p.description:
            return p.device
    raise Exception("Arduino-UNO connection not found. Please connect and try again.")

# Open serial connection
try:
    port = find_arduino_uno_port()
    ser = serial.Serial(port, BAUD_RATE, timeout=1)
    print(f"[OK] Connected to {port}")
except Exception as e:
    print(e)
    exit()

data = deque([0]*MAX_POINTS, maxlen=MAX_POINTS)

# Set up real-time plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2, color='blue')
ax.set_ylim(-300, 300)  # Set y-axis limits (adjust if needed)
ax.set_xlim(0, MAX_POINTS)
plt.title("Real-Time ECG Signal")
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)

# CSV file to save raw and filtered ECG data
csv_file = open(CSV_FILENAME, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Timestamp', 'Raw_ECG', 'Filtered_ECG'])

# Update function for real-time plotting and data logging
def update(frame):
    try:
        line_bytes = ser.readline()
        line_str = line_bytes.decode('utf-8').strip()
        if line_str.isdigit():
            val = int(line_str)

            # Normalize value to fit within 0-1023 range
            if NORMALIZE:
                val = int((val / 4095) * 1023)

            data.append(val)

            # Apply bandpass filter after a certain number of data points
            if len(data) >= FILTER_START_THRESHOLD:
                filtered = apply_bandpass_filter(list(data))
                display_data = filtered[-MAX_POINTS:]
                filtered_val = filtered[-1]
            else:
                display_data = list(data)
                filtered_val = val

            # Prevent mismatch length error
            if len(display_data) == MAX_POINTS:
                line.set_data(range(MAX_POINTS), display_data)

            # Write data to CSV (timestamp, raw, filtered)
            csv_writer.writerow([time.time(), val, filtered_val])

    except Exception as e:
        print("Update error:", e)
    return line,

# Animation for real-time plotting
ani = animation.FuncAnimation(fig, update, blit=True, interval=20, cache_frame_data=False)

# Display the plot
plt.tight_layout()
plt.show()

# Close the CSV file after plotting
csv_file.close()
