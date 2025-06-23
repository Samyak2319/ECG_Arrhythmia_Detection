# ecg_model.py

import torch
import torch.nn as nn
import numpy as np
from scipy.signal import resample
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNLSTMWithAttention(nn.Module):
    def __init__(self):
        super(CNNLSTMWithAttention, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(128, 64, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        avg_pool = torch.mean(attn_out, 1)
        return self.fc(avg_pool)

# Load model
def load_model(model_path):
    model = CNNLSTMWithAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict ECG class
def predict_ecg(signal, model):
    # Resample from 360Hz to 100Hz
    duration_sec = len(signal) / 360
    num_samples_100hz = int(duration_sec * 100)
    resampled = resample(signal, num_samples_100hz)

    # Normalize
    resampled = (resampled - np.mean(resampled)) / np.std(resampled)

    # Make sure input is 180 samples (like training)
    if len(resampled) < 180:
        padded = np.pad(resampled, (0, 180 - len(resampled)), mode='constant')
    else:
        padded = resampled[:180]

    input_tensor = torch.tensor(padded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        return "ARRHYTHMIA" if predicted.item() == 0 else "NORMAL"
       # return "NORMAL" if predicted.item() == 0 else "ARRHYTHMIA"
    
