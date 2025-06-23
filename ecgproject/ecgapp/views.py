from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import default_storage
import os
import sys
import subprocess
import torch
import numpy as np
import pandas as pd
from scipy.signal import resample
from .scripts.ecg_model import load_model, predict_ecg

# === Static Pages ===
def dashboard(request):
    return render(request, 'dashboard.html')

def sensors(request):
    return render(request, 'sensors.html')

def settings(request):
    return render(request, 'settings.html')

def sample_graph(request):
    return render(request, 'sample_graph.html')

def own_ecg(request):
    return render(request, 'make your ecg.html')

# === ECG Script Trigger (not used with live prediction) ===
def run_ecg_script(request):
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(base_dir, 'scripts', 'ECG_Monitoring.py')
        if not os.path.exists(script_path):
            return HttpResponse(f"Script not found at: {script_path}")
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, encoding='utf-8', errors='replace')
        return HttpResponse(result.stdout or result.stderr)
    except Exception as e:
        return HttpResponse(f"Error running script: {str(e)}")

# === Load the trained model once ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
model = load_model(MODEL_PATH)

def ecg_upload_view(request):
    prediction = None

    if request.method == 'POST' and request.FILES.get('ecg_file'):
        ecg_file = request.FILES['ecg_file']
        file_path = default_storage.save('temp_ecg_file.csv', ecg_file)

        try:
            full_path = os.path.join(default_storage.location, file_path)

            # Load ECG signal from uploaded file (with header)
            df = pd.read_csv(full_path)

            if 'Filtered_ECG' not in df.columns:
                raise ValueError("Column 'Filtered_ECG' not found in uploaded file.")

            filtered_signal = df['Filtered_ECG'].astype(float).values  # convert to float

            # Resample to 100Hz
            duration = len(filtered_signal) / 360
            target_len = int(duration * 100)
            resampled_signal = resample(filtered_signal, target_len)

            # Normalize
            resampled_signal = (resampled_signal - np.mean(resampled_signal)) / np.std(resampled_signal)

            # Pad or trim to fixed length (e.g., 180)
            if len(resampled_signal) < 180:
                final_signal = np.pad(resampled_signal, (0, 180 - len(resampled_signal)), mode='constant')
            else:
                final_signal = resampled_signal[:180]

            # Predict
            prediction = predict_ecg(final_signal, model)

        except Exception as e:
            prediction = f"Error: {str(e)}"

        finally:
            if default_storage.exists(file_path):
                default_storage.delete(file_path)

    return render(request, 'upload.html', {'prediction': prediction})
