# ECG_Arrhythmia_Detection
🫀 ECGenius: Real-Time ECG Arrhythmia Detection using Deep Learning
This repository contains the complete implementation of ECGenius, a deep learning-based system for real-time ECG arrhythmia classification. Built using the MIT-BIH Arrhythmia Dataset, the model combines CNN + BiLSTM + Multihead Attention and is optimized with Focal Loss to handle class imbalance and improve performance.

🔬 Project Highlights
  📊 Achieved 97–99% accuracy on both preprocessed and raw ECG data.
  🧠 Architecture: CNN + BiLSTM + Multihead Attention + Focal Loss
  🏥 Trained on MIT-BIH Arrhythmia Dataset
  💡 Real-time prediction integration with ESP32 + AD8232 ECG sensor
  🖥️ Includes preprocessing scripts, training notebooks, model evaluation, and real-time dashboard code (Streamlit/Matplotlib).
  📦 Exported model ready for deployment (.pth format)

🚀 Goals
Enable fast and accurate detection of arrhythmias from ECG signals.
Develop a prototype for IoT-enabled cardiac monitoring using microcontrollers and deep learning.
