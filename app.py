import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
import os

# 1. Page Setup
st.set_page_config(page_title="Power Grid Fault Analyzer", layout="wide")
st.title("⚡ Power Grid Fault Analyzer")
st.markdown("Simulate electrical faults and let the Random Forest AI diagnose them in real-time.")

# 2. Load the trained AI Model
model_path = os.path.join("models", "rf_fault_classifier.pkl")
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    st.error("⚠️ Model not found! Please run train_model.py first to generate the AI model.")
    st.stop()

# 3. Sidebar Controls
st.sidebar.header("Signal Generator")
fault_type = st.sidebar.selectbox(
    "Select a Grid Condition to Simulate:", 
    ["Normal", "Voltage Sag", "Voltage Swell", "Overcurrent", "Harmonics"]
)

# 4. Generate the corresponding signal
f = 50.0
fs = 10000.0
t = np.arange(0, 1.0, 1/fs)
v_peak = 325.27

signal = v_peak * np.sin(2 * np.pi * f * t)

if fault_type == "Voltage Sag":
    signal[3000:5000] *= 0.50
elif fault_type == "Voltage Swell":
    signal[6000:8000] *= 1.50
elif fault_type == "Overcurrent":
    signal[4000:] *= 2.0
elif fault_type == "Harmonics":
    signal += (0.15 * v_peak * np.sin(2 * np.pi * 3 * f * t)) + \
              (0.10 * v_peak * np.sin(2 * np.pi * 5 * f * t))

# 5. Extract Features (The Math)
rms = np.sqrt(np.mean(signal**2))
peak = np.max(np.abs(signal))

fft_spectrum = np.abs(np.fft.rfft(signal))
fundamental = fft_spectrum[int(f)]
harmonics_rms = np.sqrt(np.sum(fft_spectrum[60:]**2))
thd = (harmonics_rms / (fundamental + 1e-6)) * 100

# 6. Build the UI Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Live Grid Waveform")
    fig = go.Figure()
    # Plotting only the first 0.1 seconds (1000 data points) so the wave is readable
    fig.add_trace(go.Scatter(x=t[:1000], y=signal[:1000], mode='lines', name='Voltage', line=dict(color='#00F1FF')))
    fig.update_layout(xaxis_title="Time (s)", yaxis_title="Voltage (V)", template="plotly_dark", height=450)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Signal Telemetry")
    st.metric("RMS Voltage", f"{rms:.2f} V")
    st.metric("Peak Voltage", f"{peak:.2f} V")
    st.metric("THD", f"{thd:.2f} %")
    
    st.subheader("AI Diagnosis")
    # Feed the math into the AI
    features = np.array([[rms, peak, thd]])
    prediction = model.predict(features)[0]
    
    fault_map = {0: "Normal", 1: "Voltage Sag", 2: "Voltage Swell", 3: "Overcurrent", 4: "Harmonic Distortion"}
    diagnosis = fault_map[prediction]
    
    if diagnosis == "Normal":
        st.success(f"✅ System Stable: {diagnosis}")
    else:
        st.error(f"⚠️ Fault Detected: {diagnosis}")