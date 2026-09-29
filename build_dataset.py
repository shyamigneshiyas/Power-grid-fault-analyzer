import numpy as np
import pandas as pd
import random
import os

# Base Grid Parameters
f = 50.0
fs = 10000.0
t = np.arange(0, 1.0, 1/fs)
v_peak_base = 325.27 # 230V RMS

def extract_features(signal):
    # 1. Root Mean Square (RMS)
    rms = np.sqrt(np.mean(signal**2))
    
    # 2. Peak Voltage
    peak = np.max(np.abs(signal))
    
    # 3. Total Harmonic Distortion (THD) via FFT
    # A 1-second window means the frequency resolution is exactly 1 Hz.
    fft_spectrum = np.abs(np.fft.rfft(signal))
    
    fundamental = fft_spectrum[int(f)] # The 50 Hz bin
    # Sum the energy of all frequencies above 60 Hz to isolate harmonics
    harmonics_rms = np.sqrt(np.sum(fft_spectrum[60:]**2))
    thd = (harmonics_rms / (fundamental + 1e-6)) * 100 # Output as a percentage
    
    return rms, peak, thd

dataset = []
print("Generating 5,000 randomized grid samples...")

for _ in range(1000):
    # Base signal with slight random background noise
    noise = np.random.normal(0, 1.5, len(t))
    normal = v_peak_base * np.sin(2 * np.pi * f * t) + noise
    
    # Class 0: Normal
    rms, peak, thd = extract_features(normal)
    dataset.append([rms, peak, thd, 0])
    
    # Class 1: Voltage Sag (Random drop between 10% and 90%, random duration)
    sag = normal.copy()
    drop = random.uniform(0.1, 0.9)
    start = random.randint(1000, 5000)
    duration = random.randint(1000, 3000)
    sag[start:start+duration] *= drop
    rms, peak, thd = extract_features(sag)
    dataset.append([rms, peak, thd, 1])
    
    # Class 2: Voltage Swell (Random spike between 110% and 180%)
    swell = normal.copy()
    spike = random.uniform(1.1, 1.8)
    start = random.randint(1000, 5000)
    duration = random.randint(1000, 3000)
    swell[start:start+duration] *= spike
    rms, peak, thd = extract_features(swell)
    dataset.append([rms, peak, thd, 2])
    
    # Class 3: Overcurrent (Massive sudden spike up to 300%)
    overcurrent = normal.copy()
    spike = random.uniform(2.0, 3.0)
    start = random.randint(2000, 6000)
    overcurrent[start:] *= spike
    rms, peak, thd = extract_features(overcurrent)
    dataset.append([rms, peak, thd, 3])
    
    # Class 4: Harmonics (Randomized 3rd, 5th, and 7th harmonic injection)
    h3 = random.uniform(0.05, 0.20) * v_peak_base * np.sin(2 * np.pi * (3 * f) * t)
    h5 = random.uniform(0.05, 0.15) * v_peak_base * np.sin(2 * np.pi * (5 * f) * t)
    harmonics = normal + h3 + h5
    rms, peak, thd = extract_features(harmonics)
    dataset.append([rms, peak, thd, 4])

# Convert to a Pandas DataFrame and save to the data/ folder
columns = ["RMS_Voltage", "Peak_Voltage", "THD_Percentage", "Fault_Type"]
df = pd.DataFrame(dataset, columns=columns)

output_path = os.path.join("data", "grid_dataset.csv")
df.to_csv(output_path, index=False)
print(f"Dataset successfully created and saved to {output_path}!")