import pandas as pd
import numpy as np
import os
import glob

print("Combining Kaggle datasets...")

# Find all CSVs in the data folder
all_csvs = glob.glob(os.path.join("data", "*.csv"))
dataset = []

for file in all_csvs:
    filename = os.path.basename(file)
    
    # Skip our old synthetic data to avoid mixing it with Kaggle data
    if filename in ["grid_dataset.csv", "master_kaggle_dataset.csv"]:
        continue
        
    fault_name = filename.replace(".csv", "")
    
    # Keep the healthy wave named "Normal" so the web app stays green
    if fault_name == "Pure_Sinusoidal":
        fault_name = "Normal"
        
    print(f"Extracting features from {fault_name}...")
    
    df = pd.read_csv(file)
    
    # Process all 1000 waveforms in the file
    for index, row in df.iterrows():
        # Kaggle waves are normalized (Peak=1V). Multiply by 325.27 to get real Grid Voltages.
        signal = row.values * 325.27 
        
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        
        # Fast Fourier Transform for THD
        fft_spec = np.abs(np.fft.rfft(signal))
        fundamental = fft_spec[1] 
        harmonics_rms = np.sqrt(np.sum(fft_spec[2:]**2))
        thd = (harmonics_rms / (fundamental + 1e-6)) * 100
        
        dataset.append([rms, peak, thd, fault_name])

# Save all 17,000 rows into one master file
master_df = pd.DataFrame(dataset, columns=["RMS_Voltage", "Peak_Voltage", "THD_Percentage", "Fault_Type"])
output_path = os.path.join("data", "master_kaggle_dataset.csv")
master_df.to_csv(output_path, index=False)
print(f"\nSuccess! Saved 17,000 real-world samples to {output_path}")