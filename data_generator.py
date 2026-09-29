import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Basic electrical parameters
# -------------------------------

frequency = 50              # AC frequency in Hz
voltage_rms = 230           # RMS voltage
duration = 0.2              # Duration in seconds
sampling_rate = 5000        # Samples per second

# -------------------------------
# 2. Create time samples
# -------------------------------

time = np.arange(
    0,
    duration,
    1 / sampling_rate
)

# -------------------------------
# 3. Convert RMS voltage to peak
# -------------------------------

voltage_peak = voltage_rms * np.sqrt(2)

# -------------------------------
# # -------------------------------
# 4. Generate normal AC waveform
# -------------------------------

voltage = voltage_peak * np.sin(
    2 * np.pi * frequency * time
)

# -------------------------------
# 5. Simulate voltage sag
# -------------------------------

sag_start = 0.08
sag_end = 0.14

sag_voltage_rms = 160
sag_voltage_peak = sag_voltage_rms * np.sqrt(2)

sag_mask = (time >= sag_start) & (time <= sag_end)

voltage[sag_mask] = sag_voltage_peak * np.sin(
    2 * np.pi * frequency * time[sag_mask]
)

# -------------------------------
# 5. Display basic information
# -------------------------------

print("POWER GRID SIGNAL GENERATOR")
print("----------------------------")
print(f"Frequency       : {frequency} Hz")
print(f"RMS Voltage     : {voltage_rms} V")
print(f"Peak Voltage    : {voltage_peak:.2f} V")
print(f"Sampling Rate   : {sampling_rate} Hz")
print(f"Number Samples  : {len(time)}")

# -------------------------------
# 6. Plot the waveform
# -------------------------------

plt.figure(figsize=(10, 4))

plt.plot(time, voltage)

plt.title("50 Hz AC Voltage Waveform")
plt.xlabel("Time (seconds)")
plt.ylabel("Voltage (V)")

plt.grid()
plt.title("Power Grid Voltage Sag Simulation")
plt.show()