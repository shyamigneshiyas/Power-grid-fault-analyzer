from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained AI Model
model_path = os.path.join("models", "rf_fault_classifier.pkl")
model = joblib.load(model_path)

# Serve the HTML frontend at http://localhost:8000
@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

class GridRequest(BaseModel):
    fault_type: str

@app.post("/analyze")
def analyze_grid(request: GridRequest):
    print(f"Analyzing: '{request.fault_type}'") 
    
    # 1. Generate the base signal
    f = 50.0
    fs = 10000.0
    t = np.arange(0, 1.0, 1/fs)
    v_peak = 325.27
    signal = v_peak * np.sin(2 * np.pi * f * t)

    fault = request.fault_type.strip()

    if fault == "Voltage Sag":
        signal[3000:5000] *= 0.50
    elif fault == "Voltage Swell":
        signal[6000:8000] *= 1.50
    elif fault == "Overcurrent":
        signal[4000:] *= 2.0
    elif fault == "Harmonics":
        signal += (0.15 * v_peak * np.sin(2 * np.pi * 3 * f * t)) + \
                  (0.10 * v_peak * np.sin(2 * np.pi * 5 * f * t))

    # 2. Extract Features
    rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    fft_spectrum = np.abs(np.fft.rfft(signal))
    fundamental = fft_spectrum[int(f)]
    harmonics_rms = np.sqrt(np.sum(fft_spectrum[60:]**2))
    thd = float((harmonics_rms / (fundamental + 1e-6)) * 100)

    # 3. AI Prediction
    features = np.array([[rms, peak, thd]])
    diagnosis = str(model.predict(features)[0])
    
    # 4. Return JSON payload
    return {
        "time": t[::5].tolist(),
        "voltage": signal[::5].tolist(),
        "metrics": {"rms": round(rms, 2), "peak": round(peak, 2), "thd": round(thd, 2)},
        "diagnosis": diagnosis
    }