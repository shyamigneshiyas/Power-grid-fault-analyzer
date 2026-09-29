# ⚡ Power Grid Fault Analyzer (Full-Stack)

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103.1-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3.1-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly.js-Interactive-3f4f75.svg?logo=plotly)](https://plotly.com/javascript/)
## 🎯 Problem Statement

Power quality disturbances such as voltage sags, voltage swells, harmonics, and
overcurrent conditions can affect the reliability and performance of electrical
systems. Traditional protection systems generally rely on predefined thresholds
and protection rules.

This project explores an AI-assisted approach that analyzes electrical waveforms,
extracts signal characteristics using Digital Signal Processing (DSP), and uses
Machine Learning to classify different power-quality disturbances automatically.

## 🔄 How It Works

The system follows a complete signal-analysis and machine-learning pipeline:

```text
Electrical Waveform
        ↓
Data Preprocessing
        ↓
DSP Feature Extraction
        ↓
RMS / Peak / THD / FFT Features
        ↓
Random Forest Classifier
        ↓
Fault Classification
        ↓
FastAPI Backend
        ↓
Interactive Plotly Dashboard

An enterprise-grade, AI-powered diagnostic tool that uses Digital Signal Processing (DSP) and Machine Learning to instantly classify electrical power grid disturbances. 

Built with a decoupled architecture, this project features a high-performance **FastAPI** Python backend for mathematical processing and a custom **HTML/JavaScript** frontend with an interactive, MATLAB-style waveform viewer.

## 🚀 Key Features
* **Real-World Data Processing:** Trained on a 17,000-sample Kaggle dataset of physical power quality (PQ) events.
* **DSP Feature Extraction:** Automatically calculates RMS Voltage, Peak Voltage, and Total Harmonic Distortion (THD) using Fast Fourier Transforms (FFT).
* **AI Classification:** Utilizes a Scikit-Learn `RandomForestClassifier` to diagnose complex grid anomalies (Sags, Swells, Harmonics, Overcurrents, etc.).
* **Interactive Telemetry UI:** Custom frontend featuring scroll-wheel zooming and dynamic timeline rangesliders built with Plotly.js.

## 🏗️ Architecture & Tech Stack
* **Backend API:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn, Pandas, Joblib
* **Signal Processing:** NumPy (FFT algorithms)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript, Plotly.js

## 📂 Project Structure
```text
power_grid_analyzer/
│
├── archive/                        # Phase 1 & 2 experimental scripts
│   ├── app.py                      # Original Streamlit implementation
│   └── data_generator.py           # Synthetic data generation
│
├── data/
│   ├── (17 Kaggle CSV files)       # Raw electrical waveform data
│   └── master_kaggle_dataset.csv   # Aggregated 17,000-row master dataset
│
├── models/
│   └── rf_fault_classifier.pkl     # Trained Random Forest Model
│
├── frontend/
│   └── index.html                  # Interactive UI Dashboard
│
├── api.py                          # FastAPI Backend Server
├── train_model.py                  # ML Training Pipeline
├── combine_kaggle_data.py          # Data Aggregation Script
└── requirements.txt                # Python Dependencies
