# Arabic Dialect Fingerprinting System

## Overview
This project is an advanced AI-powered web application that analyzes spoken Arabic audio to determine its specific regional dialect. It identifies four primary Arabic dialects:
- **Egyptian** (مصري)
- **Gulf** (خليجي)
- **Levantine** (شامي)
- **Maghrebi** (مغربي)

Beyond classification, the system offers speech transcription, a rich explainability dashboard, audio mixing, and AI-driven cross-dialect voice conversion.

## Repository Structure
- `backend/`: FastAPI Python server containing the API routes, audio processing, and ML inference logic.
- `frontend/`: Next.js / React web application offering a rich, interactive, and responsive UI.
- `ML_Train/`: Machine Learning pipeline scripts for data preprocessing, feature extraction, training the XGBoost classifier, and UMAP dimensionality reduction.
- `backend/audio_samples/`: Curated reference audio recordings of the four dialects used for testing and baseline comparisons.

## Core Features & Technical Details

### 1. Audio Processing & Dialect Classification
The entire machine learning pipeline is built around a strict **16kHz sample rate**.
- When an audio file is uploaded, the backend uses `librosa` to load it at exactly 16,000 Hz. The audio is pre-emphasized and trimmed to the first 5 seconds.
- The `extract_features` function extracts exactly **162 statistical features** across 7 different acoustic properties:
  - **MFCC (0-39):** 20 Means, 20 Standard Deviations (Vocal Tract)
  - **Delta (40-79):** 20 Means, 20 Standard Deviations (Speech Dynamics)
  - **Delta² (80-119):** 20 Means, 20 Standard Deviations
  - **Chroma (120-143):** 12 Means, 12 Standard Deviations (Tonal Profile)
  - **Spectral Contrast (144-157):** 7 Means, 7 Standard Deviations (Spectral Shape)
  - **ZCR (158-159):** 1 Mean, 1 Standard Deviation
  - **RMS Energy (160-161):** 1 Mean, 1 Standard Deviation
- The 162-element feature vector is passed into the trained **XGBoost Classifier** pipeline (`arabic_dialect_xgboost_model.pkl`), which calculates the likelihood of the 4 dialects and outputs the predicted class.

### 2. High-Fidelity Explainability (SHAP & Heatmaps)
Instead of returning a "black-box" prediction, the dashboard uses the model's exact logic to explain *why* the prediction was made:
- **SHAP Value Analysis:** The backend runs `shap.TreeExplainer` on the XGBoost classifier. The dashboard plots the Top 7 features that drove the prediction. Green bars indicate features that *increased* confidence, while red bars indicate features that *decreased* confidence.
- **Category Heatmap:** The 162 SHAP values are grouped into 4 human-readable categories. A **Softmax** function is applied to the SHAP impacts across all 4 dialects, yielding a smooth `0-100%` distribution. This guarantees the heatmap is 100% mathematically aligned with the XGBoost prediction.

### 3. UMAP Visualization (2D Acoustic Space)
- A pre-fitted UMAP model projects the high-dimensional (162 features) audio into a 2D scatter plot.
- Utilizing K-Nearest Neighbors (KNN) interpolation, the backend dynamically places new user uploads onto the existing 2D projection map.
- The frontend visualizes where the uploaded audio sits relative to the training dataset, providing an intuitive, spatial understanding of dialect clustering.

### 4. Audio Transcription (Whisper)
- Uses OpenAI's **Whisper (Large model)** for accurate, state-of-the-art Arabic speech-to-text transcription.
- Operates locally on the backend to guarantee privacy and fast processing.

### 5. Dialect Conversion Pipeline
The user can select a target dialect and physically "convert" the uploaded speech:
- **Text Translation:** The transcribed Arabic text is rewritten into the target dialect's specific vocabulary and syntax using OpenAI's GPT API.
- **Speech Synthesis:** The translated text is synthesized back into natural-sounding speech using the **ElevenLabs API**. The system utilizes specific Voice IDs curated for each dialect to ensure cultural and phonetic accuracy.

### 6. Audio Mixer
- A dedicated interactive audio mixer interface allows users to blend tracks and manage audio playback directly in the browser.

## Setup & Installation

### Prerequisites
- Python 3.9+
- Node.js 18+
- API Keys: `OPENAI_API_KEY` (for dialect text conversion) and ElevenLabs setup.

### Backend Setup
Navigate to the `backend/` folder:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```
*Note: Make sure to set your `OPENAI_API_KEY` environment variable.*

### Frontend Setup
Navigate to the `frontend/` folder:
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` in your browser.

---
*Note on Repository Security: The git commit history has been sanitized to prevent the accidental leakage of sensitive API keys. Please always use environment variables for keys.*
