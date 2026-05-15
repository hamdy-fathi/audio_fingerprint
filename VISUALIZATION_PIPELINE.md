# Dialect Classification & Visualization Pipeline

This document explains the exact technical workflow of how the backend processes audio, predicts the Arabic dialect, and generates the highly accurate dual-chart visualization dashboard.

## 1. Audio Processing & Feature Extraction
The entire machine learning pipeline is built around a **16kHz sample rate**.
- When an audio file is uploaded, the backend uses `librosa` to load it at exactly 16,000 Hz. (This step is critical because extracting MFCCs at different sample rates corrupts the data).
- The audio is pre-emphasized and trimmed to the first 5 seconds.
- The `extract_features` function extracts exactly **162 statistical features** across 7 different acoustic properties:
  - **MFCC (0-39):** 20 Means, 20 Standard Deviations
  - **Delta (40-79):** 20 Means, 20 Standard Deviations
  - **Delta² (80-119):** 20 Means, 20 Standard Deviations
  - **Chroma (120-143):** 12 Means, 12 Standard Deviations
  - **Spectral Contrast (144-157):** 7 Means, 7 Standard Deviations
  - **ZCR (158-159):** 1 Mean, 1 Standard Deviation
  - **RMS Energy (160-161):** 1 Mean, 1 Standard Deviation

## 2. ML Inference (XGBoost)
The 162-element feature vector is passed into the trained `arabic_dialect_xgboost_model.pkl` pipeline.
- The pipeline scales the features using `StandardScaler`.
- The `XGBClassifier` calculates the likelihood of the 4 dialects (Egyptian, Gulf, Levantine, Maghrebi) and outputs the predicted class.

## 3. The Dual-Chart Explainability Dashboard
Instead of using naive mathematical distances, the dashboard now uses the model's exact logic to explain *why* the prediction was made.

### Chart 1: Top 7 Features Driving Prediction (SHAP)
- The backend runs `shap.TreeExplainer` on the XGBoost classifier.
- It calculates 162 SHAP values for the predicted dialect. These values represent the direct log-odds impact each feature had on the prediction.
- The top 7 largest absolute SHAP values are extracted and plotted on a horizontal bar chart.
- **Green Bars (Positive SHAP):** Features that *increased* the model's confidence in the predicted dialect.
- **Red Bars (Negative SHAP):** Features that *decreased* confidence (pushing towards another dialect).

### Chart 2: Feature Category Impact Heatmap
To show the user a broader "Category Match" without contradicting the ML model, the dashboard groups the 162 SHAP values into 4 human-readable categories:
1. **Vocal Tract (MFCC):** Features 0-39
2. **Speech Dynamics (Delta):** Features 40-119
3. **Tonal Profile (Chroma):** Features 120-143
4. **Spectral Shape & Contrast:** Features 144-161

For each category, the pipeline calculates the **Softmax** of the SHAP impacts across all 4 dialects:
1. It sums the SHAP values for the features inside that category for each dialect.
2. It applies a Softmax function (`exp(impact) / sum(exp(impacts))`) to convert these raw SHAP sums into a smooth `0-100%` distribution.
3. This guarantees that the dialect heavily favored by the XGBoost model for a specific category will score the highest percentage.

**Result:** The heatmap is 100% mathematically aligned with the XGBoost prediction, providing users with a clinically accurate and easily interpretable diagnostic scorecard.
