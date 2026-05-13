import os
import joblib
import librosa
import numpy as np


BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "arabic_dialect_xgboost_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


def extract_features(audio_array, sr=16000):
    y = audio_array.astype(np.float32)

    if len(y) < sr:
        y = np.pad(y, (0, sr - len(y)))

    # Keep this ONLY if you trained with 5 seconds max
    y = y[:sr * 5]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = []

    for feat in [mfcc, delta, delta2, chroma, contrast, zcr, rms]:
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))

    return np.array(features)


def predict_dialect(file_path):
    audio, sr = librosa.load(file_path, sr=16000)

    features = extract_features(audio, sr)
    features = features.reshape(1, -1)

    pred_num = model.predict(features)
    pred_label = label_encoder.inverse_transform(pred_num)[0]

    result = {
        "predicted_dialect": pred_label
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]

        result["confidence"] = float(np.max(probs))
        result["probabilities"] = {
            str(label_encoder.classes_[i]): float(probs[i])
            for i in range(len(label_encoder.classes_))
        }

    return result