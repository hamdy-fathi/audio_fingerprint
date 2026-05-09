"""
Classic ML dialect classifier (Random Forest + SVM). No deep learning.
"""
import os, json, joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from audio_processor import load_audio, extract_features, get_feature_vector, get_feature_names

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'audio_samples')
DIALECTS = ['egyptian', 'gulf', 'levantine', 'maghrebi']


class DialectClassifier:
    def __init__(self):
        self.rf_model = None
        self.svm_model = None
        self.scaler = None
        self.feature_names = None
        self.dialect_avg_features = {}
        self.dialect_pitch_ranges = {}
        self._load_models()

    def _load_models(self):
        rf_path = os.path.join(MODELS_DIR, 'dialect_rf_model.joblib')
        svm_path = os.path.join(MODELS_DIR, 'dialect_svm_model.joblib')
        scaler_path = os.path.join(MODELS_DIR, 'feature_scaler.joblib')
        names_path = os.path.join(MODELS_DIR, 'feature_names.json')
        avg_path = os.path.join(MODELS_DIR, 'dialect_avg_features.json')
        pitch_path = os.path.join(MODELS_DIR, 'dialect_pitch_ranges.json')

        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
            self.svm_model = joblib.load(svm_path)
            self.scaler = joblib.load(scaler_path)
            with open(names_path, 'r') as f:
                self.feature_names = json.load(f)
            if os.path.exists(avg_path):
                with open(avg_path, 'r') as f:
                    self.dialect_avg_features = json.load(f)
            if os.path.exists(pitch_path):
                with open(pitch_path, 'r') as f:
                    self.dialect_pitch_ranges = json.load(f)

    def is_trained(self):
        return self.rf_model is not None

    def classify(self, features_dict):
        if not self.is_trained():
            return {'error': 'Model not trained. Run train_model.py first.'}
        
        vec = get_feature_vector(features_dict).reshape(1, -1)
        vec_scaled = self.scaler.transform(vec)

        # Random Forest prediction
        rf_pred = self.rf_model.predict(vec_scaled)[0]
        rf_proba = self.rf_model.predict_proba(vec_scaled)[0]
        rf_classes = self.rf_model.classes_

        # SVM prediction
        svm_pred = self.svm_model.predict(vec_scaled)[0]
        svm_proba = self.svm_model.predict_proba(vec_scaled)[0]

        probabilities = {cls: float(prob) for cls, prob in zip(rf_classes, rf_proba)}
        importances = self.rf_model.feature_importances_

        return {
            'predicted_dialect': rf_pred,
            'confidence': float(max(rf_proba)),
            'probabilities': probabilities,
            'svm_prediction': svm_pred,
            'svm_probabilities': {cls: float(p) for cls, p in zip(rf_classes, svm_proba)},
            'feature_importances': importances.tolist(),
            'feature_names': self.feature_names,
            'dialect_avg_features': self.dialect_avg_features,
            'dialect_pitch_ranges': self.dialect_pitch_ranges
        }


def train_models():
    """Train Random Forest and SVM on the audio samples."""
    print("=" * 60)
    print("Training Arabic Dialect Classifier")
    print("=" * 60)

    X, y = [], []
    all_features_by_dialect = {d: [] for d in DIALECTS}
    feature_names = None

    for dialect in DIALECTS:
        dialect_dir = os.path.join(SAMPLES_DIR, dialect)
        if not os.path.exists(dialect_dir):
            print(f"  WARNING: {dialect_dir} not found, skipping")
            continue
        
        files = [f for f in os.listdir(dialect_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
        print(f"\n  [{dialect.upper()}] Found {len(files)} files")

        for fname in files:
            fpath = os.path.join(dialect_dir, fname)
            print(f"    Processing: {fname}...")
            try:
                audio, sr = load_audio(fpath)
                feats = extract_features(audio, sr)
                vec = get_feature_vector(feats)
                X.append(vec)
                y.append(dialect)
                all_features_by_dialect[dialect].append(feats)
                if feature_names is None:
                    feature_names = get_feature_names(feats)
            except Exception as e:
                print(f"    ERROR processing {fname}: {e}")

    if len(X) == 0:
        print("No audio files found! Place .wav files in audio_samples/<dialect>/ folders.")
        return

    X = np.array(X)
    y = np.array(y)
    print(f"\nTotal samples: {len(X)}, Features per sample: {X.shape[1]}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf.fit(X_scaled, y)
    if len(X) >= 4:
        rf_scores = cross_val_score(rf, X_scaled, y, cv=min(4, len(X)), scoring='accuracy')
        print(f"  RF CV Accuracy: {rf_scores.mean():.2f} (+/- {rf_scores.std():.2f})")
    else:
        print(f"  RF Training Accuracy: {rf.score(X_scaled, y):.2f}")

    # Train SVM
    print("Training SVM...")
    svm = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
    svm.fit(X_scaled, y)
    if len(X) >= 4:
        svm_scores = cross_val_score(svm, X_scaled, y, cv=min(4, len(X)), scoring='accuracy')
        print(f"  SVM CV Accuracy: {svm_scores.mean():.2f} (+/- {svm_scores.std():.2f})")
    else:
        print(f"  SVM Training Accuracy: {svm.score(X_scaled, y):.2f}")

    # Compute dialect averages
    dialect_avg_features = {}
    dialect_pitch_ranges = {}
    for dialect in DIALECTS:
        if all_features_by_dialect[dialect]:
            avg = {}
            for key in all_features_by_dialect[dialect][0]:
                vals = [f[key] for f in all_features_by_dialect[dialect]]
                avg[key] = float(np.mean(vals))
            dialect_avg_features[dialect] = avg

            pitches = [f['pitch_mean'] for f in all_features_by_dialect[dialect]]
            pitch_stds = [f['pitch_std'] for f in all_features_by_dialect[dialect]]
            dialect_pitch_ranges[dialect] = [
                float(np.mean(pitches) - np.mean(pitch_stds)),
                float(np.mean(pitches) + np.mean(pitch_stds))
            ]

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODELS_DIR, 'dialect_rf_model.joblib'))
    joblib.dump(svm, os.path.join(MODELS_DIR, 'dialect_svm_model.joblib'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'feature_scaler.joblib'))
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(feature_names, f)
    with open(os.path.join(MODELS_DIR, 'dialect_avg_features.json'), 'w') as f:
        json.dump(dialect_avg_features, f)
    with open(os.path.join(MODELS_DIR, 'dialect_pitch_ranges.json'), 'w') as f:
        json.dump(dialect_pitch_ranges, f)

    print(f"\nModels saved to {MODELS_DIR}")
    print("Top 10 important features:")
    indices = np.argsort(rf.feature_importances_)[-10:][::-1]
    for i in indices:
        print(f"  {feature_names[i]}: {rf.feature_importances_[i]:.4f}")
    print("\nDone!")


if __name__ == '__main__':
    train_models()
