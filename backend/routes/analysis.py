"""
Analysis routes: spectrogram, classification, features.
"""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from audio_processor import (
    load_audio, compute_spectrogram, compute_mel_spectrogram,
    compute_waveform, extract_features
)
from feature_visualizer import (
    plot_mfcc_comparison, plot_spectral_radar,
    plot_feature_importance, plot_pitch_contour, plot_dialect_probabilities
)
from dialect_classifier import DialectClassifier
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["analysis"])
classifier = DialectClassifier()


class AnalysisRequest(BaseModel):
    file_id: str


@router.post("/spectrogram")
async def get_spectrogram(req: AnalysisRequest):
    """Generate spectrogram visualizations for a file."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    y, sr = load_audio(path)
    spec = compute_spectrogram(y, sr)
    mel_spec = compute_mel_spectrogram(y, sr)
    waveform = compute_waveform(y, sr)

    return {
        'spectrogram': spec,
        'mel_spectrogram': mel_spec,
        'waveform': waveform,
        'duration': round(len(y) / sr, 2),
        'sample_rate': sr
    }


@router.post("/classify")
async def classify_dialect(req: AnalysisRequest):
    """Classify the dialect of an audio file."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    if not classifier.is_trained():
        raise HTTPException(500, "Model not trained. Run train_model.py first.")

    y, sr = load_audio(path)
    features = extract_features(y, sr)
    result = classifier.classify(features)

    # Generate visualization charts
    prob_chart = plot_dialect_probabilities(result['probabilities'])

    importance_chart = None
    if result.get('feature_importances') and result.get('feature_names'):
        import numpy as np
        importance_chart = plot_feature_importance(
            np.array(result['feature_importances']),
            result['feature_names']
        )

    return {
        'predicted_dialect': result['predicted_dialect'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities'],
        'svm_prediction': result['svm_prediction'],
        'probability_chart': prob_chart,
        'importance_chart': importance_chart
    }


@router.post("/features")
async def get_features(req: AnalysisRequest):
    """Extract features and generate comparison visualizations."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    y, sr = load_audio(path)
    features = extract_features(y, sr)

    # Get dialect averages for comparison
    dialect_avg = classifier.dialect_avg_features if classifier.is_trained() else {}
    pitch_ranges = classifier.dialect_pitch_ranges if classifier.is_trained() else {}

    # Generate comparison charts
    mfcc_chart = plot_mfcc_comparison(features, dialect_avg) if dialect_avg else None
    radar_chart = plot_spectral_radar(features, dialect_avg) if dialect_avg else None
    pitch_chart = plot_pitch_contour(y, sr, pitch_ranges) if pitch_ranges else None

    return {
        'features': features,
        'mfcc_chart': mfcc_chart,
        'radar_chart': radar_chart,
        'pitch_chart': pitch_chart
    }
