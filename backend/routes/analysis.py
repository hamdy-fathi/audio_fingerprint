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
from dialect_classifier import predict_dialect
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["analysis"])


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

    try:
        result = predict_dialect(path)
    except Exception as e:
        raise HTTPException(500, f"Classification failed: {str(e)}")

    # Generate visualization charts
    prob_chart = None
    if result.get('probabilities'):
        prob_chart = plot_dialect_probabilities(result['probabilities'])

    return {
        'predicted_dialect': result['predicted_dialect'],
        'confidence': result.get('confidence', 0),
        'probabilities': result.get('probabilities', {}),
        'probability_chart': prob_chart,
    }


@router.post("/features")
async def get_features(req: AnalysisRequest):
    """Extract features and generate comparison visualizations."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    y, sr = load_audio(path)
    features = extract_features(y, sr)

    # Generate comparison charts (without dialect averages since the new classifier doesn't expose them)
    pitch_chart = plot_pitch_contour(y, sr, {})

    return {
        'features': features,
        'pitch_chart': pitch_chart
    }

