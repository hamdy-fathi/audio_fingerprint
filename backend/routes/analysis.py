"""
Analysis routes: spectrogram, classification, features with full dialect comparison plots.
"""
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from audio_processor import load_audio, extract_features, compute_spectrogram, compute_mel_spectrogram, compute_waveform
from feature_visualizer import (
    get_dialect_averages, extract_extra_features, _extract_simple_features,
    plot_dialect_similarity, plot_feature_breakdown,
    plot_mfcc_heatmap, plot_mfcc_comparison, plot_spectral_radar,
    plot_pitch_contour, plot_energy_rhythm, plot_formant_scatter,
    plot_delta_comparison, plot_chromagram, plot_dialect_probabilities,
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
    """Extract features and generate all dialect comparison plots."""
    path = get_file_path(req.file_id)
    if not path or not os.path.exists(path):
        raise HTTPException(404, "File not found")

    y, sr = load_audio(path)

    # Extract features using the REAL audio_processor (same as classifier)
    file_feats = extract_features(y, sr, include_formants=False)
    
    # Extract extra features (chroma, per-band contrast) for better visualization
    extra_feats = extract_extra_features(y, sr)
    file_feats.update(extra_feats)

    # Extract the 162-element array for the ML classifier & SHAP (MUST BE 16kHz)
    import librosa
    audio_16k, sr_16k = librosa.load(path, sr=16000)
    from dialect_classifier import extract_features as clf_extract
    arr_162 = clf_extract(audio_16k, sr_16k)

    # Get dialect reference averages (cached after first call)
    dialect_avgs = get_dialect_averages()

    # Generate the 2 key comparison plots
    charts = {}
    try:
        charts['feature_breakdown'] = plot_feature_breakdown(arr_162, file_feats, dialect_avgs)
    except Exception as e:
        print(f"feature_breakdown error: {e}")

    try:
        charts['pitch_contour'] = plot_pitch_contour(y, sr, dialect_avgs)
    except Exception as e:
        print(f"pitch_contour error: {e}")

    return charts
