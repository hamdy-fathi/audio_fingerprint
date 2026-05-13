"""
Audio mixer routes - mix two files and classify result.
"""
import os, tempfile
import soundfile as sf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from audio_processor import (
    load_audio, mix_audio, extract_features,
    compute_spectrogram, compute_mel_spectrogram,
    audio_to_base64_wav
)
from feature_visualizer import plot_dialect_probabilities
from dialect_classifier import predict_dialect
from routes.upload import get_file_path

router = APIRouter(prefix="/api", tags=["mixer"])


class MixRequest(BaseModel):
    file_id_1: str
    file_id_2: str
    weight: float  # 0.0 = 100% file1, 1.0 = 100% file2


@router.post("/mix-and-classify")
async def mix_and_classify(req: MixRequest):
    """Mix two audio files and classify the result."""
    path1 = get_file_path(req.file_id_1)
    path2 = get_file_path(req.file_id_2)

    if not path1 or not os.path.exists(path1):
        raise HTTPException(404, "File 1 not found")
    if not path2 or not os.path.exists(path2):
        raise HTTPException(404, "File 2 not found")
    if not 0 <= req.weight <= 1:
        raise HTTPException(400, "Weight must be between 0 and 1")

    # Load both files
    y1, sr1 = load_audio(path1)
    y2, sr2 = load_audio(path2)

    # Mix
    mixed, mixed_sr = mix_audio(y1, sr1, y2, sr2, req.weight)

    # Generate visualizations for mixed audio
    spec = compute_spectrogram(mixed, mixed_sr)
    mel_spec = compute_mel_spectrogram(mixed, mixed_sr)

    # Classify mixed audio — write to a temp file since predict_dialect takes a path
    result = {}
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, mixed, mixed_sr)
            tmp_path = tmp.name

        result = predict_dialect(tmp_path)
        if result.get('probabilities'):
            result['probability_chart'] = plot_dialect_probabilities(result['probabilities'])
    except Exception as e:
        result = {'error': f'Classification failed: {str(e)}'}
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Convert mixed audio to base64 for playback
    mixed_audio_b64 = audio_to_base64_wav(mixed, mixed_sr)

    return {
        'spectrogram': spec,
        'mel_spectrogram': mel_spec,
        'mixed_audio': mixed_audio_b64,
        'weight': req.weight,
        'classification': result.get('predicted_dialect', 'unknown'),
        'confidence': result.get('confidence', 0),
        'probabilities': result.get('probabilities', {}),
        'probability_chart': result.get('probability_chart', None)
    }

