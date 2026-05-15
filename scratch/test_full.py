import sys
sys.path.append("backend")
from dialect_classifier import extract_features as clf_extract, predict_dialect
from feature_visualizer import plot_feature_breakdown
from audio_processor import load_audio, extract_features
from feature_visualizer import extract_extra_features, get_dialect_averages

# We need a sample file
file_path = "backend/uploads/sample.wav"
import os
import soundfile as sf
import numpy as np

if not os.path.exists(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    # create a dummy wav file
    sf.write(file_path, np.random.randn(16000), 16000)

print("Testing plot_feature_breakdown...")
try:
    y, sr = load_audio(file_path)
    file_feats = extract_features(y, sr, include_formants=False)
    file_feats.update(extract_extra_features(y, sr))
    arr_162 = clf_extract(y, sr)
    dialect_avgs = get_dialect_averages()
    
    b64 = plot_feature_breakdown(arr_162, file_feats, dialect_avgs)
    print("Heatmap generated successfully, len:", len(b64))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("Feature breakdown failed:", e)
