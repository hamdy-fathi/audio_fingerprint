"""
Audio processing module for Arabic Dialect Detection.
Handles loading, spectrogram generation, feature extraction, and audio mixing.
Uses ML_Train feature extraction for compatibility with trained models.
"""
import io, base64, tempfile
from pathlib import Path
import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf

# Feature configuration
N_MFCC = 13


def _safe_stat(x: np.ndarray, fn) -> float:
    """Safely compute statistics, handling NaN/inf values."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(fn(x))


def load_audio(file_path, sr=22050, duration=None):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    y = librosa.util.normalize(y)
    return y, sr


def compute_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Spectrogram', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.xaxis.label.set_color('#F5E6D3')
    ax.yaxis.label.set_color('#F5E6D3')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_mel_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Mel Spectrogram', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.xaxis.label.set_color('#F5E6D3')
    ax.yaxis.label.set_color('#F5E6D3')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def compute_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor('#1A1410')
    ax.set_facecolor('#1A1410')
    times = np.arange(len(y)) / sr
    ax.plot(times, y, color='#7C3AED', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('Time (s)', color='#F5E6D3')
    ax.set_ylabel('Amplitude', color='#F5E6D3')
    ax.set_title('Waveform', color='#F5E6D3', fontsize=14)
    ax.tick_params(colors='#F5E6D3')
    ax.set_xlim([0, times[-1]])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#1A1410', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def extract_mfcc_features(y, sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512):
    """MFCC + deltas; return summary statistics for fixed-length vector."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    out = {}
    for name, mat in (("mfcc", mfcc), ("delta_mfcc", delta), ("delta2_mfcc", delta2)):
        out[f"{name}_mean"] = np.array([_safe_stat(mat[i], np.mean) for i in range(mat.shape[0])])
        out[f"{name}_std"] = np.array([_safe_stat(mat[i], np.std) for i in range(mat.shape[0])])
    return out


def extract_spectral_features(y, sr, n_fft=2048, hop_length=512):
    """Spectral centroid, bandwidth, rolloff, contrast, flatness, RMS, ZCR."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

    feats = {}
    for tag, arr in (
        ("spectral_centroid", cent),
        ("spectral_bandwidth", bw),
        ("spectral_rolloff", rolloff),
        ("spectral_flatness", flatness),
        ("rms_energy", rms),
        ("zero_crossing_rate", zcr),
    ):
        feats[f"{tag}_mean"] = _safe_stat(arr, np.mean)
        feats[f"{tag}_std"] = _safe_stat(arr, np.std)
    
    feats["spectral_contrast_mean"] = float(np.nanmean(contrast))
    feats["spectral_contrast_std"] = float(np.nanstd(contrast))
    return feats


def extract_pitch_features(y, sr, fmin=75.0, fmax=500.0, frame_length=2048, hop_length=512):
    """F0 via librosa.pyin — mean, std, min, max, range."""
    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop_length
    )
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    valid = np.isfinite(f0_hz) & (f0_hz > 0)
    if not np.any(valid):
        return {
            "pitch_mean": 0.0,
            "pitch_std": 0.0,
            "pitch_min": 0.0,
            "pitch_max": 0.0,
            "pitch_range": 0.0,
            "voiced_ratio_pyin": float(np.mean(voiced_prob > 0.1)),
        }
    pv = f0_hz[valid]
    return {
        "pitch_mean": float(np.mean(pv)),
        "pitch_std": float(np.std(pv)),
        "pitch_min": float(np.min(pv)),
        "pitch_max": float(np.max(pv)),
        "pitch_range": float(np.max(pv) - np.min(pv)),
        "voiced_ratio_pyin": float(np.mean(voiced_prob > 0.1)),
    }


def _zero_formants():
    """Return zero formants when extraction fails."""
    return {
        "f1_mean": 0.0,
        "f1_std": 0.0,
        "f2_mean": 0.0,
        "f2_std": 0.0,
        "f3_mean": 0.0,
        "f3_std": 0.0,
    }


def extract_formant_features(y, sr, max_formant=5500.0):
    """F1, F2, F3 using Parselmouth (Praat)."""
    try:
        import parselmouth
    except ImportError:
        return _zero_formants()

    y = np.asarray(y, dtype=np.float64)
    y = y / (np.max(np.abs(y)) + 1e-12)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = Path(tf.name)
        sf.write(tmp_path, y, sr, subtype="PCM_16")
        
        snd = parselmouth.Sound(str(tmp_path))
        formant = snd.to_formant_burg(max_formant=max_formant)
        duration = snd.duration
        times = np.arange(0.0, duration, 0.01)

        f1s, f2s, f3s = [], [], []
        for t in times:
            f1 = formant.get_value_at_time(1, t)
            f2 = formant.get_value_at_time(2, t)
            f3 = formant.get_value_at_time(3, t)
            if np.isfinite(f1) and f1 > 0:
                f1s.append(f1)
            if np.isfinite(f2) and f2 > 0:
                f2s.append(f2)
            if np.isfinite(f3) and f3 > 0:
                f3s.append(f3)

        def ms(a):
            if not a:
                return 0.0, 0.0
            v = np.array(a, dtype=np.float64)
            return float(np.mean(v)), float(np.std(v))

        m1, s1 = ms(f1s)
        m2, s2 = ms(f2s)
        m3, s3 = ms(f3s)

        return {
            "f1_mean": m1,
            "f1_std": s1,
            "f2_mean": m2,
            "f2_std": s2,
            "f3_mean": m3,
            "f3_std": s3,
        }
    except Exception:
        return _zero_formants()
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def extract_rhythm_features(y, sr, hop_length=256, frame_length=2048, energy_percentile=20.0):
    """Rhythm / timing proxies: speech rate, energy envelope, pause ratio, voicing ratio."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_safe = rms + 1e-12
    env_db = 20.0 * np.log10(rms_safe)

    env_smooth = np.convolve(rms, np.ones(5) / 5.0, mode="same")
    peaks = []
    for i in range(1, len(env_smooth) - 1):
        if env_smooth[i] > env_smooth[i - 1] and env_smooth[i] > env_smooth[i + 1]:
            if env_smooth[i] > np.max(env_smooth) * 0.3:
                peaks.append(i)
    
    hop_sec = hop_length / float(sr)
    speech_rate = len(peaks) / max(len(env_smooth) * hop_sec, 1e-6)

    thresh = np.percentile(rms, energy_percentile)
    pause_ratio = float(np.mean(rms < thresh))

    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    zcr_thresh = np.median(zcr)
    voiced_ratio = float(np.mean(zcr < zcr_thresh))

    return {
        "speech_rate_proxy": float(speech_rate),
        "energy_env_mean_db": _safe_stat(env_db, np.mean),
        "energy_env_std_db": _safe_stat(env_db, np.std),
        "pause_ratio": pause_ratio,
        "voiced_unvoiced_ratio": voiced_ratio,
    }


# Deterministic feature order matching ML_Train
FEATURE_ORDER = []

def _build_feature_order():
    """Deterministic global feature name order for ML + explainability."""
    names = []
    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        for stat in ("mean", "std"):
            for i in range(N_MFCC):
                names.append(f"{prefix}_{stat}_{i}")

    spec_keys = [
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_bandwidth_mean",
        "spectral_bandwidth_std",
        "spectral_rolloff_mean",
        "spectral_rolloff_std",
        "spectral_contrast_mean",
        "spectral_contrast_std",
        "spectral_flatness_mean",
        "spectral_flatness_std",
        "rms_energy_mean",
        "rms_energy_std",
        "zero_crossing_rate_mean",
        "zero_crossing_rate_std",
    ]
    names.extend(spec_keys)

    names.extend([
        "pitch_mean",
        "pitch_std",
        "pitch_min",
        "pitch_max",
        "pitch_range",
        "voiced_ratio_pyin",
    ])

    names.extend([
        "f1_mean",
        "f1_std",
        "f2_mean",
        "f2_std",
        "f3_mean",
        "f3_std",
    ])

    names.extend([
        "speech_rate_proxy",
        "energy_env_mean_db",
        "energy_env_std_db",
        "pause_ratio",
        "voiced_unvoiced_ratio",
    ])
    return names

FEATURE_ORDER = _build_feature_order()

def _scalar_feature_keys():
    """Scalar blocks appended after MFCC coefficients."""
    n_mfcc_block = N_MFCC * 3 * 2
    return FEATURE_ORDER[n_mfcc_block:]


def extract_features(y, sr, include_formants=True):
    """Extract complete fixed-size feature vector matching ML_Train."""
    mf = extract_mfcc_features(y, sr)
    sp = extract_spectral_features(y, sr)
    pi = extract_pitch_features(y, sr)
    rh = extract_rhythm_features(y, sr)

    fd = {**mf, **sp, **pi, **rh}
    if include_formants:
        fo = extract_formant_features(y, sr)
        fd.update(fo)
    else:
        fd.update(_zero_formants())

    vec = []
    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        for stat in ("mean", "std"):
            arr = mf[f"{prefix}_{stat}"]
            for i in range(N_MFCC):
                vec.append(float(arr[i]))

    for key in _scalar_feature_keys():
        vec.append(float(fd[key]))

    # Create ordered dict for compatibility
    features_dict = {}
    for name, val in zip(FEATURE_ORDER, vec):
        features_dict[name] = val
    
    return features_dict


def get_feature_vector(features_dict):
    """Convert feature dict to vector using FEATURE_ORDER."""
    return np.array([features_dict.get(name, 0.0) for name in FEATURE_ORDER], dtype=np.float32)


def get_feature_names(features_dict):
    """Return ordered feature names matching FEATURE_ORDER."""
    return FEATURE_ORDER


def mix_audio(y1, sr1, y2, sr2, weight, target_sr=22050):
    if sr1 != target_sr:
        y1 = librosa.resample(y1, orig_sr=sr1, target_sr=target_sr)
    if sr2 != target_sr:
        y2 = librosa.resample(y2, orig_sr=sr2, target_sr=target_sr)
    max_len = max(len(y1), len(y2))
    if len(y1) < max_len:
        y1 = np.pad(y1, (0, max_len - len(y1)))
    if len(y2) < max_len:
        y2 = np.pad(y2, (0, max_len - len(y2)))
    mixed = (1 - weight) * y1 + weight * y2
    mixed = librosa.util.normalize(mixed)
    return mixed, target_sr


def audio_to_wav_bytes(y, sr):
    buf = io.BytesIO()
    sf.write(buf, y, sr, format='WAV')
    buf.seek(0)
    return buf.read()

def audio_to_base64_wav(y, sr):
    return base64.b64encode(audio_to_wav_bytes(y, sr)).decode('utf-8')
