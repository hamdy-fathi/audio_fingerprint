"""
Hand-crafted classical audio features (MFCC, spectral, pitch, formants, rhythm).

No neural embeddings — librosa, scipy, and parselmouth (Praat) only.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

N_MFCC = 13


def _safe_stat(x: np.ndarray, fn) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    return float(fn(x))


def extract_mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, np.ndarray]:
    """MFCC + deltas; return summary statistics for fixed-length vector."""
    import librosa

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    out: Dict[str, np.ndarray] = {}
    for name, mat in (
        ("mfcc", mfcc),
        ("delta_mfcc", delta),
        ("delta2_mfcc", delta2),
    ):
        out[f"{name}_mean"] = np.array([_safe_stat(mat[i], np.mean) for i in range(mat.shape[0])])
        out[f"{name}_std"] = np.array([_safe_stat(mat[i], np.std) for i in range(mat.shape[0])])
    return out


def extract_spectral_features(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """Spectral centroid, bandwidth, rolloff, contrast, flatness, RMS, ZCR — mean & std over frames."""
    import librosa

    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)[0]

    feats: Dict[str, float] = {}
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

    # spectral contrast: multiple bands — aggregate mean/std across bands and time
    feats["spectral_contrast_mean"] = float(np.nanmean(contrast))
    feats["spectral_contrast_std"] = float(np.nanstd(contrast))
    return feats


def extract_pitch_features(
    y: np.ndarray,
    sr: int,
    fmin: float = 75.0,
    fmax: float = 500.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> Dict[str, float]:
    """F0 via librosa.pyin — mean, std, min, max, range."""
    import librosa

    f0_hz, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
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


def extract_formant_features(
    y: np.ndarray,
    sr: int,
    max_formant: float = 5500.0,
) -> Dict[str, float]:
    """
    F1, F2, F3 using Parselmouth (Praat). Writes a temporary WAV for reliability.

    Returns NaN-filled stats if Praat/parselmouth fails (caller may impute).
    """
    try:
        import parselmouth
    except ImportError:
        logger.warning("parselmouth not installed — formant features set to 0.")
        return _zero_formants()

    # Praat expects reasonable amplitude
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

        def ms(a: List[float]) -> Tuple[float, float]:
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
    except Exception as exc:
        logger.warning("Formant extraction failed (%s) — using zeros.", exc)
        return _zero_formants()
    finally:
        if tmp_path is not None and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def _zero_formants() -> Dict[str, float]:
    return {
        "f1_mean": 0.0,
        "f1_std": 0.0,
        "f2_mean": 0.0,
        "f2_std": 0.0,
        "f3_mean": 0.0,
        "f3_std": 0.0,
    }


def extract_rhythm_features(
    y: np.ndarray,
    sr: int,
    hop_length: int = 256,
    frame_length: int = 2048,
    energy_percentile: float = 20.0,
) -> Dict[str, float]:
    """
    Rhythm / timing proxies:

    - speech_rate: peaks per second in smoothed energy envelope (syllable-like rate proxy)
    - energy envelope mean/std (dB)
    - pause_ratio: fraction of frames below energy percentile threshold
    - voiced_unvoiced_ratio: fraction of frames classified as voiced via ZCR threshold
    """
    import librosa

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_safe = rms + 1e-12
    env_db = 20.0 * np.log10(rms_safe)

    # Smooth envelope for peak picking
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

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length
    )[0]
    zcr_thresh = np.median(zcr)
    voiced_ratio = float(np.mean(zcr < zcr_thresh))

    return {
        "speech_rate_proxy": float(speech_rate),
        "energy_env_mean_db": _safe_stat(env_db, np.mean),
        "energy_env_std_db": _safe_stat(env_db, np.std),
        "pause_ratio": pause_ratio,
        "voiced_unvoiced_ratio": voiced_ratio,
    }


FEATURE_ORDER: List[str] = []


def _build_feature_order() -> List[str]:
    """Deterministic global feature name order for ML + explainability."""
    names: List[str] = []
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

    names.extend(
        [
            "pitch_mean",
            "pitch_std",
            "pitch_min",
            "pitch_max",
            "pitch_range",
            "voiced_ratio_pyin",
        ]
    )

    names.extend(
        [
            "f1_mean",
            "f1_std",
            "f2_mean",
            "f2_std",
            "f3_mean",
            "f3_std",
        ]
    )

    names.extend(
        [
            "speech_rate_proxy",
            "energy_env_mean_db",
            "energy_env_std_db",
            "pause_ratio",
            "voiced_unvoiced_ratio",
        ]
    )
    return names


FEATURE_ORDER = _build_feature_order()


def _scalar_feature_keys() -> List[str]:
    """Scalar blocks appended after MFCC coefficients (same order as FEATURE_ORDER)."""
    n_mfcc_block = N_MFCC * 3 * 2
    return FEATURE_ORDER[n_mfcc_block:]


def extract_all_features_vector(
    y: np.ndarray,
    sr: int,
    include_formants: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute full fixed-size feature vector and ordered feature names.

    Returns
    -------
    vector : shape (n_features,)
    names : same length as vector
    """
    mf = extract_mfcc_features(y, sr)
    sp = extract_spectral_features(y, sr)
    pi = extract_pitch_features(y, sr)
    rh = extract_rhythm_features(y, sr)

    fd: Dict[str, np.ndarray | float] = {**mf, **sp, **pi, **rh}
    if include_formants:
        fo = extract_formant_features(y, sr)
        fd.update(fo)
    else:
        fd.update(_zero_formants())

    vec: List[float] = []
    for prefix in ("mfcc", "delta_mfcc", "delta2_mfcc"):
        for stat in ("mean", "std"):
            arr = mf[f"{prefix}_{stat}"]
            for i in range(N_MFCC):
                vec.append(float(arr[i]))

    for key in _scalar_feature_keys():
        vec.append(float(fd[key]))

    assert len(vec) == len(FEATURE_ORDER), "Feature vector length mismatch with FEATURE_ORDER."

    return np.array(vec, dtype=np.float32), FEATURE_ORDER


def get_mfcc_matrix(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = N_MFCC,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Return MFCC matrix (n_mfcc, T) for visualization."""
    import librosa

    return librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
    )
