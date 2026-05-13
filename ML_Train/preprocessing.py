"""
Audio preprocessing for dialect identification: mono, resampling, normalization,
silence trimming, optional pre-emphasis (librosa + scipy only).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


def to_mono(y: np.ndarray) -> np.ndarray:
    """Convert multi-channel audio to mono by averaging channels."""
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    # shape (channels, samples) or (samples, channels)
    if y.shape[0] <= 8 and y.shape[0] < y.shape[-1]:
        return np.mean(y, axis=0).astype(np.float32)
    return np.mean(y, axis=1).astype(np.float32)


def resample_audio(
    y: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> Tuple[np.ndarray, int]:
    """Resample to ``target_sr`` using librosa."""
    import librosa

    if orig_sr == target_sr:
        return y.astype(np.float32, copy=False), target_sr
    y_rs = librosa.resample(y.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    return y_rs.astype(np.float32), target_sr


def normalize_peak(y: np.ndarray, peak: float = 1.0) -> np.ndarray:
    """Peak normalization to [-peak, peak] avoiding division by zero."""
    y = np.asarray(y, dtype=np.float32)
    m = np.max(np.abs(y)) + 1e-12
    return (y / m * peak).astype(np.float32)


def trim_silence(
    y: np.ndarray,
    top_db: float = 35.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading/trailing silence using librosa.effects.trim."""
    import librosa

    yt, _ = librosa.effects.trim(
        y,
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    if yt.size == 0:
        logger.warning("trim_silence removed entire signal — returning original.")
        return y
    return yt.astype(np.float32)


def pre_emphasis(y: np.ndarray, coeff: float = 0.97) -> np.ndarray:
    """Apply pre-emphasis filter y[n] - coeff * y[n-1]."""
    return signal.lfilter([1.0, -coeff], [1.0], y).astype(np.float32)


def preprocess_audio(
    y: np.ndarray,
    sample_rate: int,
    target_sr: int = 16_000,
    normalize: bool = True,
    trim: bool = True,
    trim_top_db: float = 35.0,
    apply_pre_emphasis: bool = False,
    preemph_coef: float = 0.97,
) -> Tuple[np.ndarray, int]:
    """
    Full preprocessing chain:

    1. Mono conversion
    2. Resampling to ``target_sr`` (default 16 kHz)
    3. Peak normalization (optional)
    4. Silence trimming (optional)
    5. Pre-emphasis (optional)

    Parameters
    ----------
    y : waveform (float32 recommended)
    sample_rate : original sampling rate in Hz
    target_sr : output sampling rate
    normalize : peak-normalize waveform
    trim : trim silence from start/end
    trim_top_db : threshold in dB for trim
    apply_pre_emphasis : apply FIR pre-emphasis
    preemph_coef : pre-emphasis coefficient

    Returns
    -------
    y_out : preprocessed mono waveform
    sr_out : sampling rate (equals ``target_sr``)
    """
    y = np.asarray(y, dtype=np.float32)
    y = to_mono(y)
    y, sr_out = resample_audio(y, sample_rate, target_sr)

    if trim:
        y = trim_silence(y, top_db=trim_top_db)

    if normalize:
        y = normalize_peak(y)

    if apply_pre_emphasis:
        y = pre_emphasis(y, coeff=preemph_coef)

    # Guard extremely short clips for downstream feature extractors
    min_len = int(0.05 * sr_out)
    if y.size < min_len:
        logger.warning(
            "Very short clip after preprocessing (%d samples). Padding zeros.", y.size
        )
        pad = min_len - y.size
        y = np.pad(y, (0, pad), mode="constant")

    return y.astype(np.float32), sr_out
