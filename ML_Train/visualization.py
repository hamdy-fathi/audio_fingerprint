"""
Publication-oriented exploratory plots for dialect comparison (classical features only).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from utils import PLOTS_DIR

logger = logging.getLogger(__name__)

ACCENT_COLORS = sns.color_palette("deep", 8)


def _set_pub_style() -> None:
    sns.set_theme(context="talk", style="whitegrid", font_scale=0.9)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["axes.labelsize"] = 11


def _save_fig(fig: Figure, name: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot %s", out)
    return out


def plot_waveforms_by_dialect(
    waveforms: Sequence[np.ndarray],
    dialect_labels: Sequence[str],
    sample_rate: int,
    max_samples_per_class: int = 2,
    seed: int = 42,
) -> Path:
    """Overlay normalized waveform snippets per dialect class."""
    _set_pub_style()
    rng = np.random.default_rng(seed)
    df_labels = np.array(dialect_labels)
    classes = sorted(set(df_labels))
    fig, axes = plt.subplots(len(classes), 1, figsize=(11, 2.2 * len(classes)), sharex=True)
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        idx = np.where(df_labels == cls)[0]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(max_samples_per_class, len(idx)), replace=False)
        t_last = 4.0
        for j, p in enumerate(pick):
            y = np.asarray(waveforms[p], dtype=np.float32)
            y = y / (np.max(np.abs(y)) + 1e-12)
            t = np.arange(len(y)) / float(sample_rate)
            t_last = float(t[-1]) if len(t) else t_last
            ax.plot(t, y + j * 0.2, color=ACCENT_COLORS[j % len(ACCENT_COLORS)], lw=0.8, alpha=0.85)
        ax.set_ylabel(cls, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_xlim(0, min(4.0, t_last))

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Waveforms by dialect (normalized, staggered)", y=1.01)
    return _save_fig(fig, "01_waveforms_by_dialect.png")


def plot_spectrograms_grid(
    waveforms: Sequence[np.ndarray],
    dialect_labels: Sequence[str],
    sample_rate: int,
    n_examples: int = 5,
    seed: int = 42,
) -> Path:
    """One spectrogram per dialect (random exemplar)."""
    import librosa

    _set_pub_style()
    rng = np.random.default_rng(seed)
    labels = np.array(dialect_labels)
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(classes), figsize=(3.6 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        idx = np.where(labels == cls)[0]
        i = int(rng.choice(idx))
        y = np.asarray(waveforms[i], dtype=np.float32)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        S_db = librosa.amplitude_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_db, sr=sample_rate, x_axis="time", y_axis="hz", ax=ax)
        ax.set_title(cls)
        fig.colorbar(img, ax=ax, format="%+2.0f dB", shrink=0.65)

    fig.suptitle("Log-magnitude spectrograms (STFT)", y=1.02)
    return _save_fig(fig, "02_spectrograms_by_dialect.png")


def plot_mfcc_heatmaps(
    waveforms: Sequence[np.ndarray],
    dialect_labels: Sequence[str],
    sample_rate: int,
    seed: int = 42,
) -> Path:
    """MFCC heatmaps for one random clip per dialect."""
    import librosa

    _set_pub_style()
    rng = np.random.default_rng(seed)
    labels = np.array(dialect_labels)
    classes = sorted(set(labels))
    fig, axes = plt.subplots(1, len(classes), figsize=(3.8 * len(classes), 4.5))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        idx = np.where(labels == cls)[0]
        i = int(rng.choice(idx))
        y = np.asarray(waveforms[i], dtype=np.float32)
        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=13)
        img = librosa.display.specshow(mfcc, x_axis="time", sr=sample_rate, hop_length=512, ax=ax)
        ax.set_title(cls)
        fig.colorbar(img, ax=ax, shrink=0.65)

    fig.suptitle("MFCC trajectories (13 coeffs)", y=1.02)
    return _save_fig(fig, "03_mfcc_heatmaps.png")


def plot_spectral_centroid_tracks(
    waveforms: Sequence[np.ndarray],
    dialect_labels: Sequence[str],
    sample_rate: int,
    seed: int = 42,
) -> Path:
    import librosa

    _set_pub_style()
    rng = np.random.default_rng(seed)
    labels = np.array(dialect_labels)
    classes = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    for k, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        i = int(rng.choice(idx))
        y = np.asarray(waveforms[i], dtype=np.float32)
        cent = librosa.feature.spectral_centroid(y=y, sr=sample_rate)[0]
        times = librosa.frames_to_time(np.arange(len(cent)), sr=sample_rate, hop_length=512)
        ax.plot(times, cent / 1000.0, label=cls, color=ACCENT_COLORS[k % len(ACCENT_COLORS)], lw=1.5)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spectral centroid (kHz)")
    ax.set_title("Spectral centroid contours — exemplars per dialect")
    ax.legend(ncol=2, fontsize=9)
    return _save_fig(fig, "04_spectral_centroid_tracks.png")


def plot_pitch_contours(
    waveforms: Sequence[np.ndarray],
    dialect_labels: Sequence[str],
    sample_rate: int,
    seed: int = 42,
) -> Path:
    import librosa

    _set_pub_style()
    rng = np.random.default_rng(seed)
    labels = np.array(dialect_labels)
    classes = sorted(set(labels))
    fig, ax = plt.subplots(figsize=(10, 5))

    for k, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        i = int(rng.choice(idx))
        y = np.asarray(waveforms[i], dtype=np.float32)
        f0, _, voiced_prob = librosa.pyin(y, fmin=75, fmax=500, sr=sample_rate)
        t = librosa.times_like(f0, sr=sample_rate, hop_length=512)
        mask = np.isfinite(f0)
        ax.plot(
            t[mask],
            f0[mask],
            label=cls,
            color=ACCENT_COLORS[k % len(ACCENT_COLORS)],
            lw=1.3,
            alpha=0.85,
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title("Pitch contours (PYIN) — exemplars per dialect")
    ax.legend(ncol=2, fontsize=9)
    return _save_fig(fig, "05_pitch_contours.png")


def plot_formant_space(
    f1_means: Sequence[float],
    f2_means: Sequence[float],
    dialect_labels: Sequence[str],
) -> Path:
    """Scatter / convex hull style F1 vs F2 by dialect (Hz)."""
    _set_pub_style()
    df = pd.DataFrame({"F1": f1_means, "F2": f2_means, "dialect": dialect_labels})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="F2",
        y="F1",
        hue="dialect",
        ax=ax,
        alpha=0.55,
        s=45,
        palette="deep",
        edgecolor="none",
    )
    ax.invert_yaxis()
    ax.set_title("Formant space (F1 vs F2 means per utterance)")
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_fig(fig, "06_formant_space_f1_f2.png")


def plot_embedding_scatter_2d(
    coords: np.ndarray,
    dialect_labels: Sequence[str],
    title: str,
    filename: str,
) -> Path:
    """Generic 2-D embedding scatter (PCA / t-SNE)."""
    _set_pub_style()
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "dialect": dialect_labels})
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    sns.scatterplot(data=df, x="x", y="y", hue="dialect", ax=ax, alpha=0.65, s=38, palette="deep")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    return _save_fig(fig, filename)


def plot_feature_importance(
    importances: Sequence[float],
    feature_names: Sequence[str],
    title: str,
    filename: str,
    top_k: int = 25,
) -> Path:
    """Horizontal bar chart of importance scores."""
    _set_pub_style()
    imp = np.asarray(importances, dtype=np.float64)
    order = np.argsort(imp)[::-1][:top_k]
    fig, ax = plt.subplots(figsize=(9, max(4, top_k * 0.22)))
    sns.barplot(x=imp[order], y=np.array(feature_names)[order], orient="h", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return _save_fig(fig, filename)


def plot_correlation_heatmap(
    X: np.ndarray,
    feature_names: Sequence[str],
    max_features: int = 40,
    seed: int = 42,
) -> Path:
    """Correlation matrix for a subsample of features (readability)."""
    _set_pub_style()
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    if n > max_features:
        cols = sorted(rng.choice(n, size=max_features, replace=False))
    else:
        cols = list(range(n))
    sub = X[:, cols]
    names = [feature_names[i] for i in cols]
    cm = np.corrcoef(sub, rowvar=False)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm,
        xticklabels=names,
        yticklabels=names,
        cmap="vlag",
        center=0.0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Feature correlation heatmap (subset)")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    return _save_fig(fig, "10_correlation_heatmap.png")


def plot_boxplots_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    label_col: str = "dialect",
    max_features: int = 12,
) -> Path:
    """Boxplots for selected numeric columns across dialects."""
    _set_pub_style()
    feats = list(feature_cols)[:max_features]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9))
    axes = axes.ravel()
    for ax, col in zip(axes, feats):
        sns.boxplot(data=df, x=label_col, y=col, ax=ax, palette="pastel", showfliers=False)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
    for ax in axes[len(feats) :]:
        ax.axis("off")
    fig.suptitle("Dialect-conditioned feature distributions (selection)", y=1.02)
    return _save_fig(fig, "11_boxplots_dialect_features.png")


def run_visual_suite(
    waveforms: List[np.ndarray],
    dialect_ids_waveforms: List[int],
    label_names: List[str],
    sample_rate: int,
    X: np.ndarray,
    feature_names: List[str],
    y_all: np.ndarray,
    f1_list: Optional[List[float]] = None,
    f2_list: Optional[List[float]] = None,
    pca_coords: Optional[np.ndarray] = None,
    tsne_coords: Optional[np.ndarray] = None,
    rf_importance: Optional[np.ndarray] = None,
) -> List[Path]:
    """
    Generate the full visualization set required by the benchmark.

    Parameters
    ----------
    dialect_ids_waveforms : integer labels aligned with ``waveforms`` (subset allowed).
    y_all : integer labels for every row of ``X`` (same length as ``X``).
    """
    _set_pub_style()
    paths: List[Path] = []
    dialect_str_audio = [label_names[i] for i in dialect_ids_waveforms]
    dialect_str_all = [label_names[int(i)] for i in y_all]

    if waveforms and dialect_ids_waveforms and len(waveforms) == len(dialect_ids_waveforms):
        paths.append(plot_waveforms_by_dialect(waveforms, dialect_str_audio, sample_rate))
        paths.append(plot_spectrograms_grid(waveforms, dialect_str_audio, sample_rate))
        paths.append(plot_mfcc_heatmaps(waveforms, dialect_str_audio, sample_rate))
        paths.append(plot_spectral_centroid_tracks(waveforms, dialect_str_audio, sample_rate))
        paths.append(plot_pitch_contours(waveforms, dialect_str_audio, sample_rate))
    else:
        logger.warning(
            "Skipping waveform/spectrogram/MFCC track plots — missing waveforms or label alignment."
        )

    if (
        f1_list is not None
        and f2_list is not None
        and len(f1_list) == len(f2_list)
        and len(f1_list) == len(dialect_str_all)
    ):
        paths.append(plot_formant_space(f1_list, f2_list, dialect_str_all))

    if pca_coords is not None and len(pca_coords) == len(dialect_str_all):
        paths.append(
            plot_embedding_scatter_2d(
                pca_coords,
                dialect_str_all,
                "PCA (2D) of scaled handcrafted features",
                "07_pca_embedding.png",
            )
        )
    if tsne_coords is not None and len(tsne_coords) == len(dialect_str_all):
        paths.append(
            plot_embedding_scatter_2d(
                tsne_coords,
                dialect_str_all,
                "t-SNE (2D) of scaled handcrafted features",
                "08_tsne_embedding.png",
            )
        )

    if rf_importance is not None:
        paths.append(
            plot_feature_importance(
                rf_importance,
                feature_names,
                "Random Forest mean decrease impurity",
                "09_feature_importance_rf.png",
            )
        )

    df_feat = pd.DataFrame(X, columns=feature_names)
    df_feat["dialect"] = dialect_str_all
    top_cols = list(feature_names[:12])
    paths.append(plot_boxplots_features(df_feat, top_cols))

    paths.append(plot_correlation_heatmap(X, feature_names))

    return paths
