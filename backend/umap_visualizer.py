"""
UMAP visualization module.
Loads pre-fitted artifacts saved from the notebook — never trains at runtime.

Uses KNN-based 2D projection instead of calling umap.transform() directly,
which avoids numba JIT crashes on Python 3.13.

Required files in backend/models/:
    umap_scaler.pkl
    umap_kmeans.pkl
    umap_cluster_to_dialect.pkl
    label_encoder.pkl
    X_train_scaled.npy      (162-dim scaled training features for KNN lookup)
    X_train_2d.npy          (2D training embeddings for background scatter)
    y_train.npy             (integer labels for background scatter)
"""

import io
import os
import base64
import logging

import joblib
import librosa
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# Reuse the exact same feature extraction as the classifier
from dialect_classifier import extract_features

# ── colour palette (matches the rest of the project) ──────────────────────────
COLORS = {
    "bg":      "#1A1410",
    "surface": "#2A201A",
    "text":    "#F5E6D3",
    "border":  "#3E342C",
}

DIALECT_COLORS = {
    "Egyptian Arabic":  "#7C3AED",
    "Gulf Arabic":      "#C75B39",
    "Levantine Arabic": "#7A9E7E",
    "Maghrebi Arabic":  "#E8A838",
}

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

ARTIFACTS = {
    "scaler":             os.path.join(MODELS_DIR, "umap_scaler.pkl"),
    "kmeans":             os.path.join(MODELS_DIR, "umap_kmeans.pkl"),
    "cluster_to_dialect": os.path.join(MODELS_DIR, "umap_cluster_to_dialect.pkl"),
    "label_encoder":      os.path.join(MODELS_DIR, "label_encoder.pkl"),
    "X_train_scaled":     os.path.join(MODELS_DIR, "X_train_scaled.npy"),
    "X_train_2d":         os.path.join(MODELS_DIR, "X_train_2d.npy"),
    "y_train":            os.path.join(MODELS_DIR, "y_train.npy"),
}

# ── module-level singletons (loaded once at import time) ──────────────────────
_scaler              = None
_kmeans              = None
_cluster_to_dialect  = {}
_le                  = None
_X_train_scaled      = None
_X_train_2d          = None
_y_train             = None
_knn                 = None          # sklearn NearestNeighbors index
_READY               = False

K_NEIGHBORS = 10  # number of neighbors for 2D interpolation


def _load_artifacts() -> bool:
    global _scaler, _kmeans, _cluster_to_dialect
    global _le, _X_train_scaled, _X_train_2d, _y_train, _knn, _READY

    missing = [k for k, p in ARTIFACTS.items() if not os.path.exists(p)]
    if missing:
        logging.warning(
            "UMAP: missing artifact(s): %s. "
            "Run save_umap_artifacts.py in the notebook first.",
            missing,
        )
        return False

    try:
        _scaler             = joblib.load(ARTIFACTS["scaler"])
        _kmeans             = joblib.load(ARTIFACTS["kmeans"])
        _kmeans.cluster_centers_ = _kmeans.cluster_centers_.astype(np.float64)
        _cluster_to_dialect = joblib.load(ARTIFACTS["cluster_to_dialect"])
        _le                 = joblib.load(ARTIFACTS["label_encoder"])
        _X_train_scaled     = np.load(ARTIFACTS["X_train_scaled"])
        _X_train_2d         = np.load(ARTIFACTS["X_train_2d"])
        _y_train            = np.load(ARTIFACTS["y_train"])

        # Build a KNN index over scaled training features for fast lookup
        _knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean")
        _knn.fit(_X_train_scaled)

        logging.info(
            "UMAP artifacts loaded (KNN mode). "
            "Background scatter: %d points, %d dialects.",
            len(_X_train_2d),
            len(_le.classes_),
        )
        _READY = True
        return True

    except Exception as exc:
        logging.error("UMAP artifact load failed: %s", exc)
        return False


# Load once at import — zero training, just deserialization
_READY = _load_artifacts()


# ── public API ────────────────────────────────────────────────────────────────
def is_ready() -> bool:
    return _READY


def project_and_plot(file_path: str) -> dict:
    """
    Load audio from *file_path*, project it into the pre-fitted UMAP space
    using KNN interpolation (no numba required), and return:
        {
          "umap_chart":        "<base64 PNG>",
          "predicted_dialect": "Egyptian Arabic",
          "umap_coords":       [x, y],
          "nearest_cluster":   2,
        }
    """
    if not _READY:
        raise RuntimeError(
            "UMAP visualizer is not ready. "
            "Copy the model artifacts into backend/models/ first."
        )

    # Extract features → scale (same pipeline as classifier)
    audio, sr   = librosa.load(file_path, sr=16000)
    feat        = extract_features(audio, sr)
    feat_scaled = _scaler.transform(feat[None, :])       # (1, 162)

    # KNN projection: find K nearest training samples in 162-dim space,
    # then distance-weighted average of their pre-computed 2D positions.
    distances, indices = _knn.kneighbors(feat_scaled)     # (1, K)
    dists = distances[0]
    idxs  = indices[0]

    # Inverse-distance weighting (add epsilon to avoid division by zero)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum()
    point_2d = np.average(_X_train_2d[idxs], axis=0, weights=weights).astype(np.float64)

    # Assign to nearest KMeans cluster → dialect
    cluster      = int(_kmeans.predict(point_2d[None, :])[0])
    dialect_idx  = _cluster_to_dialect[cluster]
    dialect_name = _le.classes_[dialect_idx]

    chart_b64 = _build_scatter(point_2d, dialect_name)

    return {
        "umap_chart":        chart_b64,
        "predicted_dialect": dialect_name,
        "umap_coords":       [float(point_2d[0]), float(point_2d[1])],
        "nearest_cluster":   cluster,
    }


# ── plot ───────────────────────────────────────────────────────────────────────
def _build_scatter(new_point: np.ndarray, predicted_dialect: str) -> str:
    dialect_names = list(_le.classes_)
    palette = {
        name: list(DIALECT_COLORS.values())[i % len(DIALECT_COLORS)]
        for i, name in enumerate(dialect_names)
    }

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["surface"])

    # Background: training scatter (pre-computed 2D, no transform needed)
    for cls_idx, name in enumerate(dialect_names):
        mask  = _y_train == cls_idx
        color = palette.get(name, "#888888")
        ax.scatter(
            _X_train_2d[mask, 0], _X_train_2d[mask, 1],
            s=18, alpha=0.35, color=color,
            edgecolors="none", label=name,
        )

    # Foreground: new file as a large star
    px, py     = float(new_point[0]), float(new_point[1])
    star_color = palette.get(predicted_dialect, "#FFFFFF")
    ax.scatter(
        px, py,
        s=380, marker="*", color=star_color, zorder=10,
        edgecolors="white", linewidths=1.5,
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

    # Dynamic annotation offset based on data range
    x_range = _X_train_2d[:, 0].max() - _X_train_2d[:, 0].min()
    y_range = _X_train_2d[:, 1].max() - _X_train_2d[:, 1].min()
    off_x = x_range * 0.08
    off_y = y_range * 0.08

    # Annotation bubble
    ax.annotate(
        f" ★  {predicted_dialect} ",
        xy=(px, py),
        xytext=(px + off_x, py + off_y),
        fontsize=9,
        color="white",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor=star_color,
            edgecolor="white",
            alpha=0.9,
            linewidth=1.2,
        ),
        arrowprops=dict(arrowstyle="->", color="white", lw=1.2),
    )

    # Legend
    handles = [
        plt.scatter([], [], s=50, color=palette.get(n, "#888"), label=n, alpha=0.7)
        for n in dialect_names
    ]
    handles.append(
        plt.scatter([], [], s=200, marker="*", color="white",
                    edgecolors="white", label="Your file")
    )
    ax.legend(
        handles=handles,
        loc="upper left",
        facecolor=COLORS["surface"],
        edgecolor=COLORS["border"],
        labelcolor=COLORS["text"],
        fontsize=8,
        framealpha=0.85,
    )

    ax.set_title(
        "UMAP Dialect Space — Where Does Your File Land?",
        color=COLORS["text"], fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("UMAP-1", color=COLORS["text"], fontsize=10)
    ax.set_ylabel("UMAP-2", color=COLORS["text"], fontsize=10)
    ax.tick_params(colors=COLORS["text"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["border"])
    ax.grid(alpha=0.08, color=COLORS["text"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=COLORS["bg"], dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
