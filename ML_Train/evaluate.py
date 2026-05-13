"""
Evaluation metrics, confusion matrices, multiclass ROC (OvR), reports, explainability hooks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from utils import PLOTS_DIR, REPORTS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    class_names: Sequence[str],
) -> Dict[str, float]:
    """Accuracy, macro precision/recall/F1, multiclass ROC-AUC (OvR) if scores provided."""
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None and y_score.ndim == 2:
        try:
            y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
            auc = roc_auc_score(y_bin, y_score, average="macro", multi_class="ovr")
            out["roc_auc_ovr_macro"] = float(auc)
        except Exception as exc:
            logger.warning("ROC-AUC skipped: %s", exc)
    return out


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    title: str,
    filename: str,
) -> Path:
    """Save sklearn confusion matrix plot."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names,
        xticks_rotation=45,
        ax=ax,
        colorbar=True,
        cmap="Blues",
    )
    ax.set_title(title)
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix %s", path)
    return path


def plot_multiclass_roc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    class_names: Sequence[str],
    title: str,
    filename: str,
) -> Optional[Path]:
    """One-vs-rest ROC curves for multiclass."""
    if y_score is None or y_score.ndim != 2:
        return None
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        auc_i = roc_auc_score(y_bin[:, i], y_score[:, i])
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={auc_i:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC plot %s", path)
    return path


def evaluate_models_bundle(
    bundle: Dict,
    feature_names: Sequence[str],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """
    Evaluate all fitted GridSearchCV models from ``train.train_all_models`` result.

    Writes per-model reports under outputs/reports/.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]

    for name, gs in bundle["models"].items():
        est = gs.best_estimator_
        y_pred = est.predict(X_test)
        y_score = None
        if hasattr(est, "predict_proba"):
            try:
                y_score = est.predict_proba(X_test)
            except Exception:
                y_score = None

        metrics = compute_metrics(y_test, y_pred, y_score, class_names)

        report_txt = classification_report(
            y_test,
            y_pred,
            target_names=list(class_names),
            digits=4,
            zero_division=0,
        )
        with open(REPORTS_DIR / f"classification_report_{name}.txt", "w", encoding="utf-8") as f:
            f.write(report_txt)

        plot_confusion_matrix(
            y_test,
            y_pred,
            class_names,
            title=f"Confusion matrix — {name}",
            filename=f"confusion_matrix_{name}.png",
        )
        if y_score is not None:
            plot_multiclass_roc(
                y_test,
                y_score,
                class_names,
                title=f"ROC (OvR) — {name}",
                filename=f"roc_ovr_{name}.png",
            )

        row = {"model": name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    return df


def sklearn_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    model_key: str,
    n_repeats: int = 15,
    random_state: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Model-agnostic permutation importance on held-out data."""
    logger.info("Permutation importance for %s ...", model_key)
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring="f1_macro",
    )
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    df.to_csv(REPORTS_DIR / f"permutation_importance_{model_key}.csv", index=False)
    return df


def random_forest_native_importance(
    rf_pipeline,
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Mean decrease impurity from the RF step (undo PCA if present — approximate)."""
    try:
        clf = rf_pipeline.named_steps["clf"]
        imp = clf.feature_importances_
        # If PCA present, cannot map cleanly to raw names — report PCA components
        if "pca" in rf_pipeline.named_steps:
            n_comp = imp.shape[0]
            names = [f"PCA_component_{i}" for i in range(n_comp)]
        else:
            names = list(feature_names)
        df = pd.DataFrame({"feature": names, "importance": imp}).sort_values(
            "importance", ascending=False
        )
        df.to_csv(REPORTS_DIR / "random_forest_native_importance.csv", index=False)
        if "pca" not in rf_pipeline.named_steps:
            return imp
    except Exception as exc:
        logger.warning("RF native importance skipped: %s", exc)
    return None


def shap_tree_summary(
    rf_pipeline,
    X_train_small: np.ndarray,
    feature_names: Sequence[str],
    max_samples: int = 400,
) -> None:
    """SHAP TreeExplainer for Random Forest (no deep learning)."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP plots.")
        return

    if "pca" in rf_pipeline.named_steps:
        logger.warning("SHAP skipped when PCA is in pipeline (feature mapping ambiguous).")
        return

    scaler = rf_pipeline.named_steps["scaler"]
    clf = rf_pipeline.named_steps["clf"]
    idx = np.random.choice(X_train_small.shape[0], size=min(max_samples, X_train_small.shape[0]), replace=False)
    Xs = scaler.transform(X_train_small[idx])

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xs)

    shap_dir = REPORTS_DIR / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)

    # Multi-class list — plot mean |SHAP| per class if list
    if isinstance(sv, list):
        for c, arr in enumerate(sv):
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                arr,
                Xs,
                feature_names=list(feature_names),
                show=False,
                max_display=25,
            )
            plt.tight_layout()
            plt.savefig(shap_dir / f"shap_summary_class_{c}.png", dpi=300, bbox_inches="tight")
            plt.close()
    else:
        plt.figure(figsize=(10, 8))
        shap.summary_plot(sv, Xs, feature_names=list(feature_names), show=False, max_display=25)
        plt.tight_layout()
        plt.savefig(shap_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info("Saved SHAP plots under %s", shap_dir)


def extract_rf_importances_for_plots(
    rf_pipeline,
    feature_names: Sequence[str],
) -> Optional[np.ndarray]:
    """Return importance vector aligned with ``feature_names`` when no PCA."""
    if "pca" in rf_pipeline.named_steps:
        return None
    clf = rf_pipeline.named_steps["clf"]
    return np.asarray(clf.feature_importances_, dtype=np.float64)
