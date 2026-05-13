"""
Classical ML training with stratified splits, scaling, optional PCA, GridSearchCV.
Models: SVM, Random Forest, KNN, Logistic Regression, Gaussian Naive Bayes.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import MODELS_DIR, RANDOM_SEED, REPORTS_DIR

logger = logging.getLogger(__name__)


def _make_pipeline(
    estimator: Any,
    use_pca: bool,
    n_components: Optional[float],
    random_state: int,
) -> Pipeline:
    steps: List[Tuple[str, Any]] = [
        ("scaler", StandardScaler()),
    ]
    if use_pca:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=n_components if n_components is not None else 0.95,
                    random_state=random_state,
                ),
            )
        )
    steps.append(("clf", estimator))
    return Pipeline(steps)


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
    use_pca: bool = False,
    pca_components: Optional[float] = 0.95,
    cv_folds: int = 5,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Stratified train/test split; grid-search each model; save best estimators + metadata.

    Returns
    -------
    bundle with keys: X_train, X_test, y_train, y_test, models (dict name -> fitted GridSearchCV)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    _, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min())
    effective_cv = min(cv_folds, min_class)
    if effective_cv < 2:
        raise ValueError(
            f"Stratified CV requires ≥2 samples per class; smallest class has {min_class}. "
            "Increase --max_samples or use the full dataset."
        )
    if effective_cv < cv_folds:
        logger.warning(
            "Smallest class count is %d — reducing CV folds from %d to %d.",
            min_class,
            cv_folds,
            effective_cv,
        )
        cv_folds = effective_cv

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    models: Dict[str, GridSearchCV] = {}

    # --- SVM ---
    svm_pipe = _make_pipeline(
        SVC(kernel="rbf", probability=True, random_state=random_state),
        use_pca,
        pca_components,
        random_state,
    )
    svm_grid = {
        "clf__C": [0.1, 1.0, 10.0, 100.0],
        "clf__gamma": ["scale", 1e-2, 1e-3],
    }
    if use_pca:
        svm_grid["pca__n_components"] = [pca_components] if pca_components is not None else [0.9, 0.95]

    gs_svm = GridSearchCV(
        svm_pipe,
        svm_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Grid-search SVM ...")
    gs_svm.fit(X_train, y_train)
    models["svm"] = gs_svm
    joblib.dump(gs_svm.best_estimator_, MODELS_DIR / "svm_best.joblib")

    # --- Random Forest ---
    rf_pipe = _make_pipeline(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        use_pca,
        pca_components,
        random_state,
    )
    rf_grid: Dict[str, List[Any]] = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 16, 32],
        "clf__min_samples_leaf": [1, 2],
    }
    if use_pca:
        rf_grid["pca__n_components"] = [pca_components] if pca_components is not None else [0.95]

    gs_rf = GridSearchCV(
        rf_pipe,
        rf_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Grid-search Random Forest ...")
    gs_rf.fit(X_train, y_train)
    models["random_forest"] = gs_rf
    joblib.dump(gs_rf.best_estimator_, MODELS_DIR / "random_forest_best.joblib")

    # --- KNN ---
    knn_pipe = _make_pipeline(
        KNeighborsClassifier(),
        use_pca,
        pca_components,
        random_state,
    )
    knn_grid = {
        "clf__n_neighbors": [3, 5, 9, 15],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    }
    if use_pca:
        knn_grid["pca__n_components"] = [pca_components] if pca_components is not None else [0.95]

    gs_knn = GridSearchCV(
        knn_pipe,
        knn_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Grid-search KNN ...")
    gs_knn.fit(X_train, y_train)
    models["knn"] = gs_knn
    joblib.dump(gs_knn.best_estimator_, MODELS_DIR / "knn_best.joblib")

    # --- Logistic Regression ---
    lr_pipe = _make_pipeline(
        LogisticRegression(max_iter=8000, random_state=random_state),
        use_pca,
        pca_components,
        random_state,
    )
    lr_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"],
    }
    if use_pca:
        lr_grid["pca__n_components"] = [pca_components] if pca_components is not None else [0.95]

    gs_lr = GridSearchCV(
        lr_pipe,
        lr_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Grid-search Logistic Regression ...")
    gs_lr.fit(X_train, y_train)
    models["logistic_regression"] = gs_lr
    joblib.dump(gs_lr.best_estimator_, MODELS_DIR / "logistic_regression_best.joblib")

    # --- Gaussian Naive Bayes (often better without PCA) ---
    gnb_pipe = _make_pipeline(
        GaussianNB(),
        use_pca,
        pca_components,
        random_state,
    )
    gnb_grid = {"clf__var_smoothing": np.logspace(-12, -6, num=7)}
    if use_pca:
        gnb_grid["pca__n_components"] = [pca_components] if pca_components is not None else [0.95]

    gs_gnb = GridSearchCV(
        gnb_pipe,
        gnb_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    logger.info("Grid-search Gaussian Naive Bayes ...")
    gs_gnb.fit(X_train, y_train)
    models["gaussian_nb"] = gs_gnb
    joblib.dump(gs_gnb.best_estimator_, MODELS_DIR / "gaussian_nb_best.joblib")

    meta = {
        "class_names": class_names,
        "random_state": random_state,
        "test_size": test_size,
        "use_pca": use_pca,
        "pca_components": pca_components,
        "cv_folds": cv_folds,
        "best_params": {k: v.best_params_ for k, v in models.items()},
    }
    with open(REPORTS_DIR / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "models": models,
        "meta": meta,
    }
