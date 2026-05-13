"""
End-to-end classical Arabic dialect identification pipeline.

Loads MADIS-5 from Hugging Face, preprocesses audio, extracts handcrafted features,
trains/evaluates classical ML models, generates plots and explainability outputs.

Run from this directory::

    python main.py --skip_tsne

Default load is capped (see ``--max_samples`` / ``--full_dataset``). Use ``--help`` for all options.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

_PROJECT_ROOT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

# huggingface_hub reads HF_HUB_VERBOSITY at import time (before datasets loads).
if "--verbose" not in sys.argv:
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from evaluate import (
    evaluate_models_bundle,
    extract_rf_importances_for_plots,
    random_forest_native_importance,
    shap_tree_summary,
    sklearn_permutation_importance,
)
from feature_extraction import FEATURE_ORDER, extract_all_features_vector
from preprocessing import preprocess_audio
from train import train_all_models
from utils import (
    DATA_DIR,
    PLOTS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    TARGET_SAMPLE_RATE,
    ensure_output_dirs,
    load_madis_dataset,
    set_seed,
    setup_logging,
)
from visualization import run_visual_suite


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Classical ML Arabic dialect identification")
    p.add_argument("--split", default="test", help="Hugging Face dataset split name")
    p.add_argument(
        "--max_samples",
        type=int,
        default=2500,
        help="Max utterances to load from the Hub (default 2500 for sane first-run time). Use --full_dataset for all rows.",
    )
    p.add_argument(
        "--full_dataset",
        action="store_true",
        help="Load the entire split (~4854 rows, ~1.3 GB). First run can take hours; console may look idle.",
    )
    p.add_argument(
        "--skip_formants",
        action="store_true",
        help="Skip Parselmouth formants (much faster; fills zeros for formant dims)",
    )
    p.add_argument(
        "--map_msa_to_iraqi",
        action="store_true",
        help="Relabel MSA as Iraqi for the 5-way Egyptian/Gulf/Levantine/Iraqi/Maghrebi setup",
    )
    p.add_argument("--use_pca", action="store_true", help="Use PCA inside model pipelines")
    p.add_argument(
        "--pca_components",
        type=float,
        default=0.95,
        help="PCA variance retained or int component count when --use_pca",
    )
    p.add_argument("--test_size", type=float, default=0.2, help="Held-out fraction")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument(
        "--cache_features",
        type=str,
        default=str(DATA_DIR / "features_cache.npz"),
        help="Path to save/load feature matrix (.npz)",
    )
    p.add_argument(
        "--reload_cache",
        action="store_true",
        help="Ignore existing cache and recompute features",
    )
    p.add_argument(
        "--tsne_max_samples",
        type=int,
        default=600,
        help="Max samples for t-SNE (cost grows quickly)",
    )
    p.add_argument(
        "--skip_tsne",
        action="store_true",
        help="Skip t-SNE visualization",
    )
    p.add_argument(
        "--viz_waveforms",
        type=int,
        default=400,
        help="Number of utterances to retain preprocessed waveforms for plotting",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    set_seed(args.seed)
    ensure_output_dirs()

    max_samples = None if args.full_dataset else args.max_samples
    if args.full_dataset:
        logging.info("Using --full_dataset: no row cap (long first-time download/prepare expected).")

    cache_path = Path(args.cache_features)

    if cache_path.exists() and not args.reload_cache:
        logging.info("Loading cached features from %s", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"])
        label_names = z["label_names"].tolist()
        waves_obj = z.get("waveforms_processed")
        if waves_obj is None:
            waves_list: List[np.ndarray] = []
        else:
            waves_list = [np.asarray(w, dtype=np.float32) for w in waves_obj.tolist()]
        logging.info("Cached matrix shape X=%s, waveforms for viz=%d", X.shape, len(waves_list))
    else:
        logging.info("Downloading / loading MADIS-5 …")
        waveforms, sampling_rates, labels, label_names, label_to_idx = load_madis_dataset(
            split=args.split,
            max_samples=max_samples,
            map_msa_to_iraqi=args.map_msa_to_iraqi,
        )

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        waves_list: List[np.ndarray] = []

        for i in tqdm(range(len(waveforms)), desc="Feature extraction"):
            try:
                y_raw = waveforms[i]
                sr_in = sampling_rates[i]
                y_d, sr_out = preprocess_audio(
                    y_raw,
                    sr_in,
                    target_sr=TARGET_SAMPLE_RATE,
                    normalize=True,
                    trim=True,
                    apply_pre_emphasis=False,
                )
                vec, names = extract_all_features_vector(
                    y_d,
                    sr_out,
                    include_formants=not args.skip_formants,
                )
                if i == 0:
                    assert names == FEATURE_ORDER
                X_list.append(vec)
                y_list.append(labels[i])
                if len(waves_list) < args.viz_waveforms:
                    waves_list.append(y_d.astype(np.float32))
            except Exception as exc:
                logging.warning("Skipping sample %d: %s", i, exc)

        if not X_list:
            raise RuntimeError("Feature extraction produced no rows.")

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.asarray(y_list, dtype=np.int64)

        np.savez_compressed(
            cache_path,
            X=X,
            y=y,
            feature_names=np.array(FEATURE_ORDER, dtype=object),
            label_names=np.array(label_names, dtype=object),
            waveforms_processed=np.array(waves_list, dtype=object),
            label_to_idx=np.array(list(label_to_idx.items()), dtype=object),
        )
        logging.info("Saved feature cache to %s", cache_path)

    # Visualization subset: waveforms (shorter list) vs full feature matrix
    n_w = len(waves_list) if waves_list else 0
    if n_w == 0:
        logging.warning(
            "No waveforms stored for plotting (only features). "
            "Waveform/spectrogram plots need recomputation — skipping audio plots."
        )
        dialect_ids_viz: List[int] = []
        waves_out: List[np.ndarray] = []
    else:
        waves_out = waves_list
        dialect_ids_viz = [int(yi) for yi in y[:n_w]]

    scaler_vis = StandardScaler()
    Xs = scaler_vis.fit_transform(X)

    pca_model = PCA(n_components=2, random_state=args.seed)
    pca_coords = pca_model.fit_transform(Xs)

    tsne_coords = None
    if not args.skip_tsne:
        n = min(args.tsne_max_samples, Xs.shape[0])
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(Xs.shape[0], size=n, replace=False)
        perplexity = max(5, min(30, n - 1))
        tsne = TSNE(
            n_components=2,
            random_state=args.seed,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        )
        tsne_coords_full = np.zeros((Xs.shape[0], 2), dtype=np.float32)
        tsne_coords_full[idx] = tsne.fit_transform(Xs[idx]).astype(np.float32)
        tsne_coords = tsne_coords_full

    f1_idx = FEATURE_ORDER.index("f1_mean")
    f2_idx = FEATURE_ORDER.index("f2_mean")
    f1_list = X[:, f1_idx].tolist()
    f2_list = X[:, f2_idx].tolist()

    logging.info("Training classical models …")
    bundle = train_all_models(
        X,
        y,
        class_names=label_names,
        test_size=args.test_size,
        random_state=args.seed,
        use_pca=args.use_pca,
        pca_components=args.pca_components if args.use_pca else None,
    )

    logging.info("Evaluating …")
    metrics_df = evaluate_models_bundle(bundle, FEATURE_ORDER, label_names)
    logging.info("Wrote metrics to %s", REPORTS_DIR / "metrics_summary.csv")
    print(metrics_df.to_string(index=False))

    rf_est = bundle["models"]["random_forest"].best_estimator_
    rf_imp = extract_rf_importances_for_plots(rf_est, FEATURE_ORDER)
    random_forest_native_importance(rf_est, FEATURE_ORDER)

    X_train = bundle["X_train"]
    X_test = bundle["X_test"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]

    sklearn_permutation_importance(
        rf_est,
        X_test,
        y_test,
        FEATURE_ORDER,
        model_key="random_forest",
    )
    sklearn_permutation_importance(
        bundle["models"]["svm"].best_estimator_,
        X_test,
        y_test,
        FEATURE_ORDER,
        model_key="svm",
    )

    shap_tree_summary(rf_est, X_train, FEATURE_ORDER)

    logging.info("Generating publication plots …")
    run_visual_suite(
        waveforms=waves_out,
        dialect_ids_waveforms=dialect_ids_viz,
        label_names=label_names,
        sample_rate=TARGET_SAMPLE_RATE,
        X=X,
        feature_names=FEATURE_ORDER,
        y_all=y,
        f1_list=f1_list,
        f2_list=f2_list,
        pca_coords=pca_coords,
        tsne_coords=tsne_coords,
        rf_importance=rf_imp,
    )

    logging.info("Done. Plots → %s | Models → outputs/models | Reports → %s", PLOTS_DIR, REPORTS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
