"""
Utility helpers: paths, logging, reproducibility, dataset loading, and label handling.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# -----------------------------------------------------------------------------
# Paths (project root = directory containing this file)
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"

TARGET_SAMPLE_RATE = 16_000
RANDOM_SEED = 42

# Five dialect classes requested for this benchmark (canonical English names).
CANONICAL_LABELS: Tuple[str, ...] = (
    "Egyptian",
    "Gulf",
    "Levantine",
    "Iraqi",
    "Maghrebi",
)

# Hugging Face MADIS-5 often lists MSA instead of Iraqi; aliases handle common variants.
_LABEL_SYNONYMS: Dict[str, str] = {
    # Egyptian
    "egyptian": "Egyptian",
    "egy": "Egyptian",
    "eg": "Egyptian",
    "masri": "Egyptian",
    # Gulf
    "gulf": "Gulf",
    "gcc": "Gulf",
    "khaleeji": "Gulf",
    "khaliji": "Gulf",
    # Levantine
    "levantine": "Levantine",
    "levant": "Levantine",
    "shami": "Levantine",
    "levantine arabic": "Levantine",
    # Iraqi
    "iraqi": "Iraqi",
    "iraq": "Iraqi",
    # Maghrebi
    "maghrebi": "Maghrebi",
    "maghreb": "Maghrebi",
    "north african": "Maghrebi",
    "darija": "Maghrebi",
    # MSA (mapped optionally to Iraqi for rubric alignment — see README)
    "msa": "MSA",
    "modern standard arabic": "MSA",
    "standard arabic": "MSA",
    "fuṣḥā": "MSA",
    "fus-ha": "MSA",
    "fusha": "MSA",
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger once for console + optional file."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    # Hugging Face / httpx log every HEAD/GET at INFO — very noisy and 404s look like failures (they are not).
    if level > logging.DEBUG:
        for name in ("httpx", "httpcore", "urllib3", "filelock"):
            logging.getLogger(name).setLevel(logging.WARNING)
        # Hub nudges for HF_TOKEN are emitted at WARNING; hide unless debugging Hub traffic.
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils").setLevel(logging.ERROR)


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix seeds for reproducibility (best-effort across libs)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import librosa

        fix = getattr(librosa.util, "fix_seed", None)
        if callable(fix):
            fix(seed)
    except Exception:
        pass


def ensure_output_dirs() -> None:
    """Create expected output folders."""
    for d in (DATA_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def normalize_label_text(raw: Any) -> str:
    """Lowercase, strip, collapse spaces for lookup."""
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def resolve_canonical_label(raw: Any, map_msa_to_iraqi: bool = False) -> Optional[str]:
    """
    Map a dataset label string to one of CANONICAL_LABELS or None if unknown.

    If ``map_msa_to_iraqi`` is True, MSA is renamed to Iraqi (for coursework alignment;
    linguistically MSA is not Iraqi dialect — document in README).
    """
    key = normalize_label_text(raw)
    if not key:
        return None
    if map_msa_to_iraqi and key in ("msa", "modern standard arabic", "standard arabic"):
        return "Iraqi"
    mapped = _LABEL_SYNONYMS.get(key)
    if mapped == "MSA" and map_msa_to_iraqi:
        return "Iraqi"
    if mapped and mapped != "MSA":
        return mapped
    # Direct match to canonical
    for c in CANONICAL_LABELS:
        if key == c.lower():
            return c
    # Partial contains for longer HF strings
    for c in CANONICAL_LABELS:
        if c.lower() in key or key in c.lower():
            return c
    if "egypt" in key:
        return "Egyptian"
    if "levant" in key or "sham" in key:
        return "Levantine"
    if "gulf" in key or "peninsula" in key or "najdi" in key or "khalij" in key:
        return "Gulf"
    if "iraq" in key:
        return "Iraqi"
    if "maghreb" in key or "morocc" in key or "alger" in key or "tunis" in key:
        return "Maghrebi"
    if "modern standard" in key or key == "msa":
        return "Iraqi" if map_msa_to_iraqi else None  # None = drop if not remapping
    return None


def _guess_audio_and_label_columns(
    sample: Dict[str, Any],
    features_obj: Any,
) -> Tuple[str, str]:
    """Infer column names for audio and label from one dataset row."""
    keys = list(sample.keys())
    audio_key = None
    for k in ("audio", "Audio", "speech", "file"):
        if k in keys:
            audio_key = k
            break
    if audio_key is None:
        # HF Audio column often decoded as dict under 'audio'
        for k in keys:
            v = sample[k]
            if isinstance(v, dict) and "array" in v and "sampling_rate" in v:
                audio_key = k
                break
    if audio_key is None:
        raise ValueError(f"Could not find audio column in keys: {keys}")

    label_key = None
    for k in ("dialect", "label", "labels", "class", "target", "lang", "language"):
        if k in keys and k != audio_key:
            label_key = k
            break
    if label_key is None:
        # ClassLabel feature names
        if hasattr(features_obj, "features"):
            # Dataset.features is a Features dict
            for k in features_obj:
                if k != audio_key and k != "audio":
                    label_key = k
                    break
    if label_key is None:
        raise ValueError(f"Could not find label column in keys: {keys}")
    return audio_key, label_key


def _start_download_heartbeat(message: str, interval_sec: float = 45.0) -> Tuple[threading.Event, threading.Thread]:
    """
    Emit periodic WARNING logs while ``load_dataset`` blocks (no built-in progress for huge prepares).
    Call ``stop.set()`` when loading finishes.
    """
    stop = threading.Event()

    def _run() -> None:
        n = 0
        while not stop.wait(interval_sec):
            n += 1
            logging.warning(
                "%s (heartbeat #%d, ~%d min). Note: Huge datasets (>1GB) take time to download shards even if capping rows. "
                "If this hangs indefinitely, try deleting stale *.lock files in your Hugging Face cache directory.",
                message,
                n,
                int(n * interval_sec // 60),
            )

    t = threading.Thread(target=_run, name="hf-download-heartbeat", daemon=True)
    t.start()
    return stop, t


def load_madis_via_api(
    max_samples: int = 600,
    split: str = "test",
    map_msa_to_iraqi: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[int], List[str], Dict[str, int]]:
    """
    Fetch dataset rows directly via HF Datasets Server API. 
    Much faster for small samples (avoids 1.3GB shard download).
    """
    import io
    import requests
    import soundfile as sf
    from tqdm.auto import tqdm

    base_url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": "badrex/MADIS5-spoken-arabic-dialects",
        "config": "default",
        "split": split,
    }

    all_rows: List[Dict[str, Any]] = []
    page_size = 100
    n_pages = (max_samples + page_size - 1) // page_size

    logging.info("Fetching %d rows via HF Datasets API (split=%r) ...", max_samples, split)
    
    for p in range(n_pages):
        offset = p * page_size
        length = min(page_size, max_samples - offset)
        if length <= 0:
            break
            
        p_params = {**params, "offset": offset, "length": length}
        try:
            resp = requests.get(base_url, params=p_params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            rows = [r["row"] for r in data.get("rows", [])]
            all_rows.extend(rows)
            if len(rows) < length:
                break # End of dataset
        except Exception as e:
            logging.error("API request failed at offset %d: %s", offset, e)
            break

    if not all_rows:
        raise RuntimeError("API returned no rows.")

    waveforms: List[np.ndarray] = []
    sampling_rates: List[int] = []
    labels: List[int] = []
    
    # Pre-resolve labels to detect used classes
    used_classes: set = set()
    row_data = [] # (audio_url, canonical_label)
    
    for row in all_rows:
        raw_dialect = row.get("dialect")
        c = resolve_canonical_label(raw_dialect, map_msa_to_iraqi)
        if c is None:
            continue
        
        audio_list = row.get("audio", [])
        if not audio_list or not isinstance(audio_list, list):
            continue
        
        url = audio_list[0].get("src")
        if not url:
            continue
            
        used_classes.add(c)
        row_data.append((url, c))

    # Consistency with original method: preferred order then extras
    ordered = [c for c in CANONICAL_LABELS if c in used_classes]
    for c in sorted(used_classes):
        if c not in ordered:
            ordered.append(c)
    
    label_to_idx = {name: i for i, name in enumerate(ordered)}
    
    logging.info("Downloading %d audio files ...", len(row_data))
    for url, c in tqdm(row_data, desc="Downloading audio"):
        try:
            # Download audio file
            r_audio = requests.get(url, timeout=20)
            r_audio.raise_for_status()
            
            # Read into numpy
            with io.BytesIO(r_audio.content) as f:
                data, sr = sf.read(f)
                # Ensure float32 mono
                if data.ndim > 1:
                    data = data.mean(axis=1)
                data = data.astype(np.float32)
                
            waveforms.append(data)
            sampling_rates.append(int(sr))
            labels.append(label_to_idx[c])
        except Exception as e:
            logging.debug("Failed to download audio from %s: %s", url, e)

    logging.info("Successfully loaded %d utterances via API.", len(waveforms))
    return waveforms, sampling_rates, labels, ordered, label_to_idx


def load_madis_dataset(
    split: str = "test",
    max_samples: Optional[int] = None,
    map_msa_to_iraqi: bool = False,
) -> Tuple[List[np.ndarray], List[int], List[int], List[str], Dict[str, int]]:
    """
    Load ``badrex/MADIS5-spoken-arabic-dialects`` via Hugging Face ``datasets``.

    Returns
    -------
    waveforms : list of 1D float32 mono arrays (variable length)
    sampling_rates : original sampling rate per waveform (Hz)
    label_ids : list of int in [0, n_classes)
    label_names : list of class names aligned with ids
    label_to_idx : mapping canonical name -> index
    """
    from datasets import Audio, Dataset, load_dataset

    logging.info("Loading dataset badrex/MADIS5-spoken-arabic-dialects ...")
    ds = None
    last_err: Optional[BaseException] = None

    use_slice = max_samples is not None and max_samples > 0

    # Optimization: Use the Datasets Server API to avoid downloading massive 1.3GB shards.
    # We now allow this up to 5000 rows (the full dataset) to bypass local cache lock issues.
    target_count = max_samples if max_samples is not None else 5000
    if target_count <= 5000:
        try:
            return load_madis_via_api(
                max_samples=target_count,
                split=split,
                map_msa_to_iraqi=map_msa_to_iraqi
            )
        except Exception as e:
            if max_samples is not None:
                logging.warning("API-based load failed, falling back to standard datasets load: %s", e)
            else:
                logging.debug("API load failed for full dataset; will try standard shards.")

    hb_stop: Optional[threading.Event] = None
    if use_slice:
        logging.info(
            "Using Hugging Face split slice [:%d] — only the first %d rows are materialized (avoids downloading/preparing the entire corpus).",
            max_samples,
            max_samples,
        )
    else:
        logging.warning(
            "Loading the FULL dataset split (no row cap). First prepare/download is ~1.3 GB and can take "
            "30 minutes to several hours with little console output — this is normal for ``datasets``. "
            "For a quicker run omit --full_dataset (default caps rows) or pass e.g. --max_samples 800.",
        )

    hb_stop, _hb_thr = _start_download_heartbeat(
        "Still inside Hugging Face load_dataset (downloading or preparing audio shards on disk)",
        interval_sec=60.0,
    )
    try:
        # Reorder candidates to check requested first, then known-good 'test', then 'train'.
        _candidates = [split, "test", "train", "validation", "dev"]
        split_candidates = list(dict.fromkeys(s for s in _candidates if s))
        
        for sp in split_candidates:
            split_arg = f"{sp}[:{max_samples}]" if use_slice else sp
            try:
                logging.info("Attempting Hub load (split=%r) ...", split_arg)
                # trust_remote_code=True is required for some Hub datasets in newer datasets versions.
                # We try with it first, fallback if it's an unknown argument (older versions).
                load_kwargs = {
                    "path": "badrex/MADIS5-spoken-arabic-dialects",
                    "split": split_arg,
                }
                try:
                    ds = load_dataset(**load_kwargs, trust_remote_code=True)
                except TypeError:
                    ds = load_dataset(**load_kwargs)

                if sp != split:
                    logging.info(
                        "Requested split %r missing; successfully loaded %r instead (%d rows).",
                        split,
                        split_arg,
                        len(ds),
                    )
                break
            except Exception as exc:
                last_err = exc
                logging.debug("Split %r failed: %s", split_arg, exc)
                continue
    finally:
        if hb_stop is not None:
            hb_stop.set()

    if ds is None:
        raise RuntimeError(
            "Could not load MADIS-5 from Hugging Face Hub. Please check your internet connection "
            "or try again later. (Error: %s)" % last_err
        )

    logging.info(
        "Dataset ready (%d rows in memory). If you capped with --max_samples, audio bytes are still fetched for those rows only.",
        len(ds),
    )

    if max_samples is not None and max_samples > 0 and len(ds) > max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
        logging.info("Trimmed to first %d rows.", len(ds))

    # Decode audio to numpy
    sample0 = ds[0]
    feats = ds.features
    audio_key, label_key = _guess_audio_and_label_columns(sample0, feats)

    if isinstance(ds, Dataset):
        try:
            ds = ds.cast_column(audio_key, Audio(sampling_rate=None))
        except Exception:
            logging.debug("Could not cast audio column to Audio feature — using raw decode.")

    # Resolve int labels from ClassLabel if present
    id_to_name: Dict[int, str] = {}
    str_to_canon: Dict[str, Optional[str]] = {}

    label_feature = feats[label_key] if label_key in feats else None
    if hasattr(label_feature, "names") and label_feature.names:
        id_to_name = {i: str(n) for i, n in enumerate(label_feature.names)}
        for nm in label_feature.names:
            str_to_canon[str(nm)] = resolve_canonical_label(nm, map_msa_to_iraqi)
    else:
        # Collect unique raw labels
        raw_vals = set()
        for i in range(min(500, len(ds))):
            raw_vals.add(ds[i][label_key])
        for rv in raw_vals:
            str_to_canon[str(rv)] = resolve_canonical_label(rv, map_msa_to_iraqi)

    waveforms: List[np.ndarray] = []
    sampling_rates: List[int] = []
    labels: List[int] = []
    used_classes: set = set()

    # First pass: determine which canonical labels appear
    for i in range(len(ds)):
        raw = ds[i][label_key]
        if isinstance(raw, int) and id_to_name:
            raw_name = id_to_name.get(raw, str(raw))
        else:
            raw_name = str(raw)
        c = str_to_canon.get(raw_name)
        if c is None:
            c = resolve_canonical_label(raw_name, map_msa_to_iraqi)
        if c is None:
            continue
        used_classes.add(c)

    # Order classes: preferred order then any extras
    ordered = [c for c in CANONICAL_LABELS if c in used_classes]
    for c in sorted(used_classes):
        if c not in ordered:
            ordered.append(c)

    if len(ordered) < 2:
        logging.warning(
            "Fewer than 2 classes after filtering — check label mapping. Found: %s",
            ordered,
        )

    label_to_idx = {name: i for i, name in enumerate(ordered)}
    logging.info("Classes used (n=%d): %s", len(ordered), ordered)

    for i in range(len(ds)):
        row = ds[i]
        raw = row[label_key]
        if isinstance(raw, int) and id_to_name:
            raw_name = id_to_name.get(raw, str(raw))
        else:
            raw_name = str(raw)
        c = str_to_canon.get(raw_name) or resolve_canonical_label(raw_name, map_msa_to_iraqi)
        if c is None:
            continue
        if c not in label_to_idx:
            continue

        audio = row[audio_key]
        if isinstance(audio, dict):
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr_raw = int(audio["sampling_rate"])
        else:
            raise TypeError("Unexpected audio format — expected decoded dict from datasets.Audio")

        waveforms.append(arr)
        sampling_rates.append(sr_raw)
        labels.append(label_to_idx[c])

    logging.info("Loaded %d utterances.", len(waveforms))
    label_names = ordered
    return waveforms, sampling_rates, labels, label_names, label_to_idx


def save_json(path: Path, obj: Any) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)
