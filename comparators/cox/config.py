from __future__ import annotations


# Penalised Cox — Configuration, environment checks, and reproducibility helpers
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 1 ---

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Point this only at the exact current primary-model run being compared.
# No old comparator directory or precomputed Cox prediction cube is required.
# Relative paths resolve from the Jupyter kernel's working directory. Launch
# from the current project root or set EHR_MAIN_RUN_DIR to an absolute path.
# -----------------------------------------------------------------------------
WORKFLOW = "train_and_evaluate"  # or "evaluation_only"
# Folder where the CRESCENT notebook and its outputs live.
CRESCENT_BASE_DIR = os.getenv(
    "EHR_CRESCENT_BASE_DIR",
    ".",
).strip()
# The CRESCENT run folder (its EHR_CRESCENT_OUTPUT). Validated below: it must contain
# prepared_data_source_audit.csv and prepared_splits_manifest.json, which only the new
# CRESCENT notebook writes; old runs are rejected. Override with EHR_MAIN_RUN_DIR.
MAIN_RUN_DIR = os.getenv(
    "EHR_MAIN_RUN_DIR",
    str(Path(CRESCENT_BASE_DIR) / "runs" / "CRESCENT_Final"),
).strip()
OUTPUT_ROOT = os.getenv(
    "EHR_COX_OUTPUT",
    str(Path(CRESCENT_BASE_DIR) / "runs" / "cox"),
).strip()

# Required only for evaluation_only when the bundle is not already in OUTPUT_ROOT.
PRETRAINED_BUNDLE = ""  # e.g. "/path/to/frozen_coxph_full_history_bundle.pkl"

# Optional machine-readable primary-model metric contract. When blank, common
# filenames are searched inside MAIN_RUN_DIR.
MAIN_METRIC_CONTRACT_PATH = os.getenv("EHR_MAIN_METRIC_CONTRACT", "").strip()
REQUIRE_MAIN_METRIC_CONTRACT = False

# Like-for-like estimand and metric axes.
GAP_MONTHS = 6
REPORT_HORIZONS_MONTHS = (12, 24, 60, 120)
INTEGRATION_START_MONTHS = 12
CINDEX_HORIZON_MONTHS = 60
CI_POINT_HORIZONS_MONTHS = (24, 60)

# Full-history feature/model settings. Ridge selection is validation-only.
INCLUDE_EXTENDED_HISTORY_FEATURES = True
MIN_EVENT_FEATURE_NONZERO = 25
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0)
DEFAULT_RIDGE_LAMBDA = 0.1
COX_TRAINING_ROUNDS = 100
COX_LEARNING_RATE = 0.5
MIN_TRAINING_EVENTS_TO_FIT = 10
MIN_VALIDATION_EVENTS_TO_TUNE = 20
COX_NTHREAD = int(os.getenv("EHR_COX_NTHREAD", str(os.cpu_count() or 1)))
PREDICTION_BATCH_SIZE = 4096

# Participant-level uncertainty. Use 20 only for a smoke test.
CI_BOOTSTRAP_N = 200
CI_BOOTSTRAP_SEED = 101
CI_N_JOBS = min(8, max(1, int(os.cpu_count() or 1)))

if WORKFLOW not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError("WORKFLOW must be train_and_evaluate or evaluation_only.")
if not MAIN_RUN_DIR:
    raise ValueError("Set MAIN_RUN_DIR to the exact canonical primary-model run.")
if Path(MAIN_RUN_DIR).expanduser().resolve() == Path(OUTPUT_ROOT).expanduser().resolve():
    raise ValueError("OUTPUT_ROOT must differ from the CRESCENT run folder (MAIN_RUN_DIR).")
if not OUTPUT_ROOT:
    raise ValueError("Set OUTPUT_ROOT to a dedicated comparator output directory.")
if tuple(sorted(set(REPORT_HORIZONS_MONTHS))) != tuple(REPORT_HORIZONS_MONTHS):
    raise ValueError("REPORT_HORIZONS_MONTHS must be unique and increasing.")
if CINDEX_HORIZON_MONTHS not in REPORT_HORIZONS_MONTHS:
    raise ValueError("CINDEX_HORIZON_MONTHS must be one of the report horizons.")
if tuple(CI_POINT_HORIZONS_MONTHS) != (24, 60):
    raise ValueError("Canonical pointwise bootstrap CIs must use 24 and 60 months.")
if tuple(sorted(set(RIDGE_LAMBDAS))) != tuple(RIDGE_LAMBDAS):
    raise ValueError("RIDGE_LAMBDAS must be unique and increasing.")
if DEFAULT_RIDGE_LAMBDA not in RIDGE_LAMBDAS:
    raise ValueError("DEFAULT_RIDGE_LAMBDA must be included in RIDGE_LAMBDAS.")
if min(RIDGE_LAMBDAS) < 0:
    raise ValueError("Ridge penalties must be non-negative.")
if MIN_EVENT_FEATURE_NONZERO < 1 or COX_TRAINING_ROUNDS < 1:
    raise ValueError("Feature-frequency and training-round settings must be positive.")
if not 0.0 < COX_LEARNING_RATE <= 1.0:
    raise ValueError("COX_LEARNING_RATE must lie in (0, 1].")
if COX_NTHREAD < 1:
    raise ValueError("COX_NTHREAD must be at least 1.")

os.environ["EHR_COX_WORKFLOW"] = WORKFLOW
os.environ["EHR_MAIN_RUN_DIR"] = str(Path(MAIN_RUN_DIR).expanduser())
os.environ["EHR_COX_OUTPUT"] = str(Path(OUTPUT_ROOT).expanduser())
os.environ["EHR_GAP_MONTHS"] = str(GAP_MONTHS)
os.environ["EHR_REPORT_HORIZONS"] = ",".join(map(str, REPORT_HORIZONS_MONTHS))
os.environ["EHR_INTEGRATION_START"] = str(INTEGRATION_START_MONTHS)
os.environ["EHR_COX_EXTENDED_HISTORY_FEATURES"] = (
    "YES" if INCLUDE_EXTENDED_HISTORY_FEATURES else "NO"
)
os.environ["EHR_COX_MIN_FEATURE_NONZERO"] = str(MIN_EVENT_FEATURE_NONZERO)
os.environ["EHR_COX_RIDGE_LAMBDAS"] = ",".join(map(str, RIDGE_LAMBDAS))
os.environ["EHR_COX_DEFAULT_RIDGE_LAMBDA"] = str(DEFAULT_RIDGE_LAMBDA)
os.environ["EHR_COX_TRAINING_ROUNDS"] = str(COX_TRAINING_ROUNDS)
os.environ["EHR_COX_LEARNING_RATE"] = str(COX_LEARNING_RATE)
os.environ["EHR_COX_MIN_TRAIN_EVENTS"] = str(MIN_TRAINING_EVENTS_TO_FIT)
os.environ["EHR_COX_MIN_VAL_EVENTS"] = str(MIN_VALIDATION_EVENTS_TO_TUNE)
os.environ["EHR_COX_NTHREAD"] = str(COX_NTHREAD)
os.environ["EHR_COX_PREDICTION_BATCH_SIZE"] = str(PREDICTION_BATCH_SIZE)
os.environ["EHR_COX_CI_BOOTSTRAP_N"] = str(CI_BOOTSTRAP_N)
os.environ["EHR_COX_CI_BOOTSTRAP_SEED"] = str(CI_BOOTSTRAP_SEED)
os.environ["EHR_COX_CI_N_JOBS"] = str(CI_N_JOBS)
os.environ["EHR_COX_CI_CINDEX_TAU_MONTHS"] = str(CINDEX_HORIZON_MONTHS)
os.environ["EHR_COX_CI_POINT_HORIZONS"] = ",".join(
    map(str, CI_POINT_HORIZONS_MONTHS)
)
os.environ["EHR_REQUIRE_MAIN_METRIC_CONTRACT"] = (
    "YES" if REQUIRE_MAIN_METRIC_CONTRACT else "NO"
)
if MAIN_METRIC_CONTRACT_PATH:
    os.environ["EHR_MAIN_METRIC_CONTRACT"] = MAIN_METRIC_CONTRACT_PATH
else:
    os.environ.pop("EHR_MAIN_METRIC_CONTRACT", None)
if PRETRAINED_BUNDLE:
    os.environ["EHR_COX_BUNDLE"] = PRETRAINED_BUNDLE
else:
    os.environ.pop("EHR_COX_BUNDLE", None)

print({
    "workflow": WORKFLOW,
    "main_run_dir": str(Path(MAIN_RUN_DIR).expanduser().resolve()),
    "output_root": str(Path(OUTPUT_ROOT).expanduser().resolve()),
    "history_contract": "all canonical pre-index events",
    "gap_months": GAP_MONTHS,
    "report_horizons_months": REPORT_HORIZONS_MONTHS,
    "integration_start_months": INTEGRATION_START_MONTHS,
    "cindex_horizon_months": CINDEX_HORIZON_MONTHS,
    "ci_point_horizons_months": CI_POINT_HORIZONS_MONTHS,
    "ridge_lambdas": RIDGE_LAMBDAS,
    "training_rounds": COX_TRAINING_ROUNDS,
    "cox_threads_single_node": COX_NTHREAD,
    "bootstrap_replicates": CI_BOOTSTRAP_N,
    "bootstrap_jobs_single_node": CI_N_JOBS,
})

# --- Notebook cell 3 ---
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "This sparse CoxPH notebook requires XGBoost's linear Cox solver. "
        "For a reproducible publication environment install "
        "`%pip install \"xgboost>=2.1,<3.3\"`, restart the kernel, and rerun."
    ) from exc
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def _major_minor(version: str) -> Tuple[int, int]:
    values = []
    for token in str(version).split(".")[:2]:
        digits = "".join(character for character in token if character.isdigit())
        values.append(int(digits) if digits else 0)
    return tuple((values + [0, 0])[:2])


if _major_minor(xgb.__version__) >= (3, 3):
    warnings.warn(
        "XGBoost 3.3 deprecated gblinear. The model remains linear CoxPH, "
        "but pin xgboost>=2.1,<3.3 for a locked publication rerun.",
        FutureWarning,
    )

print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "sklearn": __import__("sklearn").__version__,
    "xgboost": xgb.__version__,
    "xgboost_booster": "gblinear",
    "xgboost_updater": "coord_descent",
    "platform": platform.platform(),
})

# --- Notebook cell 5 ---
def _env_bool(name: str, default: str = "NO") -> bool:
    return os.getenv(name, default).strip().upper() == "YES"


def _env_int_tuple(name: str, default: str) -> Tuple[int, ...]:
    return tuple(
        int(value.strip())
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )


def _env_float_tuple(name: str, default: str) -> Tuple[float, ...]:
    return tuple(
        float(value.strip())
        for value in os.getenv(name, default).split(",")
        if value.strip()
    )


@dataclass(frozen=True)
class Config:
    schema_version: str = "ehr-coxph-canonical-full-history-six-month-delayed-entry.v7.0-crescent-aligned"
    model_name: str = "CoxPH_Canonical_FullHistory_SixMonthDelayedEntry_Ridge"

    main_run_dir: str = os.getenv(
        "EHR_MAIN_RUN_DIR",
        "./runs/CRESCENT_Final",
    )
    output_root: str = os.getenv(
        "EHR_COX_OUTPUT",
        "./runs/cox",
    )
    workflow: str = os.getenv(
        "EHR_COX_WORKFLOW", "train_and_evaluate"
    ).strip().lower()
    bundle_path: str = os.getenv("EHR_COX_BUNDLE", "")

    seed: int = int(os.getenv("EHR_SEED", "42"))
    gap_months: int = int(os.getenv("EHR_GAP_MONTHS", "6"))
    report_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_REPORT_HORIZONS", "12,24,60,120"
    )
    integration_start_months: int = int(
        os.getenv("EHR_INTEGRATION_START", "12")
    )

    include_extended_history_features: bool = _env_bool(
        "EHR_COX_EXTENDED_HISTORY_FEATURES", "YES"
    )
    min_event_feature_nonzero: int = int(
        os.getenv("EHR_COX_MIN_FEATURE_NONZERO", "25")
    )
    ridge_lambdas: Tuple[float, ...] = _env_float_tuple(
        "EHR_COX_RIDGE_LAMBDAS", "0.01,0.1,1.0,10.0"
    )
    default_ridge_lambda: float = float(
        os.getenv("EHR_COX_DEFAULT_RIDGE_LAMBDA", "0.1")
    )
    training_rounds: int = int(
        os.getenv("EHR_COX_TRAINING_ROUNDS", "100")
    )
    learning_rate: float = float(
        os.getenv("EHR_COX_LEARNING_RATE", "0.5")
    )
    min_training_events_to_fit: int = int(
        os.getenv("EHR_COX_MIN_TRAIN_EVENTS", "10")
    )
    min_validation_events_to_select: int = int(
        os.getenv("EHR_COX_MIN_VAL_EVENTS", "20")
    )
    nthread: int = int(os.getenv(
        "EHR_COX_NTHREAD", str(os.cpu_count() or 1)
    ))
    prediction_batch_size: int = int(
        os.getenv("EHR_COX_PREDICTION_BATCH_SIZE", "4096")
    )

    calibration_bins: int = int(os.getenv("EHR_CALIBRATION_BINS", "10"))
    min_calibration_bin_n: int = int(
        os.getenv("EHR_MIN_CALIBRATION_BIN_N", "50")
    )
    min_calibration_group_n: int = int(
        os.getenv("EHR_MIN_CALIBRATION_GROUP_N", "100")
    )
    censor_survival_floor: float = float(
        os.getenv("EHR_CENSOR_FLOOR", "0.01")
    )
    cindex_max_cases: int = int(
        os.getenv("EHR_CINDEX_MAX_CASES", "1500")
    )
    cindex_max_comparators: int = int(
        os.getenv("EHR_CINDEX_MAX_COMPARATORS", "3000")
    )
    export_unadjusted_comparison: bool = _env_bool(
        "EHR_EXPORT_UNADJUSTED", "YES"
    )
    overwrite_model: bool = _env_bool("EHR_COX_OVERWRITE_MODEL", "NO")
    overwrite_predictions: bool = _env_bool(
        "EHR_OVERWRITE_PREDICTIONS", "NO"
    )

    ci_bootstrap_n: int = int(
        os.getenv("EHR_COX_CI_BOOTSTRAP_N", "200")
    )
    ci_bootstrap_seed: int = int(
        os.getenv("EHR_COX_CI_BOOTSTRAP_SEED", "101")
    )
    ci_alpha: float = float(os.getenv("EHR_COX_CI_ALPHA", "0.05"))
    ci_n_jobs: int = int(os.getenv(
        "EHR_COX_CI_N_JOBS", str(min(8, max(1, int(os.cpu_count() or 1))))
    ))
    ci_chunk_size: int = int(os.getenv("EHR_COX_CI_CHUNK_SIZE", "25"))
    ci_min_valid_fraction: float = float(
        os.getenv("EHR_COX_CI_MIN_VALID_FRACTION", "0.90")
    )
    ci_macro_min_condition_fraction: float = float(
        os.getenv("EHR_COX_CI_MACRO_MIN_CONDITION_FRACTION", "0.90")
    )
    ci_cindex_tau_months: int = int(
        os.getenv("EHR_COX_CI_CINDEX_TAU_MONTHS", "60")
    )
    ci_point_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_COX_CI_POINT_HORIZONS", "24,60"
    )
    ci_force_recompute: bool = _env_bool(
        "EHR_COX_CI_FORCE_RECOMPUTE", "NO"
    )


CFG = Config()
ENTRY_TIME_EPS_MONTHS = 1e-6
DELAYED_ENTRY_SCHEMA_VERSION = "crescent.delayed_entry.v1"  # same tolerance/schema as CRESCENT
if CFG.workflow not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError("EHR_COX_WORKFLOW must be train_and_evaluate or evaluation_only.")
if sorted(set(CFG.report_horizons_months)) != list(CFG.report_horizons_months):
    raise ValueError("Reporting horizons must be unique and increasing.")
if tuple(sorted(set(CFG.ridge_lambdas))) != tuple(CFG.ridge_lambdas):
    raise ValueError("Ridge penalties must be unique and increasing.")
if CFG.default_ridge_lambda not in CFG.ridge_lambdas:
    raise ValueError("The default ridge penalty must be in the candidate grid.")
if min(CFG.ridge_lambdas) < 0:
    raise ValueError("Ridge penalties must be non-negative.")
if CFG.training_rounds < 1 or not 0.0 < CFG.learning_rate <= 1.0:
    raise ValueError("Invalid coordinate-descent iteration settings.")
if CFG.min_event_feature_nonzero < 1 or CFG.nthread < 1:
    raise ValueError("Feature frequency and thread count must be positive.")
if CFG.ci_bootstrap_n < 1 or CFG.ci_chunk_size < 1:
    raise ValueError("Bootstrap replicate and chunk counts must be positive.")
if CFG.ci_cindex_tau_months != 60:
    raise ValueError("This comparison contract fixes the C-index at 60 months.")
if tuple(CFG.ci_point_horizons_months) != (24, 60):
    raise ValueError("Canonical pointwise bootstrap CIs must use 24 and 60 months.")
if CFG.gap_months != 6:
    raise ValueError("This comparator is locked to a six-month lead-time gap.")
if CFG.min_calibration_group_n != 100 or CFG.min_calibration_bin_n != 50:
    raise ValueError(
        "Canonical calibration requires groups of at least 100 and bins of at least 50."
    )
if not 0.0 < CFG.ci_alpha < 1.0:
    raise ValueError("EHR_COX_CI_ALPHA must lie strictly between 0 and 1.")
if not 0.0 < CFG.ci_min_valid_fraction <= 1.0:
    raise ValueError("EHR_COX_CI_MIN_VALID_FRACTION must lie in (0, 1].")
if not 0.0 < CFG.ci_macro_min_condition_fraction <= 1.0:
    raise ValueError(
        "EHR_COX_CI_MACRO_MIN_CONDITION_FRACTION must lie in (0, 1]."
    )

MAIN_RUN_DIR = Path(CFG.main_run_dir).expanduser().resolve()
RUN_DIR = Path(CFG.output_root).expanduser().resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)
if not (
    (MAIN_RUN_DIR / "prepared_data_source_audit.csv").is_file()
    and (MAIN_RUN_DIR / "prepared_splits_manifest.json").is_file()
):
    raise RuntimeError(
        f"{MAIN_RUN_DIR} is not a CRESCENT delayed-entry run (missing "
        "prepared_data_source_audit.csv / prepared_splits_manifest.json). Point "
        "EHR_MAIN_RUN_DIR at the folder written by the CRESCENT notebook "
        "(its EHR_CRESCENT_OUTPUT); old A10 runs use different data and are refused."
    )

CANONICAL_PATHS = {
    "train": MAIN_RUN_DIR / "canonical_train_samples.pkl",
    "val": MAIN_RUN_DIR / "canonical_val_samples.pkl",
    "test": MAIN_RUN_DIR / "canonical_test_samples.pkl",
    "target_metadata": MAIN_RUN_DIR / "canonical_target_metadata.json",
    "target_config": MAIN_RUN_DIR / "cell4_death_adjusted_target_config.json",
    "source_config": MAIN_RUN_DIR / "config.json",
    "prepared_audit": MAIN_RUN_DIR / "prepared_data_source_audit.csv",
    "prepared_manifest": MAIN_RUN_DIR / "prepared_splits_manifest.json",
    "dataloader_metadata": MAIN_RUN_DIR / "cell4.5_full_history_dataloader_metadata.json",
    "vocab_scalers": MAIN_RUN_DIR / "cell3_vocab_scalers_deathsafe.json",
    "input_manifest": MAIN_RUN_DIR / "model_input_manifest.json",
    "train_index_audit": MAIN_RUN_DIR / "cell4_train_sample_index_audit.csv",
    "val_index_audit": MAIN_RUN_DIR / "cell4_val_sample_index_audit.csv",
    "test_index_audit": MAIN_RUN_DIR / "cell4_test_sample_index_audit.csv",
}

print(json.dumps({
    **asdict(CFG),
    "history_contract": "all canonical pre-index events; no truncation",
    "resolved_main_run_dir": str(MAIN_RUN_DIR),
    "resolved_output_dir": str(RUN_DIR),
}, indent=2, default=str))

# --- Notebook cell 7 ---
def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
    os.replace(temporary, path)


def atomic_pickle_dump(value: Any, path: Path) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing canonical files. Set EHR_MAIN_RUN_DIR to the current primary-model run:\n"
            + "\n".join(missing)
        )


set_seed(CFG.seed)
