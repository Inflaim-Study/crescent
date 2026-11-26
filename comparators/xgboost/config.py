from __future__ import annotations


# XGBoost — Configuration, environment checks, and reproducibility helpers
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 1 ---

import os
from pathlib import Path

# -----------------------------------------------------------------------------
# Required: point this at the exact primary-model run being compared.
# The run must contain the canonical train/validation/test sample pickles and
# their metadata/manifest sidecars listed in the configuration cell below.
# -----------------------------------------------------------------------------
WORKFLOW = "train_and_evaluate"  # or "evaluation_only"
# Folder where the CRESCENT notebook and its outputs live.
CRESCENT_BASE_DIR = os.getenv(
    "EHR_CRESCENT_BASE_DIR",
    ".",
).strip()
# The CRESCENT run folder (its EHR_CRESCENT_OUTPUT). Override with EHR_MAIN_RUN_DIR
# if the CRESCENT run was written elsewhere. The folder is validated below: it must
# contain prepared_data_source_audit.csv and prepared_splits_manifest.json, which
# only the new CRESCENT notebook writes; old runs are rejected.
MAIN_RUN_DIR = os.getenv(
    "EHR_MAIN_RUN_DIR",
    str(Path(CRESCENT_BASE_DIR) / "runs" / "CRESCENT_Final"),
).strip()
OUTPUT_ROOT = os.getenv(
    "EHR_XGB_OUTPUT",
    str(Path(CRESCENT_BASE_DIR) / "runs" / "xgboost"),
).strip()

# Required for evaluation_only unless the bundle already exists in OUTPUT_ROOT.
PRETRAINED_BUNDLE = ""  # e.g. "/path/to/frozen_xgboost_survival_full_history_bundle.pkl"

# Optional primary-model metric-definition/evaluation-manifest JSON. When left
# blank, the notebook searches common filenames inside MAIN_RUN_DIR. Set
# REQUIRE_MAIN_METRIC_CONTRACT=True for a hard publication-time requirement.
MAIN_METRIC_CONTRACT_PATH = os.getenv("EHR_MAIN_METRIC_CONTRACT", "").strip()
REQUIRE_MAIN_METRIC_CONTRACT = False

# These settings define the like-for-like estimand and metric axes. They are
# written into the frozen bundle and prediction/evaluation manifests.
GAP_MONTHS = 6
REPORT_HORIZONS_MONTHS = (12, 24, 60, 120)
INTEGRATION_START_MONTHS = 12
CINDEX_HORIZON_MONTHS = 60
CI_POINT_HORIZONS_MONTHS = (24, 60)

# Model selection and uncertainty settings.
MIN_VALIDATION_EVENTS_FOR_TUNING = 20
MIN_EVENT_FEATURE_NONZERO = 25
VALIDATION_PRINT_INTERVAL = 1000
TUNING_TRIALS = 40
MAX_BOOST_ROUNDS = 10000
V4_BASELINE_EARLY_STOPPING_ROUNDS = 20
TUNING_EARLY_STOPPING_ROUNDS = 100
MIN_VALIDATION_CINDEX_GAIN = 0.0
VALIDATION_EVENT_GAIN_SCALE = 0.0
CI_BOOTSTRAP_N = 200  # use 20 only for a smoke test
CI_BOOTSTRAP_SEED = 101
CI_N_JOBS = 8
# Optional: verify hashes against an existing Cox V6 run before fitting.
COX_REFERENCE_OUTPUT_ROOT = os.getenv("EHR_COX_REFERENCE_OUTPUT", "").strip()
# Temporary dense feature batches; reduce for a wider retained feature axis.
DMATRIX_BATCH_SIZE = 1024
XGB_MAX_BIN = 256

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
    raise ValueError("The C-index horizon must be one of the report horizons.")
if tuple(CI_POINT_HORIZONS_MONTHS) != (24, 60):
    raise ValueError("Canonical pointwise bootstrap CIs must use 24 and 60 months.")
if MIN_EVENT_FEATURE_NONZERO < 1 or VALIDATION_PRINT_INTERVAL < 1:
    raise ValueError("Feature-frequency and validation-print settings must be positive.")

os.environ["EHR_COX_REFERENCE_OUTPUT"] = COX_REFERENCE_OUTPUT_ROOT
os.environ["EHR_XGB_DMATRIX_BATCH_SIZE"] = str(DMATRIX_BATCH_SIZE)
os.environ["EHR_XGB_MAX_BIN"] = str(XGB_MAX_BIN)
os.environ["EHR_XGB_WORKFLOW"] = WORKFLOW
os.environ["EHR_MAIN_RUN_DIR"] = str(Path(MAIN_RUN_DIR).expanduser())
os.environ["EHR_XGB_OUTPUT"] = str(Path(OUTPUT_ROOT).expanduser())
os.environ["EHR_GAP_MONTHS"] = str(GAP_MONTHS)
os.environ["EHR_REPORT_HORIZONS"] = ",".join(map(str, REPORT_HORIZONS_MONTHS))
os.environ["EHR_INTEGRATION_START"] = str(INTEGRATION_START_MONTHS)
os.environ["EHR_XGB_MIN_VAL_EVENTS"] = str(MIN_VALIDATION_EVENTS_FOR_TUNING)
os.environ["EHR_XGB_MIN_FEATURE_NONZERO"] = str(MIN_EVENT_FEATURE_NONZERO)
os.environ["EHR_XGB_VALIDATION_PRINT_INTERVAL"] = str(VALIDATION_PRINT_INTERVAL)
os.environ["EHR_XGB_TUNING_TRIALS"] = str(TUNING_TRIALS)
os.environ["EHR_XGB_MAX_BOOST_ROUNDS"] = str(MAX_BOOST_ROUNDS)
os.environ["EHR_XGB_BASELINE_EARLY_STOPPING_ROUNDS"] = str(
    V4_BASELINE_EARLY_STOPPING_ROUNDS
)
os.environ["EHR_XGB_EARLY_STOPPING_ROUNDS"] = str(
    TUNING_EARLY_STOPPING_ROUNDS
)
os.environ["EHR_XGB_MIN_CINDEX_GAIN"] = str(MIN_VALIDATION_CINDEX_GAIN)
os.environ["EHR_XGB_EVENT_GAIN_SCALE"] = str(VALIDATION_EVENT_GAIN_SCALE)
os.environ["EHR_XGB_CI_BOOTSTRAP_N"] = str(CI_BOOTSTRAP_N)
os.environ["EHR_XGB_CI_BOOTSTRAP_SEED"] = str(CI_BOOTSTRAP_SEED)
os.environ["EHR_XGB_CI_N_JOBS"] = str(CI_N_JOBS)
os.environ["EHR_XGB_CI_CINDEX_TAU_MONTHS"] = str(CINDEX_HORIZON_MONTHS)
os.environ["EHR_XGB_CI_POINT_HORIZONS"] = ",".join(
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
    os.environ["EHR_XGB_BUNDLE"] = PRETRAINED_BUNDLE
else:
    os.environ.pop("EHR_XGB_BUNDLE", None)

print({
    "workflow": WORKFLOW,
    "main_run_dir": str(Path(MAIN_RUN_DIR).expanduser().resolve()),
    "output_root": str(Path(OUTPUT_ROOT).expanduser().resolve()),
    "gap_months": GAP_MONTHS,
    "report_horizons_months": REPORT_HORIZONS_MONTHS,
    "integration_start_months": INTEGRATION_START_MONTHS,
    "cindex_horizon_months": CINDEX_HORIZON_MONTHS,
    "history_contract": "all canonical pre-index events; no truncation",
    "minimum_event_feature_nonzero": MIN_EVENT_FEATURE_NONZERO,
    "validation_print_interval": VALIDATION_PRINT_INTERVAL,
    "tuning_trials": TUNING_TRIALS,
    "max_boost_rounds": MAX_BOOST_ROUNDS,
    "v4_baseline_early_stopping_rounds": V4_BASELINE_EARLY_STOPPING_ROUNDS,
    "tuning_early_stopping_rounds": TUNING_EARLY_STOPPING_ROUNDS,
    "minimum_validation_cindex_gain": MIN_VALIDATION_CINDEX_GAIN,
    "validation_event_gain_scale": VALIDATION_EVENT_GAIN_SCALE,
    "ci_point_horizons_months": CI_POINT_HORIZONS_MONTHS,
    "bootstrap_replicates": CI_BOOTSTRAP_N,
    "require_main_metric_contract": REQUIRE_MAIN_METRIC_CONTRACT,
})

# --- Notebook cell 2 ---
import hashlib
import json
import math
import os
import pickle
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.optimize import minimize
try:
    import optuna
    import xgboost as xgb
except ImportError as exc:
    raise ImportError(
        "This notebook requires xgboost and optuna. Install once with "
        "`%pip install \"xgboost>=2.1,<3.3\" \"optuna>=3.6,<5\"`, restart "
        "the kernel, and rerun from the top."
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

from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "sklearn": __import__("sklearn").__version__,
    "xgboost": xgb.__version__,
    "optuna": optuna.__version__,
    "platform": platform.platform(),
})

# --- Notebook cell 4 ---
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
    schema_version: str = "ehr-xgboost-survival-canonical-full-history-six-month-delayed-entry.v8.0-cox-matched-crescent-aligned"
    model_name: str = "XGBoostSurvival_Canonical_FullHistory_SixMonthDelayedEntry_CoxMatched"


    main_run_dir: str = os.getenv(
        "EHR_MAIN_RUN_DIR",
        "./runs/CRESCENT_Final",
    )


    output_root: str = os.getenv(
        "EHR_XGB_OUTPUT",
        "./runs/xgboost",
    )
    workflow: str = os.getenv(
        "EHR_XGB_WORKFLOW", "evaluation_only"
    ).strip().lower()
    bundle_path: str = os.getenv("EHR_XGB_BUNDLE", "")

    seed: int = int(os.getenv("EHR_SEED", "42"))
    gap_months: int = int(os.getenv("EHR_GAP_MONTHS", "6"))
    report_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_REPORT_HORIZONS", "12,24,60,120"
    )
    integration_start_months: int = int(
        os.getenv("EHR_INTEGRATION_START", "12")
    )

    # Tree-specific search with a prespecified default candidate. The anchor preserves the exact
    # V4 patience while lower-learning-rate candidates receive longer patience.
    tuning_trials: int = int(os.getenv("EHR_XGB_TUNING_TRIALS", "40"))
    max_boost_rounds: int = int(
        os.getenv("EHR_XGB_MAX_BOOST_ROUNDS", "10000")
    )
    baseline_early_stopping_rounds: int = int(
        os.getenv("EHR_XGB_BASELINE_EARLY_STOPPING_ROUNDS", "20")
    )
    early_stopping_rounds: int = int(
        os.getenv("EHR_XGB_EARLY_STOPPING_ROUNDS", "100")
    )
    min_validation_cindex_gain: float = float(
        os.getenv("EHR_XGB_MIN_CINDEX_GAIN", "0.0")
    )
    validation_event_gain_scale: float = float(
        os.getenv("EHR_XGB_EVENT_GAIN_SCALE", "0.0")
    )
    nthread: int = int(os.getenv("EHR_XGB_NTHREAD", "-1"))
    tree_method: str = os.getenv("EHR_XGB_TREE_METHOD", "hist")
    max_bin: int = int(os.getenv("EHR_XGB_MAX_BIN", "256"))
    dmatrix_batch_size: int = int(os.getenv("EHR_XGB_DMATRIX_BATCH_SIZE", "1024"))
    cox_reference_output_root: str = os.getenv("EHR_COX_REFERENCE_OUTPUT", "").strip()
    device: str = os.getenv("EHR_XGB_DEVICE", "cpu")
    retain_default_variant: bool = _env_bool(
        "EHR_XGB_RETAIN_DEFAULT_VARIANT", "YES"
    )
    include_extended_history_features: bool = _env_bool(
        "EHR_XGB_EXTENDED_HISTORY_FEATURES", "YES"
    )
    min_event_feature_nonzero: int = int(
        os.getenv("EHR_XGB_MIN_FEATURE_NONZERO", "25")
    )
    validation_print_interval: int = int(
        os.getenv("EHR_XGB_VALIDATION_PRINT_INTERVAL", "1000")
    )
    min_training_events_to_fit: int = int(
        os.getenv("EHR_XGB_MIN_TRAIN_EVENTS", "10")
    )
    min_validation_events_to_select: int = int(
        os.getenv("EHR_XGB_MIN_VAL_EVENTS", "20")
    )
    prediction_batch_size: int = int(
        os.getenv("EHR_XGB_PREDICTION_BATCH_SIZE", "4096")
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
    overwrite_model: bool = _env_bool("EHR_XGB_OVERWRITE_MODEL", "NO")
    overwrite_predictions: bool = _env_bool(
        "EHR_OVERWRITE_PREDICTIONS", "NO"
    )


    # Participant-level percentile bootstrap. These evaluation-only settings
    # do not alter the fitted XGBoost model bundle.
    ci_bootstrap_n: int = int(
        os.getenv("EHR_XGB_CI_BOOTSTRAP_N", "200")
    )
    ci_bootstrap_seed: int = int(
        os.getenv("EHR_XGB_CI_BOOTSTRAP_SEED", "101")
    )
    ci_alpha: float = float(
        os.getenv("EHR_XGB_CI_ALPHA", "0.05")
    )
    ci_n_jobs: int = int(
        os.getenv("EHR_XGB_CI_N_JOBS", "8")
    )
    ci_chunk_size: int = int(
        os.getenv("EHR_XGB_CI_CHUNK_SIZE", "25")
    )
    ci_min_valid_fraction: float = float(
        os.getenv("EHR_XGB_CI_MIN_VALID_FRACTION", "0.90")
    )
    ci_macro_min_condition_fraction: float = float(
        os.getenv("EHR_XGB_CI_MACRO_MIN_CONDITION_FRACTION", "0.90")
    )
    ci_cindex_tau_months: int = int(
        os.getenv("EHR_XGB_CI_CINDEX_TAU_MONTHS", "60")
    )
    ci_point_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_XGB_CI_POINT_HORIZONS", "24,60"
    )
    ci_force_recompute: bool = _env_bool(
        "EHR_XGB_CI_FORCE_RECOMPUTE", "NO"
    )


CFG = Config()
# Same tolerance as CRESCENT (ENTRY_TIME_EPS_MONTHS): entry requires follow-up
# strictly beyond month six; events at or before month six exclude the pair.
ENTRY_TIME_EPS_MONTHS = 1e-6
DELAYED_ENTRY_SCHEMA_VERSION = "crescent.delayed_entry.v1"
if CFG.tree_method != "hist":
    raise ValueError("V7's zero-filled QuantileDMatrix adapter requires tree_method='hist'.")
if CFG.max_bin < 2 or CFG.dmatrix_batch_size < 1:
    raise ValueError("max_bin must be >=2 and dmatrix_batch_size must be positive.")
if CFG.min_validation_cindex_gain != 0 or CFG.validation_event_gain_scale != 0:
    raise ValueError("Cox-matched selection requires both validation gain thresholds to be zero.")
version_numbers = tuple(int(part) for part in xgb.__version__.split('.')[:2])
if not (version_numbers >= (2, 1) and version_numbers < (3, 3)):
    raise RuntimeError("Use a pinned XGBoost version >=2.1,<3.3 for the matched Cox comparison.")
if CFG.workflow not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError(
        "EHR_XGB_WORKFLOW must be train_and_evaluate or evaluation_only."
    )
if CFG.ci_bootstrap_n < 1:
    raise ValueError("EHR_XGB_CI_BOOTSTRAP_N must be at least 1.")
if CFG.ci_chunk_size < 1:
    raise ValueError("EHR_XGB_CI_CHUNK_SIZE must be at least 1.")
if CFG.ci_cindex_tau_months != 60:
    raise ValueError(
        "This notebook contract fixes the C-index horizon at 60 months."
    )
if tuple(CFG.ci_point_horizons_months) != (24, 60):
    raise ValueError("Canonical pointwise bootstrap CIs must use 24 and 60 months.")
if CFG.gap_months != 6:
    raise ValueError("This comparator is locked to a six-month lead-time gap.")
if CFG.min_event_feature_nonzero < 1 or CFG.validation_print_interval < 1:
    raise ValueError("Feature-frequency and validation-print settings must be positive.")
if CFG.min_calibration_group_n != 100 or CFG.min_calibration_bin_n != 50:
    raise ValueError(
        "Canonical calibration requires groups of at least 100 and bins of at least 50."
    )
if not 0.0 < CFG.ci_alpha < 1.0:
    raise ValueError("EHR_XGB_CI_ALPHA must lie strictly between 0 and 1.")
if not 0.0 < CFG.ci_min_valid_fraction <= 1.0:
    raise ValueError(
        "EHR_XGB_CI_MIN_VALID_FRACTION must lie in (0, 1]."
    )
if not 0.0 < CFG.ci_macro_min_condition_fraction <= 1.0:
    raise ValueError(
        "EHR_XGB_CI_MACRO_MIN_CONDITION_FRACTION must lie in (0, 1]."
    )
if CFG.tuning_trials < 1:
    raise ValueError("At least one XGBoost tuning trial is required.")
if (
    CFG.max_boost_rounds < 1
    or CFG.early_stopping_rounds < 1
    or CFG.baseline_early_stopping_rounds < 1
):
    raise ValueError("Boosting and early-stopping rounds must be positive.")
if CFG.min_validation_cindex_gain < 0 or CFG.validation_event_gain_scale < 0:
    raise ValueError("Validation gain thresholds must be non-negative.")
if sorted(set(CFG.report_horizons_months)) != list(CFG.report_horizons_months):
    raise ValueError("Reporting horizons must be unique and increasing.")

MAIN_RUN_DIR = Path(CFG.main_run_dir).expanduser().resolve()
RUN_DIR = Path(CFG.output_root).expanduser().resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)

# The first configuration cell always exports EHR_MAIN_RUN_DIR, so an "unset"
# warning can never fire. Instead the folder must prove it is a CRESCENT run.
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

# --- Notebook cell 6 ---
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
            "Missing canonical files. Set EHR_MAIN_RUN_DIR to the canonical A10 run:\n"
            + "\n".join(missing)
        )


set_seed(CFG.seed)
