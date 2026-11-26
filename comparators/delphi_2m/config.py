from __future__ import annotations


# Delphi-2M-inspired — Configuration, precision setup, and reproducibility helpers
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 1 ---

import json
import os
from pathlib import Path


# Exact trained CRESCENT run supplied for this comparison. The environment
# override permits moving/copying that same run without changing this file.
DEFAULT_MAIN_RUN_DIR = Path(
    "runs/CRESCENT_Final"
).expanduser().resolve()
SUPPLIED_MAIN_RUN_DIR = Path(
    os.getenv("EHR_MAIN_RUN_DIR", str(DEFAULT_MAIN_RUN_DIR))
).expanduser().resolve()
CANONICAL_MAIN_RUN_DIR = str(SUPPLIED_MAIN_RUN_DIR)

# Read the source run's scientific contract for byte-identical inputs and the
# shared horizon/grid definitions. The source export remains zero lead; the
# comparator's analysis gap is fixed independently at six months below.
MAIN_CONFIG_PATH = SUPPLIED_MAIN_RUN_DIR / "config.json"
if not MAIN_CONFIG_PATH.is_file():
    raise FileNotFoundError(
        "The exact trained main-model config is required: "
        f"{MAIN_CONFIG_PATH}. Set EHR_MAIN_RUN_DIR to that run if it moved."
    )
with MAIN_CONFIG_PATH.open(encoding="utf-8") as handle:
    SUPPLIED_MAIN_CONFIG = json.load(handle)
if not isinstance(SUPPLIED_MAIN_CONFIG, dict):
    raise RuntimeError("The supplied main-model config.json must be a JSON object.")

_main_cfg = SUPPLIED_MAIN_CONFIG.get("config", {})
if not isinstance(_main_cfg, dict):
    raise RuntimeError("The supplied main-model config lacks its config object.")
CANONICAL_ENTRY_DELAY_MONTHS = int(
    _main_cfg.get(
        "ENTRY_DELAY_MONTHS",
        _main_cfg.get("GAP_WINDOW_MONTHS", 6),
    )
)
CANONICAL_GAP_MONTHS = int(_main_cfg.get("GAP_WINDOW_MONTHS", CANONICAL_ENTRY_DELAY_MONTHS))
CANONICAL_LEAD_TIME_MONTHS = int(_main_cfg.get("LEAD_TIME_MONTHS", CANONICAL_ENTRY_DELAY_MONTHS))
CANONICAL_REPORT_HORIZONS = tuple(
    int(value) for value in _main_cfg.get("EVAL_HORIZONS_MONTHS", ())
)
CANONICAL_INTEGRATION_START_MONTHS = int(
    _main_cfg.get("EVAL_INTEGRATION_START_MONTHS", 12)
)
ANALYSIS_LEAD_TIME_MONTHS = 6
_fixed_index_contract = SUPPLIED_MAIN_CONFIG.get("fixed_index_contract", {})
if CANONICAL_ENTRY_DELAY_MONTHS not in {0, 6}:
    raise RuntimeError(
        "The canonical source export must declare ENTRY_DELAY_MONTHS/GAP_WINDOW_MONTHS "
        f"in {{0, 6}}; observed {CANONICAL_ENTRY_DELAY_MONTHS}."
    )
if CANONICAL_REPORT_HORIZONS != (12, 24, 60, 120):
    raise RuntimeError(
        "The trained run must report horizons (12, 24, 60, 120); observed "
        f"{CANONICAL_REPORT_HORIZONS}."
    )
if CANONICAL_INTEGRATION_START_MONTHS != 12:
    raise RuntimeError("The trained run's dense integration grid must start at 12 months.")
if int(_fixed_index_contract.get("calendar_months_from_recruitment", -1)) != 1:
    raise RuntimeError(
        "The trained run must declare a one-calendar-month-capped index."
    )

COMPARATOR_OUTPUT_DIR = os.getenv(
    "EHR_DELPHI_OUTPUT",
    "runs/delphi_2m",
)
DEVICE_REQUEST = "auto"  # auto, cpu, cuda, cuda:0, ...

WORKFLOW = "train_and_evaluate"
PRETRAINED_CHECKPOINT = ""  # Required for evaluation_only unless output has best_checkpoint.pt.

# Locked A10 evaluation parity.
CI_BOOTSTRAP_N = 200
CINDEX_HORIZON_MONTHS = 60
LOCKED_SEED = 42
EVALUATION_SEED = 101
QUERY_EPSILON_DAYS = 1.0
VALIDATION_RISK_SUBSET_N = 0  # Zero means every validation participant.

if WORKFLOW not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError("Invalid WORKFLOW")
if CINDEX_HORIZON_MONTHS != 60:
    raise ValueError("This notebook contract fixes C-index at 60 months")

# Lock the resolved main input run for the rest of this process. Its location
# and the comparator output directory may be supplied before launch.
os.environ["EHR_DELPHI_WORKFLOW"] = WORKFLOW
os.environ["EHR_MAIN_RUN_DIR"] = CANONICAL_MAIN_RUN_DIR
os.environ["EHR_DELPHI_OUTPUT"] = COMPARATOR_OUTPUT_DIR
os.environ["EHR_DEVICE"] = DEVICE_REQUEST
os.environ["EHR_SEED"] = str(LOCKED_SEED)
os.environ["EHR_TRAINING_SEED"] = str(LOCKED_SEED)
os.environ["EHR_DELPHI_METRIC_SEED"] = str(EVALUATION_SEED)
os.environ["EHR_DELPHI_CI_BOOTSTRAP_SEED"] = str(EVALUATION_SEED)
os.environ["EHR_DELPHI_CI_BOOTSTRAP_N"] = str(CI_BOOTSTRAP_N)
os.environ["EHR_DELPHI_CI_CINDEX_TAU_MONTHS"] = str(CINDEX_HORIZON_MONTHS)
os.environ["EHR_DELPHI_QUERY_EPSILON_DAYS"] = str(QUERY_EPSILON_DAYS)
os.environ["EHR_DELPHI_VALIDATION_RISK_SUBSET_N"] = str(
    VALIDATION_RISK_SUBSET_N
)
os.environ["EHR_GAP_MONTHS"] = str(ANALYSIS_LEAD_TIME_MONTHS)
os.environ["EHR_REPORT_HORIZONS"] = ",".join(
    str(value) for value in CANONICAL_REPORT_HORIZONS
)
os.environ["EHR_INTEGRATION_START"] = str(CANONICAL_INTEGRATION_START_MONTHS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
if PRETRAINED_CHECKPOINT:
    os.environ["EHR_DELPHI_CHECKPOINT"] = PRETRAINED_CHECKPOINT
else:
    os.environ.pop("EHR_DELPHI_CHECKPOINT", None)

# --- Notebook cell 2 ---
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except Exception:
    minimize = None
    SCIPY_AVAILABLE = False


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

print({
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "torch": torch.__version__,
    "scipy_available": SCIPY_AVAILABLE,
})

# --- Notebook cell 4 ---
def _env_bool(name: str, default: str = "NO") -> bool:
    return os.getenv(name, default).strip().upper() == "YES"


def _env_int_tuple(name: str, default: str) -> Tuple[int, ...]:
    raw = os.getenv(name, default)
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


@dataclass(frozen=True)
class Config:
    schema_version: str = (
        "ehr-delphi2m-original-loss-canonical-comparator."
        "a10.v4.50k.one-month-cap-six-month-lead.fixed-v3"
    )

    main_run_dir: str = os.getenv(
        "EHR_MAIN_RUN_DIR", CANONICAL_MAIN_RUN_DIR
    )
    output_root: str = os.getenv(
        "EHR_DELPHI_OUTPUT", COMPARATOR_OUTPUT_DIR
    )
    workflow: str = os.getenv(
        "EHR_DELPHI_WORKFLOW", "train_and_evaluate"
    ).strip().lower()
    checkpoint_path: str = os.getenv("EHR_DELPHI_CHECKPOINT", "")
    resume_checkpoint: str = os.getenv("EHR_DELPHI_RESUME", "")

    seed: int = int(os.getenv("EHR_SEED", "42"))
    training_seed: int = int(os.getenv("EHR_TRAINING_SEED", "42"))
    metric_seed: int = int(os.getenv("EHR_DELPHI_METRIC_SEED", "101"))
    report_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_REPORT_HORIZONS",
        ",".join(str(value) for value in CANONICAL_REPORT_HORIZONS),
    )
    gap_months: int = int(
        os.getenv("EHR_GAP_MONTHS", str(ANALYSIS_LEAD_TIME_MONTHS))
    )
    integration_start_months: int = int(
        os.getenv(
            "EHR_INTEGRATION_START",
            str(CANONICAL_INTEGRATION_START_MONTHS),
        )
    )

    # Official full Delphi-2M architecture/configuration defaults.
    block_size: int = int(os.getenv("EHR_DELPHI_BLOCK_SIZE", "96"))
    d_model: int = int(os.getenv("EHR_DELPHI_D_MODEL", "120"))
    n_heads: int = int(os.getenv("EHR_DELPHI_N_HEADS", "12"))
    n_layers: int = int(os.getenv("EHR_DELPHI_N_LAYERS", "12"))
    d_ff: int = int(os.getenv("EHR_DELPHI_D_FF", "480"))
    dropout: float = float(os.getenv("EHR_DELPHI_DROPOUT", "0.10"))
    token_dropout: float = float(os.getenv("EHR_DELPHI_TOKEN_DROPOUT", "0.05"))
    bias: bool = _env_bool("EHR_DELPHI_BIAS", "NO")
    use_input_norm: bool = _env_bool("EHR_DELPHI_INPUT_NORM", "YES")
    use_output_bias: bool = _env_bool("EHR_DELPHI_OUTPUT_BIAS", "NO")
    mask_ties: bool = _env_bool("EHR_DELPHI_MASK_TIES", "YES")
    t_min_days: float = float(os.getenv("EHR_DELPHI_T_MIN_DAYS", "0.1"))
    origin_age_years: float = float(os.getenv("EHR_DELPHI_ORIGIN_AGE_YEARS", "35.0"))
    no_event_interval_months: float = float(
        os.getenv("EHR_DELPHI_NO_EVENT_INTERVAL_MONTHS", "60")
    )
    query_epsilon_days: float = float(
        os.getenv("EHR_DELPHI_QUERY_EPSILON_DAYS", "1")
    )
    include_right_censor_survival: bool = False
    censor_aware_no_event_padding: bool = True
    paper_rate_exp_logits: bool = True
    post_index_target_output_restriction: bool = _env_bool(
        "EHR_DELPHI_POST_INDEX_RESTRICTION", "YES"
    )
    dynamic_first_event_risk_sets: bool = _env_bool(
        "EHR_DELPHI_DYNAMIC_RISK_SETS", "YES"
    )
    selection_post_index_targets_only: bool = False
    selection_exclude_no_event: bool = _env_bool(
        "EHR_DELPHI_SELECTION_EXCLUDE_NO_EVENT", "YES"
    )
    selection_include_censor_survival: bool = False
    full_vocabulary_ar_likelihood: bool = True
    landmark_auxiliary_weight: float = 0.0
    landmark_brier_weight: float = 0.0
    validation_risk_subset_n: int = int(
        os.getenv("EHR_DELPHI_VALIDATION_RISK_SUBSET_N", "0")
    )
    validation_risk_horizons_months: Tuple[int, ...] = _env_int_tuple(
        "EHR_DELPHI_VALIDATION_RISK_HORIZONS",
        ",".join(str(value) for value in CANONICAL_REPORT_HORIZONS),
    )
    validation_macro_brier_weight: float = float(
        os.getenv("EHR_DELPHI_VALIDATION_BRIER_WEIGHT", "0")
    )
    validation_macro_auc_weight: float = float(
        os.getenv("EHR_DELPHI_VALIDATION_AUC_WEIGHT", "0")
    )
    enable_validation_rate_scaling: bool = _env_bool(
        "EHR_DELPHI_VALIDATION_RATE_SCALING", "NO"
    )
    rate_scaling_prior_events: float = float(
        os.getenv("EHR_DELPHI_RATE_SCALING_PRIOR_EVENTS", "5")
    )
    rate_scaling_log_limit: float = float(
        os.getenv("EHR_DELPHI_RATE_SCALING_LOG_LIMIT", "3")
    )

    max_iterations: int = int(os.getenv("EHR_DELPHI_MAX_ITERATIONS", "50000"))
    validation_interval: int = int(
        os.getenv("EHR_DELPHI_VALIDATION_INTERVAL", "1000")
    )
    validation_risk_report_interval: int = int(
        os.getenv("EHR_DELPHI_VALIDATION_RISK_REPORT_INTERVAL", "5000")
    )
    validation_batches: int = int(
        os.getenv("EHR_DELPHI_VALIDATION_BATCHES", "0")
    )
    batch_size: int = int(os.getenv("EHR_BATCH_SIZE", "128"))
    learning_rate: float = float(os.getenv("EHR_LEARNING_RATE", "3e-4"))
    min_learning_rate: float = float(os.getenv("EHR_MIN_LEARNING_RATE", "3e-5"))
    weight_decay: float = float(os.getenv("EHR_WEIGHT_DECAY", "0.1"))
    ema_decay: float = float(os.getenv("EHR_DELPHI_EMA_DECAY", "0.999"))
    beta1: float = float(os.getenv("EHR_ADAM_BETA1", "0.9"))
    beta2: float = float(os.getenv("EHR_ADAM_BETA2", "0.99"))
    warmup_iterations: int = int(os.getenv("EHR_WARMUP_ITERATIONS", "1000"))
    patience_validations: int = int(os.getenv("EHR_PATIENCE_VALIDATIONS", "0"))
    min_delta: float = float(os.getenv("EHR_MIN_DELTA", "1e-4"))
    max_gradient_norm: float = float(os.getenv("EHR_MAX_GRADIENT_NORM", "1.0"))
    num_workers: int = int(os.getenv("EHR_NUM_WORKERS", "0"))
    precision: str = os.getenv("EHR_PRECISION", "auto").strip().lower()
    log_validation_metrics: bool = _env_bool(
        "EHR_LOG_VALIDATION_METRICS", "YES"
    )
    run_preflight_batch: bool = _env_bool("EHR_RUN_PREFLIGHT_BATCH", "YES")

    calibration_bins: int = int(os.getenv("EHR_CALIBRATION_BINS", "10"))
    min_calibration_group_n: int = int(
        os.getenv("EHR_MIN_CALIBRATION_GROUP_N", "100")
    )
    min_calibration_bin_n: int = int(os.getenv("EHR_MIN_CALIBRATION_BIN_N", "50"))
    censor_survival_floor: float = float(os.getenv("EHR_CENSOR_FLOOR", "0.01"))
    cindex_max_cases: int = int(os.getenv("EHR_CINDEX_MAX_CASES", "1500"))
    cindex_max_comparators: int = int(os.getenv("EHR_CINDEX_MAX_COMPARATORS", "3000"))
    export_unadjusted_comparison: bool = _env_bool("EHR_EXPORT_UNADJUSTED", "YES")
    export_full_prediction_curves: bool = _env_bool("EHR_EXPORT_FULL_CURVES", "YES")
    overwrite_predictions: bool = _env_bool("EHR_OVERWRITE_PREDICTIONS", "NO")

    # Participant-level percentile bootstrap. These evaluation-only settings
    # are excluded from the scientific checkpoint hash.
    ci_bootstrap_n: int = int(
        os.getenv("EHR_DELPHI_CI_BOOTSTRAP_N", "200")
    )
    ci_bootstrap_seed: int = int(
        os.getenv("EHR_DELPHI_CI_BOOTSTRAP_SEED", "101")
    )
    ci_alpha: float = float(
        os.getenv("EHR_DELPHI_CI_ALPHA", "0.05")
    )
    ci_n_jobs: int = int(
        os.getenv("EHR_DELPHI_CI_N_JOBS", "8")
    )
    ci_chunk_size: int = int(
        os.getenv("EHR_DELPHI_CI_CHUNK_SIZE", "25")
    )
    ci_min_valid_fraction: float = float(
        os.getenv("EHR_DELPHI_CI_MIN_VALID_FRACTION", "0.90")
    )
    ci_macro_min_condition_fraction: float = float(
        os.getenv("EHR_DELPHI_CI_MACRO_MIN_CONDITION_FRACTION", "0.90")
    )
    ci_cindex_tau_months: int = int(
        os.getenv("EHR_DELPHI_CI_CINDEX_TAU_MONTHS", "60")
    )
    ci_force_recompute: bool = _env_bool(
        "EHR_DELPHI_CI_FORCE_RECOMPUTE", "NO"
    )


CFG = Config()
if CFG.workflow not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError(
        "EHR_DELPHI_WORKFLOW must be train_and_evaluate or evaluation_only."
    )
if CFG.ci_bootstrap_n < 1:
    raise ValueError("EHR_DELPHI_CI_BOOTSTRAP_N must be at least 1.")
if CFG.ci_chunk_size < 1:
    raise ValueError("EHR_DELPHI_CI_CHUNK_SIZE must be at least 1.")
if CFG.ci_cindex_tau_months != 60:
    raise ValueError(
        "This notebook contract fixes the C-index horizon at 60 months."
    )
if not 0.0 < CFG.ci_alpha < 1.0:
    raise ValueError("EHR_DELPHI_CI_ALPHA must lie strictly between 0 and 1.")
if not 0.0 < CFG.ci_min_valid_fraction <= 1.0:
    raise ValueError(
        "EHR_DELPHI_CI_MIN_VALID_FRACTION must lie in (0, 1]."
    )
if not 0.0 < CFG.ci_macro_min_condition_fraction <= 1.0:
    raise ValueError(
        "EHR_DELPHI_CI_MACRO_MIN_CONDITION_FRACTION must lie in (0, 1]."
    )
if CFG.precision not in {"auto", "fp32", "bf16", "fp16"}:
    raise ValueError("EHR_PRECISION must be auto, fp32, bf16, or fp16.")
if CFG.block_size < 2:
    raise ValueError("EHR_DELPHI_BLOCK_SIZE must be at least 2.")
if CFG.d_model % 2 or CFG.d_model % CFG.n_heads:
    raise ValueError("Delphi d_model must be even and divisible by n_heads.")
if CFG.d_ff != 4 * CFG.d_model:
    raise ValueError("The paper-faithful Delphi feed-forward width is 4*d_model.")
if CFG.max_iterations < 1 or CFG.validation_interval < 1:
    raise ValueError("Training and validation iteration counts must be positive.")
if CFG.batch_size < 1:
    raise ValueError("EHR_BATCH_SIZE must be at least 1.")
if CFG.t_min_days <= 0 or CFG.no_event_interval_months <= 0:
    raise ValueError("Delphi time stabilizer and no-event interval must be positive.")
if CFG.query_epsilon_days <= 0:
    raise ValueError("EHR_DELPHI_QUERY_EPSILON_DAYS must be positive.")
if CFG.origin_age_years < 0.0 or CFG.origin_age_years > 60.0:
    raise ValueError("EHR_DELPHI_ORIGIN_AGE_YEARS must lie in [0, 60].")
if not (0.0 <= CFG.ema_decay < 1.0):
    raise ValueError("EHR_DELPHI_EMA_DECAY must lie in [0, 1).")
if CFG.bias:
    raise ValueError(
        "The Delphi autoregressive comparator requires EHR_DELPHI_BIAS=NO."
    )
if CFG.include_right_censor_survival or CFG.selection_include_censor_survival:
    raise ValueError(
        "The original Delphi loss does not add a separate terminal "
        "administrative-censor survival term."
    )
if not CFG.paper_rate_exp_logits or not CFG.full_vocabulary_ar_likelihood:
    raise ValueError(
        "The original-loss baseline requires exp(logit) Delphi rates."
    )
if CFG.selection_post_index_targets_only:
    raise ValueError(
        "Keep full-trajectory validation selection for the original-loss baseline."
    )
if not CFG.censor_aware_no_event_padding:
    raise ValueError("The original-loss baseline retains NO_EVENT padding.")
if (
    CFG.landmark_auxiliary_weight != 0
    or CFG.landmark_brier_weight != 0
    or CFG.validation_risk_subset_n < 0
    or CFG.validation_macro_brier_weight != 0
    or CFG.validation_macro_auc_weight != 0
    or CFG.enable_validation_rate_scaling
    or CFG.rate_scaling_prior_events <= 0
    or CFG.rate_scaling_log_limit <= 0
):
    raise ValueError(
        "The paper-faithful original-loss baseline requires zero auxiliary weights "
        "and disables post-hoc validation rate scaling."
    )
if tuple(CFG.validation_risk_horizons_months) != tuple(
    CFG.report_horizons_months
):
    raise ValueError(
        "The original-loss baseline must report validation discrimination at "
        "all locked horizons: 12, 24, 60 and 120 months."
    )
if not (CFG.seed == CFG.training_seed == 42):
    raise ValueError(
        "The locked Delphi model and training streams require seed 42."
    )
if not (CFG.metric_seed == CFG.ci_bootstrap_seed == 101):
    raise ValueError(
        "The A10-aligned metric and participant-bootstrap streams require seed 101."
    )
if CFG.gap_months != ANALYSIS_LEAD_TIME_MONTHS:
    raise ValueError(
        "This comparator fixes the conditional analysis lead time at six months."
    )
if CFG.report_horizons_months != CANONICAL_REPORT_HORIZONS:
    raise ValueError(
        "The comparator reporting horizons must match the canonical source: "
        f"{CANONICAL_REPORT_HORIZONS}."
    )
if CFG.integration_start_months != CANONICAL_INTEGRATION_START_MONTHS:
    raise ValueError(
        "The comparator dense integration start must exactly match the trained "
        f"main run ({CANONICAL_INTEGRATION_START_MONTHS} months)."
    )
if not (
    CFG.calibration_bins == 10
    and CFG.min_calibration_group_n == 100
    and CFG.min_calibration_bin_n == 50
    and np.isclose(CFG.censor_survival_floor, 0.01)
):
    raise ValueError(
        "The A10 metric contract fixes 10 calibration bins, minimum group n=100, "
        "minimum bin n=50, and censoring-survival floor=0.01."
    )
if not (
    CFG.export_unadjusted_comparison
    and CFG.export_full_prediction_curves
):
    raise ValueError(
        "The locked output schema requires adjusted, unadjusted, death, and "
        "dense prediction arrays; keep both export flags enabled."
    )
if not CFG.report_horizons_months:
    raise ValueError("At least one reporting horizon is required.")
if sorted(set(CFG.report_horizons_months)) != list(CFG.report_horizons_months):
    raise ValueError("EHR_REPORT_HORIZONS must be unique and increasing.")


MAIN_RUN_DIR = Path(CFG.main_run_dir).expanduser().resolve()
if MAIN_RUN_DIR != SUPPLIED_MAIN_RUN_DIR:
    raise RuntimeError(
        "The comparator main_run_dir changed after the exact supplied run was locked."
    )
RUN_DIR = Path(CFG.output_root).expanduser().resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)

CANONICAL_PATHS = {
    "train": MAIN_RUN_DIR / "canonical_train_samples.pkl",
    "val": MAIN_RUN_DIR / "canonical_val_samples.pkl",
    "test": MAIN_RUN_DIR / "canonical_test_samples.pkl",
    "target_metadata": MAIN_RUN_DIR / "canonical_target_metadata.json",
    "target_config": MAIN_RUN_DIR / "cell4_death_adjusted_target_config.json",
    "vocab_scalers": MAIN_RUN_DIR / "cell3_vocab_scalers_deathsafe.json",
    "input_manifest": MAIN_RUN_DIR / "model_input_manifest.json",
    "dataloader_metadata": (
        MAIN_RUN_DIR / "cell4.5_full_history_dataloader_metadata.json"
        if (MAIN_RUN_DIR / "cell4.5_full_history_dataloader_metadata.json").is_file()
        else MAIN_RUN_DIR / "cell4.5_dataloader_metadata_death_adjusted.json"
    ),
    "main_config": MAIN_RUN_DIR / "config.json",
    "main_per_condition_metrics": (
        MAIN_RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
    ),
    "train_index_audit": MAIN_RUN_DIR / "cell4_train_sample_index_audit.csv",
    "val_index_audit": MAIN_RUN_DIR / "cell4_val_sample_index_audit.csv",
    "test_index_audit": MAIN_RUN_DIR / "cell4_test_sample_index_audit.csv",
}

print(json.dumps({
    **asdict(CFG),
    "resolved_main_run_dir": str(MAIN_RUN_DIR),
    "resolved_output_dir": str(RUN_DIR),
    "raw_sources_opened_by_comparator": False,
}, indent=2, default=str))

# --- Notebook cell 6 ---
def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_json_dump(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)
    os.replace(temp, path)


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(value), temp)
    os.replace(temp, path)


def require_files(paths: Iterable[Path]) -> None:
    missing = [str(Path(path)) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def choose_device() -> torch.device:
    requested = os.getenv("EHR_DEVICE", "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"EHR_DEVICE={requested!r} requires CUDA.")
    return device


def choose_precision(device: torch.device, requested: str) -> str:
    if device.type != "cuda":
        return "fp32"
    if requested == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if requested == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 was requested but is unsupported by this GPU.")
    return requested


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def capture_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


set_seed(CFG.seed)
DEVICE = choose_device()
PRECISION = choose_precision(DEVICE, CFG.precision)
print({"device": str(DEVICE), "precision": PRECISION})



def torch_load_compat(path: Path, map_location: Any):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)
