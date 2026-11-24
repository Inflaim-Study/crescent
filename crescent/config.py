"""Stage 1: configuration, delayed-entry contract and input verification.

Extracted from the CRESCENT_Final notebook (cell 2). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 2
# =============================================================================

# CELL 1 — SIX-MONTH DELAYED-ENTRY CONFIGURATION ON THE NEWLY PREPARED DATA
# This cell defines the delayed-entry contract. The likelihood, rate
# initialization, and evaluation below all apply entry at month six.

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os
import random
import shutil

import numpy as np
import pandas as pd
import torch


# =============================================================================
# CELL 1 — LOCK THE SAME CANONICAL A10 SOURCE USED BY THE DELPHI COMPARATOR
# =============================================================================

# =============================================================================
# CELL 1 — LOCATE AND VERIFY THE NEWLY PREPARED DATA (relative paths only)
# =============================================================================
# The data-preparation notebook writes train.csv / val.csv / test.csv,
# splits_manifest.json and static_covariate_vocabulary.json into
# ./splits_229950_one_month_cap, next to the notebooks. This notebook reads them
# from there; nothing is read from an old "canonical A10" run directory.
#   CRESCENT_PREPARED_DATA_DIR  overrides the data folder
#   EHR_CRESCENT_FINAL_OUTPUT         overrides the output folder (must be fresh)

PREPARED_DATA_DIR = Path(
    os.getenv("CRESCENT_PREPARED_DATA_DIR", "splits_229950_one_month_cap")
).expanduser().resolve()

RUN_DIR = Path(
    os.getenv(
        "EHR_CRESCENT_FINAL_OUTPUT",
        "runs/CRESCENT_Final",
    )
).expanduser().resolve()

if RUN_DIR == PREPARED_DATA_DIR or PREPARED_DATA_DIR in RUN_DIR.parents:
    raise ValueError(
        "EHR_CRESCENT_FINAL_OUTPUT must not be the prepared-data folder or inside it; "
        "the prepared files are read-only inputs."
    )

PREPARED_PATHS = {
    "train": PREPARED_DATA_DIR / "train.csv",
    "val": PREPARED_DATA_DIR / "val.csv",
    "test": PREPARED_DATA_DIR / "test.csv",
    "manifest": PREPARED_DATA_DIR / "splits_manifest.json",
    "static_vocabulary": PREPARED_DATA_DIR / "static_covariate_vocabulary.json",
}


def require_files(paths):
    missing = [str(Path(path)) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing prepared-data files. Run the data-preparation notebook first "
            "and start this notebook from the same folder (or set "
            "CRESCENT_PREPARED_DATA_DIR):\n  - " + "\n  - ".join(missing)
        )


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()


require_files(PREPARED_PATHS.values())

with PREPARED_PATHS["manifest"].open(encoding="utf-8") as handle:
    PREPARED_MANIFEST = json.load(handle)
if not isinstance(PREPARED_MANIFEST, dict) or "files" not in PREPARED_MANIFEST:
    raise TypeError("splits_manifest.json is not a valid split manifest.")
if not str(PREPARED_MANIFEST.get("schema_version", "")).startswith("one_month_cap_context_splits"):
    raise RuntimeError(
        "splits_manifest.json was not written by the updated data-preparation notebook."
    )

# Hash check before any file is parsed: the model may only train on the exact
# files the preparation notebook wrote.
for _split in ("train", "val", "test"):
    _entry = PREPARED_MANIFEST["files"].get(_split, {})
    if _entry.get("file") != PREPARED_PATHS[_split].name:
        raise RuntimeError(f"Manifest lists a different file for {_split}: {_entry.get('file')}")
    _observed = sha256_file(PREPARED_PATHS[_split])
    if str(_entry.get("sha256")) != _observed:
        raise RuntimeError(
            f"{PREPARED_PATHS[_split]} does not match splits_manifest.json "
            f"(expected {_entry.get('sha256')}, observed {_observed}). "
            "Re-run the preparation notebook or restore the original file."
        )

INPUT_HASHES = {
    f"prepared_{_s}_csv_sha256": PREPARED_MANIFEST["files"][_s]["sha256"]
    for _s in ("train", "val", "test")
}
INPUT_HASHES["static_covariate_vocabulary_sha256"] = sha256_file(PREPARED_PATHS["static_vocabulary"])
INPUT_HASHES["splits_manifest_sha256"] = sha256_file(PREPARED_PATHS["manifest"])
_main_hash = PREPARED_MANIFEST.get("main_model_table_sha256") or hashlib.sha256(
    "|".join(INPUT_HASHES[f"prepared_{_s}_csv_sha256"] for _s in ("train", "val", "test")).encode()
).hexdigest()
# Names kept so the downstream provenance fields keep working unchanged.
INPUT_HASHES["near_index_key_sha256"] = _main_hash
INPUT_HASHES["fixed_index_key_sha256"] = _main_hash

PREPARED_DATA_SOURCE_INFO = {
    "prepared_data_dir": str(PREPARED_DATA_DIR),
    "split_files": {name: str(PREPARED_PATHS[name]) for name in ("train", "val", "test")},
    "splits_manifest": str(PREPARED_PATHS["manifest"]),
    "static_covariate_vocabulary": str(PREPARED_PATHS["static_vocabulary"]),
    "manifest_schema_version": str(PREPARED_MANIFEST.get("schema_version")),
    "hashes_verified_before_parsing": True,
}

BACKDATE_DAYS_BEFORE_INDEX = 1
PREPARATION_METADATA_MIN_INDEX_OFFSET_MONTHS = -1
PREPARATION_METADATA_MAX_INDEX_OFFSET_MONTHS = 1

ALLOW_EXISTING_RUN_DIR = os.getenv(
    "EHR_CRESCENT_ALLOW_EXISTING_RUN_DIR", "0"
).strip().lower() in {"1", "true", "yes"}
stale_patterns = (
    "*.pt",
    "canonical_*_samples.pkl",
    "training_history.csv",
    "test_prediction_arrays*.npz",
)
stale_run_artifacts = []
if RUN_DIR.exists():
    for pattern in stale_patterns:
        stale_run_artifacts.extend(sorted(RUN_DIR.glob(pattern)))
if stale_run_artifacts and not ALLOW_EXISTING_RUN_DIR:
    preview = "\n".join(f"  - {p}" for p in stale_run_artifacts[:10])
    raise RuntimeError(
        "EHR_CRESCENT_FINAL_OUTPUT already contains model/sample artifacts. Choose a "
        "new directory or explicitly set EHR_CRESCENT_ALLOW_EXISTING_RUN_DIR=1:\n"
        + preview
    )
RUN_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Reproducibility
# =============================================================================

def set_global_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


CLUSTER_CUDA_DEVICE_ID = 1


def choose_device(cuda_id: int = CLUSTER_CUDA_DEVICE_ID):
    """
    Lock execution to the requested scheduler-visible CUDA device.

    On a scheduled cluster job, CUDA_VISIBLE_DEVICES may map an allocated
    physical GPU to logical cuda:0. We deliberately respect that mapping and
    never fall back to another GPU or to CPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This model run is locked to cuda:0, but CUDA is unavailable. "
            "Start the notebook inside a GPU allocation and confirm that the "
            "cluster exposes at least one GPU to this process."
        )

    n_gpu = torch.cuda.device_count()
    cuda_id = int(cuda_id)
    if cuda_id != 0:
        raise ValueError("This notebook is intentionally locked to cuda_id=0.")
    if n_gpu <= cuda_id:
        raise RuntimeError(
            f"Requested cuda:{cuda_id}, but only {n_gpu} CUDA device(s) are visible."
        )

    torch.cuda.set_device(cuda_id)
    device = torch.device(f"cuda:{cuda_id}")
    if torch.cuda.current_device() != cuda_id:
        raise RuntimeError(
            f"PyTorch did not activate cuda:{cuda_id}; current device is "
            f"cuda:{torch.cuda.current_device()}."
        )
    return device


# =============================================================================
# Hazard binning
# =============================================================================

class HazardBinning:
    """
    Discrete time axis in months.

    Standard bins cover [0, MAX_HORIZON_MONTHS] in steps of BIN_WIDTH_MONTHS.
    If use_catch_all=False, events beyond MAX_HORIZON_MONTHS are treated as
    outside the modelled horizon.
    """

    def __init__(
        self,
        bin_width_months: int,
        max_horizon_months: int,
        use_catch_all: bool = False,
    ):
        self.BIN_WIDTH_MONTHS = int(bin_width_months)
        self.MAX_HORIZON_MONTHS = int(max_horizon_months)

        if self.BIN_WIDTH_MONTHS <= 0:
            raise ValueError("bin_width_months must be > 0.")

        if self.MAX_HORIZON_MONTHS <= 0:
            raise ValueError("max_horizon_months must be > 0.")

        if self.MAX_HORIZON_MONTHS % self.BIN_WIDTH_MONTHS != 0:
            raise ValueError("MAX_HORIZON_MONTHS must be divisible by BIN_WIDTH_MONTHS.")

        self.N_STANDARD_BINS = self.MAX_HORIZON_MONTHS // self.BIN_WIDTH_MONTHS
        self.USE_CATCH_ALL = bool(use_catch_all)
        self.N_BINS = self.N_STANDARD_BINS + (1 if self.USE_CATCH_ALL else 0)

        self.BIN_EDGES = np.arange(
            0,
            self.MAX_HORIZON_MONTHS + self.BIN_WIDTH_MONTHS,
            self.BIN_WIDTH_MONTHS,
            dtype=np.int32,
        )

        self.BIN_CENTERS = (
            np.arange(self.N_STANDARD_BINS, dtype=np.float32) * self.BIN_WIDTH_MONTHS
            + 0.5 * self.BIN_WIDTH_MONTHS
        )


haz = HazardBinning(
    bin_width_months=6,
    max_horizon_months=132,
    use_catch_all=False,
)


# =============================================================================
# Config
# =============================================================================

def window_start_bin(entry_delay_months):
    """Return the last hazard bin excluded by delayed entry."""
    entry = float(entry_delay_months)
    width = float(haz.BIN_WIDTH_MONTHS)
    if not np.isfinite(entry) or entry < 0 or entry > float(haz.MAX_HORIZON_MONTHS):
        raise ValueError("Entry delay must be finite and within the model time axis.")
    bins = entry / width
    if not np.isclose(bins, round(bins), rtol=0, atol=1e-8):
        raise ValueError("Entry delay must align with hazard-bin boundaries.")
    return int(round(bins)) - 1


@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Model identity
    # -------------------------------------------------------------------------
    # Six-month delayed-entry design uses the unchanged one-month-cap index as predictor cutoff.
    MODEL_VARIANT: str = (
        "CRESCENT_ONE_MONTH_CAP_6MONTH_DELAYED_ENTRY_CRP_3LAYER_TEMPORAL_INTERACTION_NLL_SHARED_STATE"
        "_CRP_PIECEWISE_LINEAR"
    )
    MAX_CALENDAR_MONTHS_FROM_RECRUITMENT: int = 1
    BACKDATE_DAYS_BEFORE_INDEX: int = int(BACKDATE_DAYS_BEFORE_INDEX)
    MAX_INDEX_OFFSET_MONTHS: int = int(
        PREPARATION_METADATA_MAX_INDEX_OFFSET_MONTHS
    )
    MIN_INDEX_OFFSET_MONTHS: int = int(
        PREPARATION_METADATA_MIN_INDEX_OFFSET_MONTHS
    )
    ALLOW_PRE_RECRUITMENT_INDEX: bool = True

    # -------------------------------------------------------------------------
    # Death-adjusted competing-risk design
    # -------------------------------------------------------------------------
    USE_DEATH_HEAD: bool = True

    # Each condition has its own condition/death/neither competing-risk process.
    # Event-type labels used later in per-condition CR targets:
    #   0 = censored / neither observed before censoring
    #   1 = incident condition k
    #   2 = death before condition k
    CR_TARGET_CENSOR: int = 0
    CR_TARGET_CONDITION: int = 1
    CR_TARGET_DEATH: int = 2

    # Canonical target provenance; targets are loaded rather than reconstructed.
    DEATH_TIE_POLICY: str = "condition_if_same_date_else_death"

    # Numerical safety for probability/log and rate calculations.
    PROB_EPS: float = 1e-7
    RATE_EPS: float = 1e-8

    # -------------------------------------------------------------------------
    # System
    # -------------------------------------------------------------------------
    DEVICE: torch.device = choose_device(cuda_id=0)
    NUM_WORKERS: int = 0


    # -------------------------------------------------------------------------
    # Data / sampling
    # -------------------------------------------------------------------------
    # One-day tolerance only for date-to-age floating-point conversion. The old
    # 0.25-year tolerance could leak roughly three months of future information.
    AGE_TOL_YEARS: float = 1.0 / 365.25

    SEED: int = int(os.getenv("CRESCENT_SEED", "42"))
    TRAINING_SEED: int = int(os.getenv("CRESCENT_SEED", "42"))

    # -------------------------------------------------------------------------
    # Fixed-landmark history representation
    # -------------------------------------------------------------------------
    # The Transformer uses up to 25 events, retaining CRP when present.
    FULL_SEQUENCE_MAX_EVENTS: int = 25
    MAX_LEN: int = FULL_SEQUENCE_MAX_EVENTS + 1  # recent event window + one CLS token

    # Shared representation widths.
    D_MODEL: int = 128
    D_FF: int = 512
    # 8 heads of width 16 (was 16 heads of width 8).
    N_HEADS: int = int(os.getenv("CRESCENT_N_HEADS", "16"))
    N_LAYERS: int = 3
    D_STATIC: int = 32
    DROPOUT_ENCODER: float = float(os.getenv("CRESCENT_DROPOUT", "0.20"))
    DROPOUT_HAZARD: float = float(os.getenv("CRESCENT_DROPOUT", "0.20"))

    # Shared condition-time and death decoders.
    HAZARD_HEAD_HIDDEN_DIM: int = 64
    HAZARD_HEAD_LAYERS: int = 2
    RESIDUAL_INIT_STD: float = 1e-3
    USE_SHARED_STATE_HAZARD_BRANCH: bool = True
    SHARED_STATE_HEAD_HIDDEN_DIM: int = 64
    SHARED_STATE_HEAD_LAYERS: int = 2
    SHARED_STATE_HEAD_ZERO_INIT: bool = True

    # Safe final log-rate range. Saturation fractions are exported and should
    # remain near zero.
    LOG_RATE_MIN: float = -18.0
    LOG_RATE_MAX: float = 2.0

    # Train-only global event/exposure estimates initialize the rate intercepts.
    CONDITION_BASELINE_FALLBACK_INITIAL_LOG_RATE: float = -8.5
    DEATH_BASELINE_FALLBACK_INITIAL_LOG_RATE: float = -8.5

    # A10 reader-temporal-kernel architecture switches; task/data construction remains locked.
    READER_TEMPORAL_BIAS_MODE: str = os.getenv(
        "A10_READER_TEMPORAL_BIAS_MODE", "interaction_mlp"
    ).strip().lower()
    READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER: int = 4
    READER_TEMPORAL_INTERACTION_DROPOUT: float = 0.10
    # Set to 0 for the architecture control run.
    USE_CONDITION_IDENTITY_PATH: bool = os.getenv(
        "A10_USE_CONDITION_IDENTITY_PATH", "1"
    ).strip().lower() in {"1", "true", "yes"}

    # -------------------------------------------------------------------------
    # Death head
    # -------------------------------------------------------------------------
    DEATH_HEAD_HIDDEN_DIM: int = 128
    DEATH_HEAD_LAYERS: int = 2
    DROPOUT_DEATH: float = float(os.getenv("CRESCENT_DROPOUT", "0.20"))

    # -------------------------------------------------------------------------
    # Event-value channel
    #
    # Canonical preparation supplies:
    #   token = CRP_MEASURED
    #   type  = crp
    #   value = a non-negative numeric CRP value
    # -------------------------------------------------------------------------
    USE_EVENT_VALUE_CHANNEL: bool = True

    # CRP remains a timed event in the model input. Its prepared
    # numeric value contributes only when the accompanying value mask is one.
    EXCLUDE_CRP_FROM_INPUTS: bool = False
    INCLUDE_CRP_AS_EVENT_VALUE: bool = True

    # -------------------------------------------------------------------------
    # Continuous static covariates
    #
    # Canonical preparation supplies the continuous covariates.
    # -------------------------------------------------------------------------
    # index age (standardised on train), Townsend z-score (already train-standardised,
    # training mean imputed when missing) and a Townsend-observed indicator.
    SAMPLE_CONTINUOUS_COLS_RAW: tuple = (
        "index_age_years",
        "townsend_z",
        "townsend_observed",
    )

    # -------------------------------------------------------------------------
    # Regularisation
    # -------------------------------------------------------------------------
    EVENT_MASK_PROB: float = float(os.getenv("CRESCENT_EVENT_MASK_PROB", "0.10"))
    COVARIATE_DROP_PROB: float = 0.0

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    BATCH_SIZE: int = int(os.getenv("CRESCENT_BATCH_SIZE", "256"))
    LR: float = float(os.getenv("CRESCENT_LR", "1.2e-4"))
    WEIGHT_DECAY: float = float(os.getenv("CRESCENT_WEIGHT_DECAY", "5e-2"))
    # Weight decay is skipped for 0-/1-D parameters (norm gains/biases, log-rate intercepts, gates).
    WEIGHT_DECAY_EXEMPT_1D: bool = os.getenv("CRESCENT_WD_EXEMPT_1D", "1").strip().lower() in {"1", "true", "yes"}
    # Exponential moving average of the weights, used for validation, checkpoint selection and test.
    # 0 disables it. Decay is warmed up as min(EMA_DECAY, (1+t)/(10+t)).
    EMA_DECAY: float = float(os.getenv("CRESCENT_EMA_DECAY", "0.999"))
    # Optional second-difference penalty on the time profile of the condition residual (0 = off;
    # any positive value adds an auxiliary term to the otherwise pure-NLL objective).
    TIME_SMOOTHNESS_WEIGHT: float = float(os.getenv("CRESCENT_TIME_SMOOTHNESS_WEIGHT", "0.0"))
    # Architecture switches added in this revision (defaults = revised behaviour).
    STATIC_CONT_INPUT_LAYERNORM: bool = os.getenv("CRESCENT_STATIC_CONT_LAYERNORM", "0").strip().lower() in {"1", "true", "yes"}
    USE_ATTAINED_AGE_TERM: bool = os.getenv("CRESCENT_ATTAINED_AGE_TERM", "1").strip().lower() in {"1", "true", "yes"}
    USE_READER_TEMPORAL_BIAS: bool = os.getenv("CRESCENT_READER_TEMPORAL_BIAS", "1").strip().lower() in {"1", "true", "yes"}
    MAX_ITERATIONS: int = int(os.getenv("A10_CANONICAL_MAX_ITERATIONS", "25000"))
    VALIDATION_INTERVAL: int = int(os.getenv("A10_CANONICAL_VALIDATION_INTERVAL", "1000"))
    LOG_INTERVAL: int = int(os.getenv("A10_CANONICAL_LOG_INTERVAL", "100"))
    WARMUP_ITERATIONS: int = int(os.getenv("A10_CANONICAL_WARMUP_ITERATIONS", "1000"))
    SCHEDULER_MIN_LR_RATIO: float = 0.05
    GRAD_CLIP: float = 1.0

    # Delphi-style fixed-budget training: zero disables patience stopping.
    PATIENCE_VALIDATIONS: int = int(os.getenv("A10_CANONICAL_PATIENCE_VALIDATIONS", "0"))
    MIN_ITERATIONS_BEFORE_EARLY_STOP: int = int(
        os.getenv("A10_CANONICAL_MIN_ITERATIONS_BEFORE_EARLY_STOP", "0")
    )
    MIN_DELTA: float = 1e-4

    # -------------------------------------------------------------------------
    # Joint from-scratch optimization
    # -------------------------------------------------------------------------
    TRAIN_FROM_SCRATCH: bool = True
    TRAIN_ALL_MODULES_JOINTLY: bool = True
    EXTERNAL_WARM_START_CHECKPOINT: str = ""

    # Every parameter is trainable throughout; there is no frozen or staged
    # optimization period. With PATIENCE_VALIDATIONS=0, training always reaches
    # MAX_ITERATIONS and the configured validation-selected checkpoint is restored.

    # -------------------------------------------------------------------------
    # Loss — exact piecewise-exponential cause-specific objective
    # -------------------------------------------------------------------------
    # Downstream training must integrate post-entry exposure through the first
    # condition or death event; the death term appears once per participant.
    # -------------------------------------------------------------------------
    LOSS_ALPHA_NLL: float = 1.0

    # False selects one global mean over all active condition-person pairs,
    # matching the comparator.
    NORMALISE_DISEASE_CR_NLL_BY_CONDITION: bool = False

    LOSS_GAMMA_DEATH: float = .10

    # Training loss is exactly 1.0 * disease NLL + 0.10 * death NLL (no auxiliary terms).


    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    EVAL_HORIZONS_MONTHS: tuple = (12, 24, 60, 120)
    EVAL_CALIBRATION_HORIZONS_MONTHS: tuple = (12, 24, 60, 120)

    # Horizons are measured after a six-month delayed-entry landmark.
    ENTRY_DELAY_MONTHS: int = 6
    ENTRY_TIME_EPS_MONTHS: float = 1e-6

    # Calibration uses Aalen-Johansen observed incidence.
    EVAL_N_CALIBRATION_BINS: int = 10
    EVAL_MIN_CALIBRATION_GROUP_N: int = 100
    EVAL_MIN_CALIBRATION_BIN_N: int = 50

    # Censoring-aware metric estimation.
    EVAL_IPCW_TRUNCATION_EPS: float = 1e-3
    EVAL_CENSOR_SURVIVAL_FLOOR: float = 0.01

    # Publication result workflow. Cell 7 creates locked point predictions;
    # Cell 7A creates the dense-grid point metrics; Cell 7B resamples test eids
    # while keeping the fitted model and train-derived censoring distributions frozen.
    EVAL_COMPUTE_WOLBERS_CINDEX: bool = True
    EVAL_CINDEX_TAU_MONTHS: int = 60
    EVAL_CI_POINT_HORIZONS_MONTHS: tuple = (24, 60)
    EVAL_INTEGRATION_START_MONTHS: int = 12
    EVAL_INTEGRATION_END_MONTHS: int = 120
    EVAL_BOOTSTRAP_N: int = 200
    EVAL_BOOTSTRAP_ALPHA: float = 0.05
    EVAL_BOOTSTRAP_SEED: int = 101
    EVAL_BOOTSTRAP_UNIT: str = "eid"
    EVAL_CI_MIN_VALID_BOOTSTRAP_FRACTION: float = 0.90
    EVAL_CI_MACRO_MIN_CONDITION_FRACTION: float = 0.90

    # -------------------------------------------------------------------------
    # Validation / checkpointing protocol
    # The primary (and only restored) checkpoint is the minimum validation joint
    # NLL, i.e. 1.0 * disease NLL + 0.10 * death NLL on the validation split.
    # -------------------------------------------------------------------------
    PRIMARY_CHECKPOINT: str = "best_model_by_val_loss.pt"



cfg = Config()

if str(cfg.DEVICE) != "cuda:0":
    raise RuntimeError(
        f"Device contract violation: expected cuda:0, resolved {cfg.DEVICE}."
    )

CLUSTER_GPU_INFO = {
    "device": str(cfg.DEVICE),
    "logical_device_id": 0,
    "device_name": torch.cuda.get_device_name(0),
    "visible_cuda_device_count": int(torch.cuda.device_count()),
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    "scheduler_semantics": (
        "cuda:0 is the first GPU exposed to this process by the cluster scheduler"
    ),
    "cpu_fallback_allowed": False,
}


def _signed_month_label(months):
    months = int(months)
    if months < 0:
        return f"minus_{abs(months)}"
    if months > 0:
        return f"plus_{months}"
    return "zero"


def expected_near_index_source(cfg_obj):
    return "hybrid_one_month_cap_toward_recruitment"


if int(cfg.ENTRY_DELAY_MONTHS) != 6:
    raise ValueError("This model requires a six-month delayed-entry landmark.")
if window_start_bin(cfg.ENTRY_DELAY_MONTHS) != 0:
    raise ValueError("The six-month entry delay must exclude exactly the first hazard bin.")
if int(cfg.ENTRY_DELAY_MONTHS) + max(map(int, cfg.EVAL_HORIZONS_MONTHS)) > int(haz.MAX_HORIZON_MONTHS):
    raise ValueError("The largest post-entry endpoint exceeds the hazard time axis.")

EXPECTED_NEAR_INDEX_SOURCE = expected_near_index_source(cfg)
EXPECTED_NEAR_INDEX_WINDOW_DESCRIPTION = (
    "original hybrid-index date capped to ±1 calendar month around recruitment"
)

if int(cfg.MIN_INDEX_OFFSET_MONTHS) != -1 or int(cfg.MAX_INDEX_OFFSET_MONTHS) != 1:
    raise ValueError("The one-month-cap hybrid model uses descriptive bounds of -1 to +1 month.")
if int(cfg.MAX_CALENDAR_MONTHS_FROM_RECRUITMENT) != 1:
    raise ValueError("The index must use a one-calendar-month cap.")
if int(cfg.BACKDATE_DAYS_BEFORE_INDEX) != 1:
    raise ValueError("Post-index dynamic covariates must be placed one day before the index.")
if not bool(cfg.ALLOW_PRE_RECRUITMENT_INDEX):
    raise ValueError("The one-month-cap hybrid index must allow pre-recruitment landmarks.")

if int(cfg.EVAL_MIN_CALIBRATION_GROUP_N) < 1:
    raise ValueError("EVAL_MIN_CALIBRATION_GROUP_N must be positive.")
if int(cfg.EVAL_MIN_CALIBRATION_BIN_N) < 1:
    raise ValueError("EVAL_MIN_CALIBRATION_BIN_N must be positive.")
if int(cfg.EVAL_N_CALIBRATION_BINS) < 1:
    raise ValueError("EVAL_N_CALIBRATION_BINS must be positive.")
if 60 not in tuple(int(x) for x in cfg.EVAL_CALIBRATION_HORIZONS_MONTHS):
    raise ValueError("The locked calibration horizons must include 60 months.")
if tuple(int(x) for x in cfg.EVAL_CALIBRATION_HORIZONS_MONTHS) != tuple(
    int(x) for x in cfg.EVAL_HORIZONS_MONTHS
):
    raise ValueError(
        "Calibration horizons must exactly match the saved evaluation horizons."
    )
if not 0.0 < float(cfg.EVAL_CENSOR_SURVIVAL_FLOOR) <= 1.0:
    raise ValueError("EVAL_CENSOR_SURVIVAL_FLOOR must lie in (0, 1].")
if not bool(cfg.EVAL_COMPUTE_WOLBERS_CINDEX):
    raise ValueError("V11 requires the Wolbers competing-risk C-index workflow.")
if int(cfg.EVAL_CINDEX_TAU_MONTHS) not in range(1, int(haz.MAX_HORIZON_MONTHS) + 1):
    raise ValueError("EVAL_CINDEX_TAU_MONTHS lies outside the model horizon.")
if tuple(cfg.EVAL_CI_POINT_HORIZONS_MONTHS) != (24, 60):
    raise ValueError("V11 publication point-horizon CIs must use (24, 60).")
if int(cfg.EVAL_INTEGRATION_START_MONTHS) != 12 or int(cfg.EVAL_INTEGRATION_END_MONTHS) != 120:
    raise ValueError("V11 requires the canonical 12–120 month dense integration window.")
if int(cfg.EVAL_BOOTSTRAP_N) < 1:
    raise ValueError("EVAL_BOOTSTRAP_N must be positive.")
if not 0.0 < float(cfg.EVAL_BOOTSTRAP_ALPHA) < 1.0:
    raise ValueError("EVAL_BOOTSTRAP_ALPHA must lie in (0, 1).")
if str(cfg.EVAL_BOOTSTRAP_UNIT) != "eid":
    raise ValueError("V11 bootstrap unit must be eid.")
if int(cfg.D_MODEL) % int(cfg.N_HEADS) != 0:
    raise ValueError("D_MODEL must be divisible by N_HEADS.")
if str(cfg.READER_TEMPORAL_BIAS_MODE) not in {"additive", "interaction_mlp"}:
    raise ValueError(
        "READER_TEMPORAL_BIAS_MODE must be 'additive' or 'interaction_mlp'."
    )
if int(cfg.READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER) < 1:
    raise ValueError("READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER must be positive.")
if not 0.0 <= float(cfg.READER_TEMPORAL_INTERACTION_DROPOUT) < 1.0:
    raise ValueError("READER_TEMPORAL_INTERACTION_DROPOUT must lie in [0, 1).")
if float(cfg.LOSS_ALPHA_NLL) != 1.0 or float(cfg.LOSS_GAMMA_DEATH) != 0.10:
    raise ValueError("Training loss must be 1.0*disease_nll + 0.10*death_nll.")
if int(cfg.FULL_SEQUENCE_MAX_EVENTS) < 1:
    raise ValueError("FULL_SEQUENCE_MAX_EVENTS must be positive.")
if int(cfg.MAX_LEN) != int(cfg.FULL_SEQUENCE_MAX_EVENTS) + 1:
    raise ValueError("MAX_LEN must equal FULL_SEQUENCE_MAX_EVENTS + 1.")
if int(cfg.MAX_ITERATIONS) < 1:
    raise ValueError("MAX_ITERATIONS must be positive.")
if int(cfg.VALIDATION_INTERVAL) < 1:
    raise ValueError("VALIDATION_INTERVAL must be positive.")
if int(cfg.LOG_INTERVAL) < 1:
    raise ValueError("LOG_INTERVAL must be positive.")
if not 0 <= int(cfg.WARMUP_ITERATIONS) < int(cfg.MAX_ITERATIONS):
    raise ValueError("WARMUP_ITERATIONS must lie in [0, MAX_ITERATIONS).")
if int(cfg.PATIENCE_VALIDATIONS) < 0:
    raise ValueError("PATIENCE_VALIDATIONS must be non-negative.")
if int(cfg.MIN_ITERATIONS_BEFORE_EARLY_STOP) < 0:
    raise ValueError("MIN_ITERATIONS_BEFORE_EARLY_STOP must be non-negative.")
if not 0.0 < float(cfg.SCHEDULER_MIN_LR_RATIO) <= 1.0:
    raise ValueError("SCHEDULER_MIN_LR_RATIO must lie in (0, 1].")
if float(cfg.LOG_RATE_MIN) >= float(cfg.LOG_RATE_MAX):
    raise ValueError("LOG_RATE_MIN must be smaller than LOG_RATE_MAX.")
if not bool(cfg.TRAIN_FROM_SCRATCH):
    raise ValueError("V11 requires TRAIN_FROM_SCRATCH=True.")
if not bool(cfg.TRAIN_ALL_MODULES_JOINTLY):
    raise ValueError("V11 requires TRAIN_ALL_MODULES_JOINTLY=True.")
if str(cfg.EXTERNAL_WARM_START_CHECKPOINT).strip():
    raise ValueError("V11 does not accept an external warm-start checkpoint.")
if Path(str(cfg.PRIMARY_CHECKPOINT)).name != "best_model_by_val_loss.pt":
    raise ValueError("PRIMARY_CHECKPOINT must be best_model_by_val_loss.pt (validation NLL).")

set_global_seed(cfg.SEED)


# The hazard time axis is checked against the built target metadata in the loader cell.


ENTRY_TIME_EPS_MONTHS = float(cfg.ENTRY_TIME_EPS_MONTHS)
DELAYED_ENTRY_SCHEMA_VERSION = "crescent.delayed_entry.v1"


def delayed_entry_contract():
    """Serializable task identity shared by config, checkpoints, and exports."""
    return {
        "schema_version": DELAYED_ENTRY_SCHEMA_VERSION,
        "entry_delay_months": float(cfg.ENTRY_DELAY_MONTHS),
        "entry_time_eps_months": ENTRY_TIME_EPS_MONTHS,
        "entry_rule": "condition-free, alive, and observed strictly beyond entry",
        "events_on_or_before_entry": "excluded from the condition-specific risk set",
        "death_entry_rule": "alive and observed strictly beyond entry",
        "training_likelihood": "exact delayed-entry likelihood over post-entry exposure only",
        "predictor_cutoff": "original one-month-cap hybrid index; no entry-delay predictors added",
        "reported_horizons": "months after the delayed-entry landmark",
        "horizons_months_after_entry": list(map(int, cfg.EVAL_HORIZONS_MONTHS)),
        "maximum_report_endpoint_months_from_index": int(
            cfg.ENTRY_DELAY_MONTHS + max(cfg.EVAL_HORIZONS_MONTHS)
        ),
        "implementation_status": "implemented in initialization, likelihood, and evaluation",
    }


def make_delayed_entry_outcomes_numpy(
    at_risk, event_type, event_time, censor_time, entry_delay_months=None,
):
    """Use canonical first-event labels; express outcomes relative to entry."""
    entry = float(cfg.ENTRY_DELAY_MONTHS if entry_delay_months is None else entry_delay_months)
    active = np.asarray(at_risk, dtype=float) > 0.5
    types = np.asarray(event_type, dtype=np.int8)
    times = np.asarray(event_time, dtype=np.float64)
    censor = np.asarray(censor_time, dtype=np.float64)
    if active.ndim != 2 or types.shape != active.shape or times.shape != active.shape:
        raise ValueError("Delayed-entry condition arrays must have aligned (N, K) shapes.")
    if censor.shape != (active.shape[0],):
        raise ValueError("Delayed-entry censor times must have shape (N,).")
    pair_event = np.isfinite(times) & np.isin(types, [int(cfg.CR_TARGET_CONDITION), int(cfg.CR_TARGET_DEATH)])
    before_entry = active & pair_event & (times <= entry + ENTRY_TIME_EPS_MONTHS)
    eligible = active & np.isfinite(censor[:, None]) & (censor[:, None] > entry + ENTRY_TIME_EPS_MONTHS) & (~before_entry)
    observed_event = eligible & pair_event & (times > entry + ENTRY_TIME_EPS_MONTHS)
    status = np.where(observed_event, types, 0).astype(np.int8)
    exit_time = np.broadcast_to(np.maximum(censor - entry, 0.0)[:, None], active.shape).copy()
    exit_time[observed_event] = times[observed_event] - entry
    if np.any(eligible & (~np.isfinite(exit_time))):
        raise ValueError("Non-finite eligible delayed-entry outcome.")
    return {
        "eligible": eligible, "status": status,
        "exit_time": exit_time, "exit_time_untruncated": exit_time.copy(),
    }


def make_delayed_entry_death_outcomes_numpy(censor_time, death_observed, death_time, entry_delay_months=None):
    entry = float(cfg.ENTRY_DELAY_MONTHS if entry_delay_months is None else entry_delay_months)
    censor = np.asarray(censor_time, dtype=np.float64)
    times = np.asarray(death_time, dtype=np.float64)
    observed = (np.asarray(death_observed, dtype=float) > 0.5) & np.isfinite(times)
    before_entry = observed & (times <= entry + ENTRY_TIME_EPS_MONTHS)
    eligible = np.isfinite(censor) & (censor > entry + ENTRY_TIME_EPS_MONTHS) & (~before_entry)
    event = eligible & observed & (times > entry + ENTRY_TIME_EPS_MONTHS)
    exit_time = np.maximum(censor - entry, 0.0)
    exit_time[event] = times[event] - entry
    return {"eligible": eligible, "status": event.astype(np.int8), "exit_time": exit_time}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, tuple):
        return [_json_safe(x) for x in value]
    if isinstance(value, list):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


cfg_dict = _json_safe(asdict(cfg))
full_config = {
    "model_variant": str(cfg.MODEL_VARIANT),
    "config": cfg_dict,
    "prepared_data_source": PREPARED_DATA_SOURCE_INFO,
    "fixed_index_contract": {
        "design": "one-calendar-month-cap hybrid index (+/-1 month around recruitment)",
        "calendar_months_from_recruitment": 1,
        "indices_sampled": False,
        "indices_shifted_in_model": False,
    },
    "hazard_binning": {
        "BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS),
        "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS),
        "N_BINS": int(haz.N_BINS),
    },
    "delayed_entry_contract": delayed_entry_contract(),
    "cluster_gpu": CLUSTER_GPU_INFO,
}
config_path = RUN_DIR / "config.json"
with config_path.open("w", encoding="utf-8") as handle:
    json.dump(full_config, handle, indent=2, default=str)

print(json.dumps({
    "prepared_data_dir": str(PREPARED_DATA_DIR),
    "crescent_output_dir": str(RUN_DIR),
    "prepared_train_path": str(PREPARED_PATHS["train"]),
    "prepared_validation_path": str(PREPARED_PATHS["val"]),
    "prepared_test_path": str(PREPARED_PATHS["test"]),
    "entry_delay_months": int(cfg.ENTRY_DELAY_MONTHS),
    "maximum_report_endpoint_months_from_index": int(cfg.ENTRY_DELAY_MONTHS + max(cfg.EVAL_HORIZONS_MONTHS)),
    "device": str(cfg.DEVICE),
}, indent=2))
