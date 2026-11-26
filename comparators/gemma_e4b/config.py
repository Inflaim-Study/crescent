# Gemma 4 E4B (frozen encoder) — Configuration, parity checks against CRESCENT, and lock-before-test protocol
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 2 ---
# CELL 1 — GEMMA 4 E4B COMPARATOR (V7) — CONFIGURATION AND PARITY CONTRACT
# Frozen / LoRA-adapted Gemma 4 E4B reading each participant's history as text, with a trained survival head, on the CRESCENT
# six-month delayed-entry task. Same data, labels, likelihood, recipe, evaluation code and output schema as the revised CRESCENT notebook.
# Notebook role: B   (EXTRACT = text + hidden-state cache; A = embedding + simple head; B = Gemma encoder + CRESCENT decoder; C = LoRA)
# This notebook generates no text; it is not a test of Gemma answering clinical questions.
# Gemma is distributed under Apache 2.0 (see the notice cell). Cells copied from the CRESCENT notebook are marked verbatim.

import sys
import os
# ---- GPU choice (V7 two-GPU option) ----------------------------------------------------------------------------------------
# This notebook trains a small decoder on cached states, so a single run cannot use two GPUs usefully. To use both GPUs, run two
# seeds at the same time: one kernel/job with GEMMA_GPU=0 and another with GEMMA_GPU=1 (run_gemma_seeds_2gpu.py does this).
# GEMMA_GPU indexes the GPUs visible to this job; the chosen GPU then appears as logical cuda:0, so the device lock below is unchanged.
# It must take effect before CUDA starts, i.e. in a fresh kernel. Leave it unset for the original single-GPU behaviour.
GEMMA_GPU = os.getenv("GEMMA_GPU", "").strip()
if GEMMA_GPU and os.environ.get("_GEMMA_GPU_APPLIED") != GEMMA_GPU:
    if not GEMMA_GPU.isdigit():
        raise ValueError("GEMMA_GPU must be one GPU index, e.g. 0 or 1.")
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
        raise RuntimeError("GEMMA_GPU must be set in a fresh kernel, before CUDA is initialised.")
    _visible = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if _visible and int(GEMMA_GPU) >= len(_visible):
        raise ValueError(f"GEMMA_GPU={GEMMA_GPU}, but this job only sees CUDA_VISIBLE_DEVICES={','.join(_visible)}.")
    os.environ["CUDA_VISIBLE_DEVICES"] = _visible[int(GEMMA_GPU)] if _visible else GEMMA_GPU
    os.environ["_GEMMA_GPU_APPLIED"] = GEMMA_GPU

from dataclasses import asdict, dataclass
from pathlib import Path
import contextlib
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
GEMMA_STUDY_VERSION = "V7"
ARM = "B"
CRESCENT_NOTEBOOK_SHA256 = "594e801cdf58f80e6ab8378744980a34f301ca706cf559313dd5f3f98d6a4e82"       # revised CRESCENT notebook this file was built from
TEXT_CONTRACT_VERSION = "gemma.canonical_text.v7"
SELECTION_LOCK_SCHEMA = "gemma.selection_lock.v7"

# all (default): train this seed (or reuse it if already trained), lock it on validation results only, then evaluate on test.
# train / select / evaluate run one stage at a time (needed for several seeds or a validation grid).
PHASE = os.getenv("GEMMA_PHASE", "all").strip().lower()     # all | train | select | evaluate
RUN_ALL = PHASE == "all"
if RUN_ALL:
    PHASE = "train"                                           # becomes "evaluate" below if this seed is already trained
EXTRACT_SPLITS = [s.strip() for s in os.getenv("GEMMA_EXTRACT_SPLITS", "train,val").split(",") if s.strip()]    # extraction notebook only
if ARM == "EXTRACT":
    if not set(EXTRACT_SPLITS) <= {"train", "val", "test"} or not EXTRACT_SPLITS:
        raise ValueError("GEMMA_EXTRACT_SPLITS must be a comma list drawn from train, val, test.")
    PHASE = "evaluate" if "test" in EXTRACT_SPLITS else "train"     # the test pickle is parsed only when test extraction is requested (after the lock)
if PHASE not in {"train", "evaluate", "select"}:
    raise ValueError("GEMMA_PHASE must be train, select or evaluate.")
SEED = int(os.getenv("GEMMA_SEED", "42"))
if RUN_ALL:                                                   # a single-run study locks only this arm and this seed
    os.environ.setdefault("GEMMA_REQUIRED_SEEDS", str(SEED))
    os.environ.setdefault("GEMMA_REQUIRED_ARMS", ARM)
HPID = os.getenv("GEMMA_HPID", "final").strip()
TEST_ONLY_CPU = os.getenv("GEMMA_TEST_ONLY_CPU", "0").strip().lower() in {"1", "true", "yes"}     # unit tests only


def _dir(variable, default=None):
    value = os.getenv(variable, "").strip()
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        raise EnvironmentError(f"Set {variable}.")
    return Path(default).expanduser().resolve()


# Everything defaults to <working directory>/runs/gemma_e4b/...  (override with GEMMA_OUTPUT_ROOT)
OUTPUT_ROOT = _dir("GEMMA_OUTPUT_ROOT", Path.cwd() / "runs" / "gemma_e4b")
CRESCENT_RUN_DIR = _dir("GEMMA_CRESCENT_RUN_DIR", "runs/CRESCENT_Final")
CACHE_DIR = _dir("GEMMA_CACHE_DIR", OUTPUT_ROOT / "GEMMA4_E4B_cache_V7")
COMPARISON_DIR = _dir("GEMMA_COMPARISON_DIR", OUTPUT_ROOT / "GEMMA4_E4B_V7_comparison")
if ARM == "EXTRACT":
    RUN_DIR = CACHE_DIR
else:
    RUN_DIR = _dir("GEMMA_RUN_DIR", OUTPUT_ROOT / (f"GEMMA4_E4B_{ARM}_V7_" + (f"hp{HPID}_seed{SEED}" if HPID != "final" else f"seed{SEED}")))
RESUME = os.getenv("GEMMA_RESUME", "0").strip().lower() in {"1", "true", "yes"}
ALLOW_EXISTING_RUN_DIR = os.getenv("GEMMA_ALLOW_EXISTING_RUN_DIR", "0").strip().lower() in {"1", "true", "yes"}
for _other in (CRESCENT_RUN_DIR,):
    if RUN_DIR == _other or _other in RUN_DIR.parents or RUN_DIR in _other.parents:
        raise ValueError("The Gemma run/cache directory must be separate from the CRESCENT run directory.")
if RUN_ALL:
    if not (CACHE_DIR / "test_cache_meta.json").is_file():
        raise FileNotFoundError(f"GEMMA_PHASE=all evaluates on the test set at the end, but {CACHE_DIR / 'test_cache_meta.json'} is missing. "
                                "Run notebook 01 with GEMMA_EXTRACT_SPLITS=train,val,test first (or use GEMMA_PHASE=train).")
    _summary = RUN_DIR / "cell6_training_summary_death_adjusted.json"
    if _summary.is_file() and json.loads(_summary.read_text()).get("training_complete") is True and (RUN_DIR / "run_selection_locked.json").is_file():
        PHASE = "evaluate"                                    # already trained: reuse the completed run
    print(f"GEMMA_PHASE=all: {'training, then' if PHASE == 'train' else 'reusing the completed training run;'} validation-only lock, then test evaluation.")
if ARM != "EXTRACT" and PHASE == "train":
    _existing = sorted(RUN_DIR.glob("*.pt")) if RUN_DIR.exists() else []
    if _existing and not (RESUME or ALLOW_EXISTING_RUN_DIR):
        raise RuntimeError("GEMMA_RUN_DIR already holds checkpoints. Use a fresh directory (change GEMMA_HPID or GEMMA_SEED), or set GEMMA_RESUME=1:\n"
                           + "\n".join(f"  - {p}" for p in _existing[:10]))
RUN_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
LOCK_PATH = COMPARISON_DIR / "selection_locked.json"


# ---- encoder identity (pinned and verified in the extraction notebook; the arms read it back from the cache) ----------------
ENCODER_DIR = Path(os.getenv("GEMMA_ENCODER_DIR", "")).expanduser().resolve() if os.getenv("GEMMA_ENCODER_DIR", "").strip() else None
ENCODER_REPO = os.getenv("GEMMA_ENCODER_REPO", "google/gemma-4-E4B")
ENCODER_REVISION = os.getenv("GEMMA_ENCODER_REVISION", "").strip()
ENCODER_REFERENCE_HASHES = Path(os.getenv("GEMMA_ENCODER_REFERENCE_HASHES", "")).expanduser().resolve() if os.getenv("GEMMA_ENCODER_REFERENCE_HASHES", "").strip() else None
LAYER_CHOICE = os.getenv("GEMMA_LAYER", "final").strip().lower()                 # final | twothirds
if LAYER_CHOICE not in {"final", "twothirds"}:
    raise ValueError("GEMMA_LAYER must be 'final' or 'twothirds'.")
POOLING = os.getenv("GEMMA_POOLING", "index_end").strip().lower()                # Arm A: index_end | mean_pool
if POOLING not in {"index_end", "mean_pool"}:
    raise ValueError("GEMMA_POOLING must be 'index_end' or 'mean_pool'.")
HEAD_KIND = os.getenv("GEMMA_HEAD", "mlp").strip().lower()                       # Arm A: linear | mlp
if HEAD_KIND not in {"linear", "mlp"}:
    raise ValueError("GEMMA_HEAD must be 'linear' or 'mlp'.")
HEAD_DROPOUT = float(os.getenv("GEMMA_HEAD_DROPOUT", "0.2"))
MAX_DOCUMENT_TOKENS = int(os.getenv("GEMMA_MAX_DOCUMENT_TOKENS", "2048"))
PARENT_ARM = ARM if ARM in {"A", "B"} else os.getenv("GEMMA_PARENT_ARM", "B").strip().upper()       # Arm C trains the architecture of its parent arm end to end
if PARENT_ARM not in {"A", "B"}:
    raise ValueError("GEMMA_PARENT_ARM must be A or B.")
LORA_RANK = int(os.getenv("GEMMA_LORA_RANK", "16"))
LORA_ALPHA = int(os.getenv("GEMMA_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("GEMMA_LORA_DROPOUT", "0.05"))
ADAPTER_LR = float(os.getenv("GEMMA_ADAPTER_LR", "1e-4"))
EFFECTIVE_BATCH = int(os.getenv("GEMMA_EFFECTIVE_BATCH", "64"))
MAX_EPOCHS_C = int(os.getenv("GEMMA_MAX_EPOCHS", "10"))
EVAL_BATCH_C = int(os.getenv("GEMMA_EVAL_BATCH", "4"))

CLUSTER_CUDA_DEVICE_ID = 0
def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(int(chunk_size)), b""):
            digest.update(block)
    return digest.hexdigest()


def set_global_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def choose_device(cuda_id: int = CLUSTER_CUDA_DEVICE_ID):
    """
    Lock execution to the requested scheduler-visible CUDA device.

    In a multi-GPU or scheduled environment, CUDA_VISIBLE_DEVICES may map an allocated
    physical GPU to logical cuda:0. We deliberately respect that mapping and
    never fall back to another GPU or to CPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "This model run is locked to cuda:0, but CUDA is unavailable. "
            "Start the notebook inside a GPU allocation and confirm that the "
            "environment exposes at least one GPU to this process."
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
if TEST_ONLY_CPU:
    def choose_device(cuda_id=0):          # unit-test override; never use for a reported run
        return torch.device("cpu")
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
haz = HazardBinning(
    bin_width_months=6,
    max_horizon_months=132,
    use_catch_all=False,
)
BACKDATE_DAYS_BEFORE_INDEX = 1
PREPARATION_METADATA_MIN_INDEX_OFFSET_MONTHS = -1
PREPARATION_METADATA_MAX_INDEX_OFFSET_MONTHS = 1

# ---- the revised CRESCENT Config, verbatim. Its environment variables (CRESCENT_LR, CRESCENT_DROPOUT, ...) keep CRESCENT's recipe as
# ---- the default (batch 256, 25,000 updates, warm-up + cosine, EMA 0.999, weight decay 0.05). Pre-registered grid factors change only LR and dropout.
@dataclass
class Config:
    # -------------------------------------------------------------------------
    # Model identity
    # -------------------------------------------------------------------------
    # Six-month delayed-entry design uses the unchanged one-month-cap index as predictor cutoff.
    MODEL_VARIANT: str = (
        "CRESCENT_ONE_MONTH_CAP_6MONTH_DELAYED_ENTRY_CRP_3LAYER_TEMPORAL_INTERACTION_NLL_SHARED_STATE"
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
    N_HEADS: int = int(os.getenv("CRESCENT_N_HEADS", "8"))
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

set_global_seed(SEED)
cfg = Config()
cfg.MODEL_VARIANT = ("Gemma4_E4B_" + {"A": "Frozen_EmbeddingHead", "B": "Frozen_CrescentDecoder", "C": "LoRA", "EXTRACT": "Extraction"}[ARM]
                     + "_OneMonthCap_SixMonth_DelayedEntry_V7")
cfg.SEED = SEED
cfg.TRAINING_SEED = SEED
PIPELINE_CODE_SHA256 = "e0ace03920887275b605e0cacd5015f0fa314da63d7f66f937d20f45931c9279"
if ARM == "C":
    cfg.BATCH_SIZE = int(os.getenv("GEMMA_MICRO_BATCH", "2"))             # micro-batch; effective batch 64 by accumulation (specification 6.3)

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

# ---- parity checks against the completed CRESCENT run ---------------------------------------------------------------------
CRESCENT_CONFIG_PATH = CRESCENT_RUN_DIR / "config.json"
if not CRESCENT_CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Missing CRESCENT config.json: {CRESCENT_CONFIG_PATH}")
CRESCENT_CONFIG = json.loads(CRESCENT_CONFIG_PATH.read_text(encoding="utf-8"))
_theirs_all = CRESCENT_CONFIG["config"]

_field_environments = {'MODEL_VARIANT': [], 'MAX_CALENDAR_MONTHS_FROM_RECRUITMENT': [], 'BACKDATE_DAYS_BEFORE_INDEX': [], 'MAX_INDEX_OFFSET_MONTHS': [], 'MIN_INDEX_OFFSET_MONTHS': [], 'ALLOW_PRE_RECRUITMENT_INDEX': [], 'USE_DEATH_HEAD': [], 'CR_TARGET_CENSOR': [], 'CR_TARGET_CONDITION': [], 'CR_TARGET_DEATH': [], 'DEATH_TIE_POLICY': [], 'PROB_EPS': [], 'RATE_EPS': [], 'DEVICE': [], 'NUM_WORKERS': [], 'AGE_TOL_YEARS': [], 'SEED': ['CRESCENT_SEED'], 'TRAINING_SEED': ['CRESCENT_SEED'], 'FULL_SEQUENCE_MAX_EVENTS': [], 'MAX_LEN': [], 'D_MODEL': [], 'D_FF': [], 'N_HEADS': ['CRESCENT_N_HEADS'], 'N_LAYERS': [], 'D_STATIC': [], 'DROPOUT_ENCODER': ['CRESCENT_DROPOUT'], 'DROPOUT_HAZARD': ['CRESCENT_DROPOUT'], 'HAZARD_HEAD_HIDDEN_DIM': [], 'HAZARD_HEAD_LAYERS': [], 'RESIDUAL_INIT_STD': [], 'USE_SHARED_STATE_HAZARD_BRANCH': [], 'SHARED_STATE_HEAD_HIDDEN_DIM': [], 'SHARED_STATE_HEAD_LAYERS': [], 'SHARED_STATE_HEAD_ZERO_INIT': [], 'LOG_RATE_MIN': [], 'LOG_RATE_MAX': [], 'CONDITION_BASELINE_FALLBACK_INITIAL_LOG_RATE': [], 'DEATH_BASELINE_FALLBACK_INITIAL_LOG_RATE': [], 'READER_TEMPORAL_BIAS_MODE': ['A10_READER_TEMPORAL_BIAS_MODE'], 'READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER': [], 'READER_TEMPORAL_INTERACTION_DROPOUT': [], 'USE_CONDITION_IDENTITY_PATH': ['A10_USE_CONDITION_IDENTITY_PATH'], 'DEATH_HEAD_HIDDEN_DIM': [], 'DEATH_HEAD_LAYERS': [], 'DROPOUT_DEATH': ['CRESCENT_DROPOUT'], 'USE_EVENT_VALUE_CHANNEL': [], 'EXCLUDE_CRP_FROM_INPUTS': [], 'INCLUDE_CRP_AS_EVENT_VALUE': [], 'SAMPLE_CONTINUOUS_COLS_RAW': [], 'EVENT_MASK_PROB': ['CRESCENT_EVENT_MASK_PROB'], 'COVARIATE_DROP_PROB': [], 'BATCH_SIZE': ['CRESCENT_BATCH_SIZE'], 'LR': ['CRESCENT_LR'], 'WEIGHT_DECAY': ['CRESCENT_WEIGHT_DECAY'], 'WEIGHT_DECAY_EXEMPT_1D': ['CRESCENT_WD_EXEMPT_1D'], 'EMA_DECAY': ['CRESCENT_EMA_DECAY'], 'TIME_SMOOTHNESS_WEIGHT': ['CRESCENT_TIME_SMOOTHNESS_WEIGHT'], 'STATIC_CONT_INPUT_LAYERNORM': ['CRESCENT_STATIC_CONT_LAYERNORM'], 'USE_ATTAINED_AGE_TERM': ['CRESCENT_ATTAINED_AGE_TERM'], 'USE_READER_TEMPORAL_BIAS': ['CRESCENT_READER_TEMPORAL_BIAS'], 'MAX_ITERATIONS': ['A10_CANONICAL_MAX_ITERATIONS'], 'VALIDATION_INTERVAL': ['A10_CANONICAL_VALIDATION_INTERVAL'], 'LOG_INTERVAL': ['A10_CANONICAL_LOG_INTERVAL'], 'WARMUP_ITERATIONS': ['A10_CANONICAL_WARMUP_ITERATIONS'], 'SCHEDULER_MIN_LR_RATIO': [], 'GRAD_CLIP': [], 'PATIENCE_VALIDATIONS': ['A10_CANONICAL_PATIENCE_VALIDATIONS'], 'MIN_ITERATIONS_BEFORE_EARLY_STOP': ['A10_CANONICAL_MIN_ITERATIONS_BEFORE_EARLY_STOP'], 'MIN_DELTA': [], 'TRAIN_FROM_SCRATCH': [], 'TRAIN_ALL_MODULES_JOINTLY': [], 'EXTERNAL_WARM_START_CHECKPOINT': [], 'LOSS_ALPHA_NLL': [], 'NORMALISE_DISEASE_CR_NLL_BY_CONDITION': [], 'LOSS_GAMMA_DEATH': [], 'EVAL_HORIZONS_MONTHS': [], 'EVAL_CALIBRATION_HORIZONS_MONTHS': [], 'ENTRY_DELAY_MONTHS': [], 'ENTRY_TIME_EPS_MONTHS': [], 'EVAL_N_CALIBRATION_BINS': [], 'EVAL_MIN_CALIBRATION_GROUP_N': [], 'EVAL_MIN_CALIBRATION_BIN_N': [], 'EVAL_IPCW_TRUNCATION_EPS': [], 'EVAL_CENSOR_SURVIVAL_FLOOR': [], 'EVAL_COMPUTE_WOLBERS_CINDEX': [], 'EVAL_CINDEX_TAU_MONTHS': [], 'EVAL_CI_POINT_HORIZONS_MONTHS': [], 'EVAL_INTEGRATION_START_MONTHS': [], 'EVAL_INTEGRATION_END_MONTHS': [], 'EVAL_BOOTSTRAP_N': [], 'EVAL_BOOTSTRAP_ALPHA': [], 'EVAL_BOOTSTRAP_SEED': [], 'EVAL_BOOTSTRAP_UNIT': [], 'EVAL_CI_MIN_VALID_BOOTSTRAP_FRACTION': [], 'EVAL_CI_MACRO_MIN_CONDITION_FRACTION': [], 'PRIMARY_CHECKPOINT': []}
for _key, _value in _theirs_all.items():
    if hasattr(cfg,_key) and _key not in {'DEVICE','MODEL_VARIANT','SEED','TRAINING_SEED'} and not any(os.getenv(e) is not None for e in _field_environments.get(_key,[])):
        setattr(cfg,_key,tuple(_value) if isinstance(getattr(cfg,_key),tuple) else _value)
cfg.EVENT_MASK_PROB=0.0
cfg.COVARIATE_DROP_PROB=0.0
cfg.TIME_SMOOTHNESS_WEIGHT=0.0
if 'USE_ATTAINED_AGE_TERM' not in _theirs_all:cfg.USE_ATTAINED_AGE_TERM=False
if ARM=='C':cfg.BATCH_SIZE=int(os.getenv('GEMMA_MICRO_BATCH','2'))
if TEST_ONLY_CPU:torch.set_num_threads(1)



def _canonical_json(value):
    return json.loads(json.dumps(_json_safe(value), default=str))


PARITY_KEYS = sorted([name for name in asdict(cfg) if name.startswith("EVAL_")]
                     + ["ENTRY_DELAY_MONTHS", "ENTRY_TIME_EPS_MONTHS", "PROB_EPS", "RATE_EPS", "DEATH_TIE_POLICY", "LOSS_ALPHA_NLL",
                        "LOSS_GAMMA_DEATH", "NORMALISE_DISEASE_CR_NLL_BY_CONDITION", "FULL_SEQUENCE_MAX_EVENTS", "CR_TARGET_CENSOR",
                        "CR_TARGET_CONDITION", "CR_TARGET_DEATH", "LOG_RATE_MIN", "LOG_RATE_MAX", "SAMPLE_CONTINUOUS_COLS_RAW",
                        "EXCLUDE_CRP_FROM_INPUTS", "INCLUDE_CRP_AS_EVENT_VALUE", "USE_EVENT_VALUE_CHANNEL"])
PARITY_TABLE = pd.DataFrame([{"setting": k, "gemma": _canonical_json(asdict(cfg)[k]), "crescent": _canonical_json(_theirs_all.get(k, "<missing>")),
                              "equal": _canonical_json(asdict(cfg)[k]) == _canonical_json(_theirs_all.get(k, "<missing>"))} for k in PARITY_KEYS])
if not PARITY_TABLE["equal"].all():
    raise RuntimeError("Task/evaluation settings differ from the CRESCENT run:\n" + PARITY_TABLE[~PARITY_TABLE["equal"]].to_string(index=False))
if float(cfg.LOSS_GAMMA_DEATH) != float(_theirs_all["LOSS_GAMMA_DEATH"]) or float(cfg.LOSS_ALPHA_NLL) != float(_theirs_all["LOSS_ALPHA_NLL"]):
    raise ValueError("Loss weights must match the chosen source CRESCENT run.")
if int(cfg.ENTRY_DELAY_MONTHS) != 6 or window_start_bin(cfg.ENTRY_DELAY_MONTHS) != 0:
    raise ValueError("This comparator requires the six-month delayed-entry landmark.")
for _key, _value in CRESCENT_CONFIG["delayed_entry_contract"].items():
    if _canonical_json(delayed_entry_contract().get(_key)) != _canonical_json(_value):
        raise RuntimeError(f"delayed_entry_contract differs from CRESCENT at key {_key!r}.")
if set(delayed_entry_contract()) != set(CRESCENT_CONFIG["delayed_entry_contract"]):
    raise RuntimeError("delayed_entry_contract key set differs from CRESCENT.")
if Path(str(cfg.PRIMARY_CHECKPOINT)).name != "best_model_by_val_loss.pt":
    raise ValueError("PRIMARY_CHECKPOINT must be best_model_by_val_loss.pt (validation joint NLL).")
RECIPE_KEYS = ["BATCH_SIZE", "MAX_ITERATIONS", "VALIDATION_INTERVAL", "WARMUP_ITERATIONS", "LR", "WEIGHT_DECAY", "EMA_DECAY", "GRAD_CLIP",
               "SCHEDULER_MIN_LR_RATIO", "DROPOUT_ENCODER", "DROPOUT_HAZARD", "DROPOUT_DEATH", "EVENT_MASK_PROB", "PATIENCE_VALIDATIONS", "MIN_DELTA"]
RECIPE_COMPARISON = pd.DataFrame([{"setting": k, "gemma": _canonical_json(asdict(cfg)[k]), "crescent": _canonical_json(_theirs_all.get(k, "<missing>")),
                                   "equal": _canonical_json(asdict(cfg)[k]) == _canonical_json(_theirs_all.get(k, "<missing>"))} for k in RECIPE_KEYS])
RECIPE_COMPARISON.to_csv(RUN_DIR / "training_recipe_vs_crescent.csv", index=False)

CLUSTER_GPU_INFO = {"device": str(cfg.DEVICE), "logical_device_id": 0, "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    "visible_cuda_device_count": int(torch.cuda.device_count()), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "cpu_fallback_allowed": False, "test_only_cpu_override": bool(TEST_ONLY_CPU),
                    "gemma_gpu_selector": GEMMA_GPU or None}
if TEST_ONLY_CPU:
    warnings.warn("GEMMA_TEST_ONLY_CPU=1: unit-test mode. Never use this setting for a reported run.", RuntimeWarning)
print(json.dumps({"notebook_role": ARM, "phase": PHASE, "seed": SEED, "hpid": HPID, "model_variant": cfg.MODEL_VARIANT, "crescent_run_dir": str(CRESCENT_RUN_DIR),
                  "run_dir": str(RUN_DIR), "cache_dir": str(CACHE_DIR), "comparison_dir": str(COMPARISON_DIR), "device": str(cfg.DEVICE),
                  "parity_settings_checked": int(len(PARITY_TABLE)), "recipe_settings_differing_from_crescent": RECIPE_COMPARISON[~RECIPE_COMPARISON["equal"]]["setting"].tolist()}, indent=2))

# Shared three-notebook protocol. Source data are read-only; generated files are new V7 artifacts.
def _plain(value):
    if isinstance(value, (Path, torch.device)):
        return str(value)
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def canonical_hash(value):
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(_plain(payload), indent=2, allow_nan=False))
    tmp.replace(path)


def benchmark_binding():
    return {'input_hashes': dict(MODEL_INPUT_HASHES), 'condition_names': list(condition_names),
            'K': int(K), 'target_metadata_sha256': sha256_file(CRESCENT_RUN_DIR/'canonical_target_metadata.json'),
            'entry_contract': delayed_entry_contract(), 'continuous_columns': list(cfg.SAMPLE_CONTINUOUS_COLS_RAW),
            'max_events': int(cfg.FULL_SEQUENCE_MAX_EVENTS)}


def expected_cache_binding(identity):
    return {'schema': 'gemma.cache_binding.v7', 'benchmark': benchmark_binding(),
            'encoder_fingerprint_sha256': identity['fingerprint_sha256'],
            'text_serialisation_sha256': identity['text_serialisation_sha256'],
            'layers': identity['layers'], 'extract_dtype': identity['extract_dtype'],
            'cache_dtype': identity['cache_dtype'], 'transformers_version': identity['transformers_version'],
            'torch_version': identity['torch_version'], 'pipeline_code_sha256': PIPELINE_CODE_SHA256}


def bind_or_check_cache(identity, create=False):
    path = CACHE_DIR/'cache_binding.json'
    expected = expected_cache_binding(identity)
    if path.exists():
        if json.loads(path.read_text()) != _plain(expected):
            raise RuntimeError('Cache identity changed (data/tokenizer/encoder/text/code/dtype/framework). Choose a fresh GEMMA_CACHE_DIR.')
    elif create:
        # Do not bless an existing unbound partial cache.
        if any(CACHE_DIR.glob('*_tokens.npz')) or any(CACHE_DIR.glob('*_progress.json')) or any(CACHE_DIR.glob('*_event_states.npy')):
            raise RuntimeError('Unbound cache artifacts exist. Use a fresh V7 cache directory.')
        atomic_json(path, expected)
    else:
        raise FileNotFoundError('No V7 cache identity. Run the extraction notebook first.')
    return canonical_hash(expected)


def effective_spec(include_seed=True):
    settings = _plain(asdict(cfg))
    settings.pop('DEVICE', None)
    if not include_seed:
        settings.pop('SEED', None); settings.pop('TRAINING_SEED', None)
    return {'schema': 'gemma.effective_spec.v7', 'arm': ARM, 'config': settings,
            'benchmark': benchmark_binding(), 'pipeline_code_sha256': PIPELINE_CODE_SHA256,
            'encoder_fingerprint_sha256': CACHE_IDENTITY['fingerprint_sha256'],
            'text_serialisation_sha256': CACHE_IDENTITY['text_serialisation_sha256'],
            'layer_choice': LAYER_CHOICE, 'parent_arm': 'B',
            'lora': {'rank': LORA_RANK, 'alpha': LORA_ALPHA, 'dropout': LORA_DROPOUT,
                     'adapter_lr': ADAPTER_LR, 'targets': ['q_proj','k_proj','v_proj','o_proj'],
                     'effective_batch': EFFECTIVE_BATCH, 'max_epochs': MAX_EPOCHS_C} if ARM=='C' else None,
            'cpu_test_only': TEST_ONLY_CPU}


def guard_test_access(identity, check_this_run=False):
    """Validate both arm decisions and checkpoint hashes BEFORE any test pickle/cache read."""
    required = [a.strip().upper() for a in os.getenv('GEMMA_REQUIRED_ARMS','B,C').split(',') if a.strip()]
    if not required or not set(required) <= {'B','C'}:
        raise ValueError('GEMMA_REQUIRED_ARMS must contain B and/or C.')
    expected_benchmark = _plain(benchmark_binding())
    decisions = {}
    for arm in required:
        path = COMPARISON_DIR/f'selection_{arm}_locked.json'
        if not path.is_file():
            raise FileNotFoundError(f'Finish validation-only selection for arm {arm}: {path}')
        decision = json.loads(path.read_text())
        if decision.get('schema') != 'gemma.arm_selection.v7' or decision['arm'] != arm or decision.get('test_parsed_before_lock') is not False:
            raise ValueError('Invalid arm selection lock.')
        if decision['benchmark'] != expected_benchmark or decision['pipeline_code_sha256'] != PIPELINE_CODE_SHA256:
            raise ValueError('Selection lock belongs to a different benchmark/code version.')
        for key in ('fingerprint_sha256','text_serialisation_sha256'):
            if decision['encoder'][key] != identity[key]:
                raise ValueError('Selection lock encoder/text identity differs from the current encoder.')
        if bool(decision['cpu_test_only']) != bool(TEST_ONLY_CPU):
            raise ValueError('Synthetic and cohort runs cannot be mixed.')
        for run in decision['runs']:
            checkpoint = Path(run['run_dir'])/'best_model_by_val_loss.pt'
            if not checkpoint.is_file() or sha256_file(checkpoint) != run['checkpoint_sha256']:
                raise ValueError('A selected checkpoint changed or is missing.')
        decisions[arm] = decision
    if check_this_run:
        if ARM not in decisions:
            raise ValueError('This arm is not among GEMMA_REQUIRED_ARMS.')
        runs = [r for r in decisions[ARM]['runs'] if Path(r['run_dir']).resolve()==RUN_DIR.resolve() and int(r['seed'])==SEED]
        if len(runs)!=1 or runs[0]['effective_spec_sha256'] != canonical_hash(effective_spec()):
            raise ValueError('Current effective configuration differs from the selected run.')
    return decisions


def initialize_lora_budget():
    if ARM != 'C':
        return
    steps_per_epoch = math.ceil(len(train_samples)/EFFECTIVE_BATCH)
    cfg.MAX_ITERATIONS = int(os.getenv('GEMMA_LORA_MAX_ITERATIONS', str(MAX_EPOCHS_C*steps_per_epoch)))
    cfg.VALIDATION_INTERVAL = int(os.getenv('GEMMA_VALIDATION_INTERVAL', str(max(1, math.ceil(steps_per_epoch/4)))))
    cfg.WARMUP_ITERATIONS = min(cfg.WARMUP_ITERATIONS, max(1,cfg.MAX_ITERATIONS//10))
    cfg.PATIENCE_VALIDATIONS = int(os.getenv('GEMMA_PATIENCE_VALIDATIONS','0'))


def select_and_lock_arm():
    if PHASE != 'select':
        return
    mode = os.getenv('GEMMA_SELECT_MODE','lock').lower()
    rows = []
    for run in sorted(OUTPUT_ROOT.glob(f'GEMMA4_E4B_{ARM}_V7_*')):
        record = {'run_dir': str(run.resolve()), 'status': 'incomplete'}
        try:
            meta = json.loads((run/'run_record.json').read_text())
            record.update(meta)
            summary_path = run/'cell6_training_summary_death_adjusted.json'
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                local = json.loads((run/'run_selection_locked.json').read_text())
                checkpoint=run/'best_model_by_val_loss.pt'
                if local['checkpoint_sha256'] != sha256_file(checkpoint):
                    raise ValueError('Checkpoint changed after training selection.')
                if summary['training_complete'] is not True:
                    raise ValueError('Incomplete training summary.')
                record.update({'status':'complete','best_val_loss':summary['best_val_loss'],
                               'checkpoint_sha256':local['checkpoint_sha256'],
                               'test_parsed_before_lock':local['test_parsed_before_lock']})
        except Exception as e:
            record['status']=f'failed: {type(e).__name__}: {str(e)[:160]}'
        rows.append(record)
    table=pd.DataFrame(rows)
    table.to_csv(COMPARISON_DIR/f'arm_{ARM}_search_log.csv',index=False)
    if table.empty:
        raise RuntimeError('No runs found for this arm.')
    selected_path=COMPARISON_DIR/f'selected_configuration_{ARM}.json'
    if mode=='grid':
        candidates=[r for r in rows if r.get('status')=='complete' and r.get('hpid')!='final' and int(r.get('seed',-1))==42]
        if not candidates:
            raise RuntimeError('Grid selection needs completed non-final trials at seed 42.')
        best=min(candidates,key=lambda r:r['best_val_loss'])
        atomic_json(selected_path,{'arm':ARM,'hp_signature':best['hp_signature'],'configuration':best['hp_spec'],
                                  'selected_hpid':best['hpid'],'validation_loss':best['best_val_loss'],
                                  'rule':'minimum validation joint NLL among completed grid trials; failures retained in search log'})
        print('Selected configuration:',best['hpid'],'; rerun as HPID=final for all required seeds.')
        return
    if mode!='lock':
        raise ValueError('GEMMA_SELECT_MODE must be grid or lock.')
    path=COMPARISON_DIR/f'selection_{ARM}_locked.json'
    if path.exists():
        raise FileExistsError('Selection locks are immutable. Use a new comparison directory for a new study.')
    seeds=sorted(int(s) for s in os.getenv('GEMMA_REQUIRED_SEEDS','42,43,44,45,46').split(',') if s.strip())
    finals=[r for r in rows if r.get('hpid')=='final']
    if sorted(int(r.get('seed',-1)) for r in finals)!=seeds or any(r['status']!='complete' for r in finals):
        raise RuntimeError(f'Complete exactly one final run for every required seed {seeds}.')
    signatures={r['hp_signature'] for r in finals}
    if len(signatures)!=1:
        raise ValueError('Final seeds differ in their complete hyperparameter specifications.')
    if selected_path.exists():
        chosen=json.loads(selected_path.read_text())
        if next(iter(signatures))!=chosen['hp_signature']:
            raise ValueError('Final configuration differs from validation grid selection.')
        selection='validation-selected grid configuration'
    else:
        if any(r.get('hpid')!='final' for r in rows):
            raise ValueError('Grid trials exist: run select/grid before locking final seeds.')
        selection='single prespecified configuration; no grid search'
    if any(CACHE_DIR.glob('test_*')) and not RUN_ALL:
        raise ValueError('Test cache already exists; cannot create a new validation-only lock.')
    for r in finals:
        if r['benchmark']!=_plain(benchmark_binding()) or r['pipeline_code_sha256']!=PIPELINE_CODE_SHA256:
            raise ValueError('Final seeds used different data/code.')
        if r.get('test_parsed_before_lock') is not False or (Path(r['run_dir'])/'canonical_test_samples.pkl').exists():
            raise ValueError('Test rows were present before selection lock.')
        if r['encoder_fingerprint_sha256'] != CACHE_IDENTITY['fingerprint_sha256'] or r['text_serialisation_sha256'] != CACHE_IDENTITY['text_serialisation_sha256']:
            raise ValueError('Final seeds used different encoder/text identities.')
    atomic_json(path,{'schema':'gemma.arm_selection.v7','arm':ARM,'selection':selection,
                      'locked_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'test_parsed_before_lock':False,
                      'benchmark':benchmark_binding(),'pipeline_code_sha256':PIPELINE_CODE_SHA256,
                      'cpu_test_only':TEST_ONLY_CPU,'mode':'single_run_train_lock_evaluate' if RUN_ALL else 'staged',
                      'test_cache_present_at_lock':bool(any(CACHE_DIR.glob('test_*'))),'encoder':{k:CACHE_IDENTITY[k] for k in ('fingerprint_sha256','text_serialisation_sha256')},
                      'hp_signature':next(iter(signatures)),
                      'runs':[{k:r[k] for k in ('run_dir','seed','checkpoint_sha256','effective_spec_sha256')} for r in finals]})
    print('Locked arm',ARM,'for seeds',seeds,':',path)


def load_locked_test_dataset():
    global test_samples, test_ds
    guard_test_access(CACHE_IDENTITY,check_this_run=True)
    source=CRESCENT_RUN_DIR/'canonical_test_samples.pkl'
    test_samples=pd.read_pickle(source)
    validate_canonical_samples(test_samples,'test')
    compare_with_crescent_index_audit(test_samples,'test')
    if set(test_samples.eid.astype(str)) & (set(train_samples.eid.astype(str))|set(val_samples.eid.astype(str))):
        raise ValueError('Test/train participant overlap.')
    shutil.copy2(source,RUN_DIR/'canonical_test_samples.pkl')
    test_ds=FixedLandmarkSequenceDataset(test_samples,split_name='test',full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS)
    return test_ds

# --- Notebook cell 4 ---
lock = COMPARISON_DIR / "selection_B_locked.json"
print(lock.is_file(), lock)
