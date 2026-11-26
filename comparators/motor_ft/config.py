# MOTOR-FT — Run settings and delayed-entry landmark contract
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 2 ---
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

SOURCE = Path("runs/CRESCENT_Final").expanduser().resolve()
RUN_DIR = Path(os.getenv("MOTOR_FINETUNE_RUN_DIR", "runs/motor_ft")).expanduser().resolve()
WORKFLOW = os.getenv("MOTOR_WORKFLOW", "train_and_evaluate")
RESUME = os.getenv("MOTOR_RESUME_CHECKPOINT", "").strip()
device_request = os.getenv("EHR_DEVICE", "auto").strip().lower()
if device_request == "auto":
    device_request = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(device_request)
if WORKFLOW not in {"train_and_evaluate", "evaluation_only"}:
    raise ValueError("Select train_and_evaluate or evaluation_only.")
if SOURCE == RUN_DIR or SOURCE in RUN_DIR.parents or RUN_DIR in SOURCE.parents:
    raise ValueError("Use separate source and output directories.")
if RESUME and Path(RESUME).expanduser().resolve() != RUN_DIR / "last_checkpoint.pt":
    raise ValueError("Resume from this comparator run's last_checkpoint.pt.")
if WORKFLOW == "evaluation_only" and RESUME:
    raise ValueError("Evaluation-only loads the best checkpoint. Clear MOTOR_RESUME_CHECKPOINT.")
if WORKFLOW == "train_and_evaluate" and not RESUME and RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Fresh training requires an empty output directory: {RUN_DIR}")
if RESUME and not (RUN_DIR / "last_checkpoint.pt").is_file():
    raise FileNotFoundError(RESUME)
if WORKFLOW == "evaluation_only" and not (RUN_DIR / "best_model_by_val_loss.pt").is_file():
    raise FileNotFoundError(RUN_DIR / "best_model_by_val_loss.pt")

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))

def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temp.replace(path)

def sha256_file(path):
    with open(path, "rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

reference_config = read_json(SOURCE / "config.json")
settings = reference_config["config"]
ARCHITECTURE_SOURCE = "MOTOR_landmark_PE_v1"
# Inherit the effective run settings, including any environment overrides. Keep
# architecture-specific CRESCENT fields out of the comparator configuration.
shared_keys = """AGE_TOL_YEARS ENTRY_DELAY_MONTHS ENTRY_TIME_EPS_MONTHS FULL_SEQUENCE_MAX_EVENTS
BATCH_SIZE NUM_WORKERS SEED TRAINING_SEED MAX_ITERATIONS WARMUP_ITERATIONS
VALIDATION_INTERVAL MIN_DELTA LOG_INTERVAL LR SCHEDULER_MIN_LR_RATIO WEIGHT_DECAY WEIGHT_DECAY_EXEMPT_1D
EMA_DECAY GRAD_CLIP DROPOUT_ENCODER EVENT_MASK_PROB COVARIATE_DROP_PROB
LOSS_ALPHA_NLL LOSS_GAMMA_DEATH NORMALISE_DISEASE_CR_NLL_BY_CONDITION
TIME_SMOOTHNESS_WEIGHT PATIENCE_VALIDATIONS RATE_EPS PROB_EPS LOG_RATE_MIN LOG_RATE_MAX
CONDITION_BASELINE_FALLBACK_INITIAL_LOG_RATE DEATH_BASELINE_FALLBACK_INITIAL_LOG_RATE
CR_TARGET_CENSOR CR_TARGET_CONDITION CR_TARGET_DEATH DEATH_TIE_POLICY
EXCLUDE_CRP_FROM_INPUTS INCLUDE_CRP_AS_EVENT_VALUE USE_EVENT_VALUE_CHANNEL
SAMPLE_CONTINUOUS_COLS_RAW""".split()
cfg = SimpleNamespace(**{k:v for k,v in settings.items() if k in shared_keys or k.startswith("EVAL_")})
# The downstream comparison uses a fixed optimizer-update budget.
cfg.MAX_ITERATIONS = 50_000
cfg.VALIDATION_INTERVAL = 1_000
cfg.VAL_PERFORMANCE_INTERVAL = 5_000
cfg.DEVICE = DEVICE
cfg.MODEL_VARIANT = "Task-adapted MOTOR"
cfg.PRIMARY_CHECKPOINT = "best_model_by_val_loss.pt"
if (cfg.ENTRY_DELAY_MONTHS != 6 or tuple(cfg.EVAL_HORIZONS_MONTHS) != (12, 24, 60, 120)
        or cfg.FULL_SEQUENCE_MAX_EVENTS != 25 or cfg.LOSS_ALPHA_NLL != 1.0
        or cfg.LOSS_GAMMA_DEATH != 0.10 or cfg.TIME_SMOOTHNESS_WEIGHT != 0
        or cfg.PATIENCE_VALIDATIONS != 0 or cfg.NORMALISE_DISEASE_CR_NLL_BY_CONDITION):
    raise ValueError("Source differs from the specified six-month, fixed-budget CRESCENT task.")
haz = SimpleNamespace(**reference_config["hazard_binning"])
if (haz.BIN_WIDTH_MONTHS, haz.MAX_HORIZON_MONTHS, haz.N_BINS) != (6, 132, 22):
    raise ValueError("The expected hazard grid has 22 six-month bins.")
if reference_config["fixed_index_contract"]["calendar_months_from_recruitment"] != 1:
    raise ValueError("One-month-cap source required.")
CFG = SimpleNamespace(D_MODEL=768, N_HEADS=12, N_LAYERS=6, D_FF=3072, ATTENTION_WIDTH=26,
                      DROPOUT=0.10, VALUE_D_MODEL=64, STATIC_D_MODEL=128, SURVIVAL_DIM=256)
RUN_DIR.mkdir(parents=True, exist_ok=True)

def set_global_seed(seed: int):
    seed = int(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



set_global_seed(cfg.SEED)
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
