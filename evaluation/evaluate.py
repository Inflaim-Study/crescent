"""Stage 6: locked test evaluation, then dense-grid and integrated metrics.

Extracted from the CRESCENT_Final notebook (cells 18, 20). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 18
# =============================================================================

import json
import math
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# =============================================================================
# Required globals
# =============================================================================

assert "cfg" in globals() and "haz" in globals() and "RUN_DIR" in globals(), "Run Cell 1 first."
assert "model" in globals(), "Run Cell 5/6 first."
assert "test_loader" in globals() and "test_samples" in globals(), "Run Cell 4.5 first."
assert "train_samples" in globals(), "Canonical train samples are required for censoring KMs."
assert "K" in globals(), "Run canonical loader Cell 2 first."

RUN_DIR = Path(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(getattr(cfg, "DEVICE", "cuda:0"))
if str(DEVICE) != "cuda:0" or torch.cuda.current_device() != 0:
    raise RuntimeError("Evaluation is locked to the scheduler-visible cuda:0 device.")

ENTRY_DELAY_MONTHS = float(cfg.ENTRY_DELAY_MONTHS)
EVAL_HORIZONS_MONTHS = tuple(int(x) for x in getattr(cfg, "EVAL_HORIZONS_MONTHS", (12, 24, 60, 120)))

PROB_EPS = float(getattr(cfg, "PROB_EPS", 1e-7))
TIME_EPS = 1e-8

CR_TARGET_CENSOR = int(getattr(cfg, "CR_TARGET_CENSOR", 0))
CR_TARGET_CONDITION = int(getattr(cfg, "CR_TARGET_CONDITION", 1))
CR_TARGET_DEATH = int(getattr(cfg, "CR_TARGET_DEATH", 2))

DEATH_TIE_POLICY = str(
    getattr(cfg, "DEATH_TIE_POLICY", "condition_if_same_date_else_death")
)

N_CAL_BINS = int(getattr(cfg, "EVAL_N_CALIBRATION_BINS", 10))
MIN_CAL_GROUP_N = int(getattr(cfg, "EVAL_MIN_CALIBRATION_GROUP_N", 100))
MIN_CAL_BIN_N = int(getattr(cfg, "EVAL_MIN_CALIBRATION_BIN_N", 50))
IPCW_TRUNC_EPS = float(getattr(cfg, "EVAL_IPCW_TRUNCATION_EPS", 1e-3))
CENSOR_SURVIVAL_FLOOR = float(
    getattr(cfg, "EVAL_CENSOR_SURVIVAL_FLOOR", 0.01)
)

# Force no-bootstrap for this run.
EVAL_BOOTSTRAP_CI = False
EVAL_BOOTSTRAP_N = 0
EVAL_BOOTSTRAP_SEED = int(getattr(cfg, "EVAL_BOOTSTRAP_SEED", 101))
EVAL_BOOTSTRAP_ALPHA = float(getattr(cfg, "EVAL_BOOTSTRAP_ALPHA", 0.05))
EVAL_BOOTSTRAP_INCLUDE_CTD = False

EVAL_CTD_MAX_CASES = int(getattr(cfg, "EVAL_CTD_MAX_CASES", 1500))
EVAL_CTD_MAX_COMPARATORS = int(getattr(cfg, "EVAL_CTD_MAX_COMPARATORS", 3000))

RNG = np.random.default_rng(EVAL_BOOTSTRAP_SEED)

print("=" * 88)
print("CELL 7 — Locked test evaluation, no bootstrap")
print("=" * 88)
print(
    json.dumps(
        {
            "RUN_DIR": str(RUN_DIR),
            "DEVICE": str(DEVICE),
            "ENTRY_DELAY_MONTHS": ENTRY_DELAY_MONTHS,
            "EVAL_HORIZONS_MONTHS": list(EVAL_HORIZONS_MONTHS),
            "K": int(K),
            "EVAL_BOOTSTRAP_CI": EVAL_BOOTSTRAP_CI,
            "EVAL_BOOTSTRAP_N": EVAL_BOOTSTRAP_N,
            "EVAL_CTD_MAX_CASES": EVAL_CTD_MAX_CASES,
            "EVAL_CTD_MAX_COMPARATORS": EVAL_CTD_MAX_COMPARATORS,
            "CTD_STATUS": "disabled_not_validly_implemented",
            "CENSORING_KM_SOURCE": "canonical_train_samples",
        },
        indent=2,
    )
)


# =============================================================================
# Condition names
# =============================================================================

if "condition_names" in globals() and isinstance(condition_names, list) and len(condition_names) == int(K):
    CONDITION_NAMES = [str(x) for x in condition_names]
elif "id_to_code" in globals() and isinstance(id_to_code, dict):
    CONDITION_NAMES = [str(id_to_code.get(i, f"condition_{i}")) for i in range(int(K))]
else:
    CONDITION_NAMES = [f"condition_{i}" for i in range(int(K))]


# =============================================================================
# General helpers
# =============================================================================

def move_batch_to_device(batch, device):
    out = {}

    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v

    return out


def month_to_haz_bin(months):
    m = float(months)
    b = int(np.ceil(max(m, 0.0) / float(haz.BIN_WIDTH_MONTHS)) - 1)
    b = max(0, min(b, int(haz.N_BINS) - 1))
    return int(b)


def _safe_np(x, dtype=None):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    arr = np.asarray(x)

    if dtype is not None:
        arr = arr.astype(dtype)

    return arr


def _metric_float(x):
    try:
        x = float(x)
    except Exception:
        return np.nan

    return x if np.isfinite(x) else np.nan


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not bool(valid.any()):
        return np.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[min(index, values.size - 1)])


def condition_wins_tie_eval(t_k, t_d):
    """
    Tie policy for observed condition-vs-death labels in delayed-entry evaluation.
    """
    policy = str(DEATH_TIE_POLICY)

    if policy == "condition_if_same_bin":
        return month_to_haz_bin(t_k) == month_to_haz_bin(t_d)

    if policy == "death_if_same_bin":
        return False

    if policy == "condition_if_same_bin_else_death":
        return month_to_haz_bin(t_k) == month_to_haz_bin(t_d)

    if policy == "death_if_same_date":
        return False

    # Default: condition_if_same_date_else_death.
    return abs(float(t_k) - float(t_d)) <= 1e-8


def fixed_binary_metrics(y, pred, mask, weights=None):
    """IPCW-weighted AUROC/AUPRC under a Boolean known mask."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if weights is None:
        weights = np.ones_like(pred, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = (
        mask
        & np.isfinite(y)
        & np.isfinite(pred)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    y = y[valid]
    pred = pred[valid]
    weights = weights[valid]
    out = {
        "auroc": np.nan,
        "auprc": np.nan,
        "event_rate": (
            float(np.average(y, weights=weights)) if y.size > 0 else np.nan
        ),
    }
    if y.size == 0 or not SKLEARN_AVAILABLE:
        return out
    if len(np.unique(y.astype(int))) == 2:
        out["auroc"] = float(
            roc_auc_score(y, pred, sample_weight=weights)
        )
        out["auprc"] = float(
            average_precision_score(y, pred, sample_weight=weights)
        )
    return out


def weighted_brier_nbll(y, pred, weights, normalization_n):
    """Horvitz-Thompson IPCW Brier and binary log loss."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    pred = np.clip(pred, PROB_EPS, 1.0 - PROB_EPS)
    valid = (
        np.isfinite(y)
        & np.isfinite(pred)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(normalization_n) <= 0 or not bool(valid.any()):
        return {
            "brier_ipcw": np.nan,
            "nbll_ipcw": np.nan,
            "weight_sum": 0.0,
            "normalization_n": int(normalization_n),
        }
    denominator = float(int(normalization_n))
    brier = np.sum(
        weights[valid] * ((pred[valid] - y[valid]) ** 2)
    ) / denominator
    nbll = -np.sum(
        weights[valid]
        * (
            y[valid] * np.log(pred[valid])
            + (1.0 - y[valid]) * np.log1p(-pred[valid])
        )
    ) / denominator
    return {
        "brier_ipcw": float(brier),
        "nbll_ipcw": float(nbll),
        "weight_sum": float(np.sum(weights[valid])),
        "normalization_n": int(normalization_n),
    }


class Cell7CensoringKM:
    """Training-split administrative-censoring KM with G(t-) queries."""

    def __init__(self, times, censor_events):
        time = np.asarray(times, dtype=np.float64)
        event = np.asarray(censor_events, dtype=bool)
        valid = np.isfinite(time) & (time >= 0.0)
        time = time[valid]
        event = event[valid]
        if time.size == 0:
            self.times = np.asarray([], dtype=np.float64)
            self.survival_after = np.asarray([], dtype=np.float64)
            return
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order].astype(np.int64)
        unique, first = np.unique(time, return_index=True)
        at_risk = time.size - first
        censor_count = np.add.reduceat(event, first)
        factors = 1.0 - censor_count / np.maximum(at_risk, 1)
        self.times = unique
        self.survival_after = np.cumprod(np.clip(factors, 0.0, 1.0))

    def before(self, query):
        q = np.asarray(query, dtype=np.float64)
        index = np.searchsorted(self.times, q, side="left") - 1
        out = np.ones_like(q, dtype=np.float64)
        valid = index >= 0
        if bool(valid.any()):
            out[valid] = self.survival_after[index[valid]]
        return np.maximum(out, CENSOR_SURVIVAL_FLOOR)


def make_condition_outcomes(frame):
    """Competing-risk outcomes conditional on delayed-entry eligibility."""
    return make_delayed_entry_outcomes_numpy(
        np.stack(frame["cr_at_risk"].to_numpy()),
        np.stack(frame["cr_event_type"].to_numpy()),
        np.stack(frame["cr_event_time_months"].to_numpy()),
        frame["C_months"].to_numpy(dtype=float),
    )


def _fit_cell7_training_censoring_kms(samples):
    outcomes = make_condition_outcomes(samples)
    condition_kms = [
        Cell7CensoringKM(outcomes["exit_time"][outcomes["eligible"][:, k], k], outcomes["status"][outcomes["eligible"][:, k], k] == 0)
        for k in range(int(K))
    ]
    death = make_delayed_entry_death_outcomes_numpy(
        samples["C_months"].to_numpy(float), samples["death_observed_post_index"].to_numpy(float), samples["death_time_months"].to_numpy(float),
    )
    selected = death["eligible"]
    death_km = Cell7CensoringKM(death["exit_time"][selected], death["status"][selected] == 0)
    return condition_kms, death_km


CELL7_CONDITION_CENSORING_KMS, CELL7_DEATH_CENSORING_KM = (
    _fit_cell7_training_censoring_kms(train_samples)
)


def ipcw_weights_for_binary_window(
    time_rel,
    status,
    censor_event,
    horizon_months,
    eligible,
    censoring_km,
):
    """Train-KM IPCW weights; unknown censored rows and ineligible rows get zero."""
    time_rel = np.asarray(time_rel, dtype=np.float64)
    status = np.asarray(status, dtype=np.int16)
    censor_event = np.asarray(censor_event, dtype=bool)
    eligible = np.asarray(eligible, dtype=bool)
    horizon = float(horizon_months)
    weights = np.zeros_like(time_rel, dtype=np.float64)
    event_or_competing = eligible & ((status == 1) | (status == 2))
    if bool(event_or_competing.any()):
        weights[event_or_competing] = 1.0 / censoring_km.before(
            time_rel[event_or_competing]
        )
    observed_control = eligible & (status == 0) & (~censor_event)
    if bool(observed_control.any()):
        weights[observed_control] = 1.0 / float(
            censoring_km.before(horizon)
        )
    weights[~np.isfinite(weights)] = 0.0
    return weights


def ctd_style_concordance(*args, **kwargs):
    """Removed: the previous implementation was sampled fixed-window AUROC."""
    return np.nan


def aalen_johansen_cif_at_horizon(time, status, horizon_months):
    """
    Aalen-Johansen CIF for cause 1 at a fixed horizon.

    status:
      0 = no event / administrative censoring
      1 = condition event
      2 = death competing event
    """
    time = np.asarray(time, dtype=np.float64)
    status = np.asarray(status, dtype=np.int16)

    h = float(horizon_months)

    valid = np.isfinite(time) & (time >= 0)

    time = time[valid]
    status = status[valid]

    if time.size == 0:
        return np.nan

    # Events after horizon are treated as no event by horizon.
    status_h = np.where(time <= h + TIME_EPS, status, 0)
    time_h = np.minimum(time, h)

    event_times = np.unique(time_h[(status_h == 1) | (status_h == 2)])

    if event_times.size == 0:
        return 0.0

    S = 1.0
    cif1 = 0.0

    for t in np.sort(event_times):
        at_risk = time_h >= t - TIME_EPS
        n_risk = int(at_risk.sum())

        if n_risk <= 0:
            continue

        d1 = int(((np.abs(time_h - t) <= TIME_EPS) & (status_h == 1)).sum())
        d2 = int(((np.abs(time_h - t) <= TIME_EPS) & (status_h == 2)).sum())

        cif1 += S * (d1 / n_risk)
        S *= max(0.0, 1.0 - ((d1 + d2) / n_risk))

    return float(cif1)


# =============================================================================
# Delayed-entry disease labels
# =============================================================================

def make_condition_window_observed_arrays(
    t_tilde,
    at_risk,
    C_months,
    death_observed_post_index,
    death_time_months,
    horizon_months,
):
    """
    Shared validation/test label function for the death-adjusted disease estimand.

    Conditional estimand:
      P(condition k occurs within (entry, entry+horizon] before death
        | condition-free and alive through the entry delay)
    """
    t_tilde = np.asarray(t_tilde, dtype=np.float64)
    at_risk = np.asarray(at_risk, dtype=bool)
    C_months = np.asarray(C_months, dtype=np.float64)
    death_observed_post_index = np.asarray(death_observed_post_index, dtype=bool)
    death_time_months = np.asarray(death_time_months, dtype=np.float64)

    B, Kloc = t_tilde.shape

    h = float(horizon_months)
    endpoint = ENTRY_DELAY_MONTHS + h

    C_mat = C_months[:, None]
    death_time_mat = death_time_months[:, None]
    death_obs_mat = death_observed_post_index[:, None]

    cond_finite = np.isfinite(t_tilde)
    death_finite = np.isfinite(death_time_mat)

    cond_before_or_at_entry = cond_finite & (t_tilde <= ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
    death_before_or_at_entry = (
        death_obs_mat
        & death_finite
        & (death_time_mat <= ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
    )

    eligible = (
        at_risk
        & (C_mat > ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
        & (~cond_before_or_at_entry)
        & (~death_before_or_at_entry)
    )

    cond_in_window = (
        eligible
        & cond_finite
        & (t_tilde > ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
        & (t_tilde <= endpoint + TIME_EPS)
    )

    death_in_window = (
        eligible
        & death_obs_mat
        & death_finite
        & (death_time_mat > ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
        & (death_time_mat <= endpoint + TIME_EPS)
    )

    condition_event = np.zeros((B, Kloc), dtype=bool)
    death_competing = np.zeros((B, Kloc), dtype=bool)

    rows, cols = np.where(eligible)

    for i, k in zip(rows.tolist(), cols.tolist()):
        has_c = bool(cond_in_window[i, k])
        has_d = bool(death_in_window[i, k])

        if has_c and has_d:
            tc = float(t_tilde[i, k])
            td = float(death_time_months[i])

            if tc < td - TIME_EPS:
                condition_event[i, k] = True
            elif td < tc - TIME_EPS:
                death_competing[i, k] = True
            else:
                if condition_wins_tie_eval(tc, td):
                    condition_event[i, k] = True
                else:
                    death_competing[i, k] = True

        elif has_c:
            condition_event[i, k] = True

        elif has_d:
            death_competing[i, k] = True

    observed_through_horizon = eligible & (C_mat >= endpoint - TIME_EPS)

    known = (
        condition_event
        | death_competing
        | observed_through_horizon
    )

    y = condition_event.astype(np.int8)

    status = np.zeros((B, Kloc), dtype=np.int16)
    status[condition_event] = 1
    status[death_competing] = 2

    time_rel = np.full((B, Kloc), h, dtype=np.float64)

    # Condition/death event times are relative to entry.
    time_rel[condition_event] = t_tilde[condition_event] - ENTRY_DELAY_MONTHS

    if death_competing.any():
        death_rel = death_time_mat - ENTRY_DELAY_MONTHS
        time_rel[death_competing] = np.repeat(death_rel, Kloc, axis=1)[death_competing]

    # Administrative censoring before horizon for unknown rows.
    censor_event = eligible & (~condition_event) & (~death_competing) & (C_mat < endpoint - TIME_EPS)

    if censor_event.any():
        censor_rel = C_mat - ENTRY_DELAY_MONTHS
        time_rel[censor_event] = np.repeat(censor_rel, Kloc, axis=1)[censor_event]

    time_rel = np.clip(time_rel, 0.0, h)

    admin_censored_known = known & (status == 0)

    return {
        "eligible": eligible,
        "known": known,
        "y": y,
        "status": status,
        "time_rel": time_rel,
        "censor_event": censor_event,
        "condition_event": condition_event,
        "death_competing": death_competing,
        "admin_censored_known": admin_censored_known,
    }


def make_death_window_observed_arrays(
    C_months,
    death_observed_post_index,
    death_time_months,
    horizon_months,
):
    """
    Patient-level death-head labels for the same post-entry window.
    """
    C_months = np.asarray(C_months, dtype=np.float64)
    death_observed_post_index = np.asarray(death_observed_post_index, dtype=bool)
    death_time_months = np.asarray(death_time_months, dtype=np.float64)

    h = float(horizon_months)
    endpoint = ENTRY_DELAY_MONTHS + h

    death_finite = np.isfinite(death_time_months)

    death_before_or_at_entry = (
        death_observed_post_index
        & death_finite
        & (death_time_months <= ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
    )

    eligible = (
        (C_months > ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
        & (~death_before_or_at_entry)
    )

    death_event = (
        eligible
        & death_observed_post_index
        & death_finite
        & (death_time_months > ENTRY_DELAY_MONTHS + ENTRY_TIME_EPS_MONTHS)
        & (death_time_months <= endpoint + TIME_EPS)
    )

    observed_through_horizon = eligible & (C_months >= endpoint - TIME_EPS)

    known = death_event | observed_through_horizon

    y = death_event.astype(np.int8)

    status = np.zeros_like(y, dtype=np.int16)
    status[death_event] = 1

    time_rel = np.full_like(C_months, h, dtype=np.float64)
    time_rel[death_event] = death_time_months[death_event] - ENTRY_DELAY_MONTHS

    censor_event = eligible & (~death_event) & (C_months < endpoint - TIME_EPS)

    if censor_event.any():
        time_rel[censor_event] = C_months[censor_event] - ENTRY_DELAY_MONTHS

    time_rel = np.clip(time_rel, 0.0, h)

    return {
        "eligible": eligible,
        "known": known,
        "y": y,
        "status": status,
        "time_rel": time_rel,
        "censor_event": censor_event,
        "death_event": death_event,
    }


# =============================================================================
# Prediction helpers
# =============================================================================

def conditional_window_cif_local(
    p_condition,
    p_death_k,
    p_neither,
    start_bin,
    end_bin,
):
    """
    Conditional window CIF from analytical piecewise-rate probabilities.

    Uses the same convention as the existing model helper:
      window = bins (start_bin, end_bin]
    """
    T = p_condition.shape[-1]

    s = int(max(-1, min(int(start_bin), T - 1)))
    e = int(max(0, min(int(end_bin), T - 1)))

    if e <= s:
        z = torch.zeros_like(p_condition[:, :, 0])
        return {
            "window_cif_condition": z,
            "window_cif_death_k": z,
            "window_survival_end": torch.ones_like(z),
        }

    pc = p_condition[:, :, s + 1 : e + 1]
    pd = p_death_k[:, :, s + 1 : e + 1]
    pn = p_neither[:, :, s + 1 : e + 1]

    one = torch.ones_like(pn[:, :, :1])

    survival_before = torch.cat(
        [
            one,
            torch.cumprod(pn[:, :, :-1], dim=-1),
        ],
        dim=-1,
    )

    window_cif_condition = torch.sum(survival_before * pc, dim=-1)
    window_cif_death_k = torch.sum(survival_before * pd, dim=-1)
    window_survival_end = torch.prod(pn, dim=-1)

    return {
        "window_cif_condition": window_cif_condition,
        "window_cif_death_k": window_cif_death_k,
        "window_survival_end": window_survival_end,
    }


def unadjusted_window_risk_from_p_cond(p_cond_given_alive, start_bin, end_bin):
    """
    Death-ignorant disease risk in the same post-entry window.
    """
    T = p_cond_given_alive.shape[-1]

    s = int(max(-1, min(int(start_bin), T - 1)))
    e = int(max(0, min(int(end_bin), T - 1)))

    if e <= s:
        return torch.zeros_like(p_cond_given_alive[:, :, 0])

    p = p_cond_given_alive[:, :, s + 1 : e + 1].clamp(PROB_EPS, 1.0 - PROB_EPS)

    return (1.0 - torch.prod(1.0 - p, dim=-1)).clamp(0.0, 1.0)


def death_window_risk_from_p_death(p_death, start_bin, end_bin):
    """
    Patient-level death risk in the same post-entry window.
    """
    T = p_death.shape[-1]

    s = int(max(-1, min(int(start_bin), T - 1)))
    e = int(max(0, min(int(end_bin), T - 1)))

    if e <= s:
        return torch.zeros_like(p_death[:, 0])

    p = p_death[:, s + 1 : e + 1].clamp(PROB_EPS, 1.0 - PROB_EPS)

    return (1.0 - torch.prod(1.0 - p, dim=-1)).clamp(0.0, 1.0)


# =============================================================================
# Strict primary checkpoint loading
# =============================================================================

def maybe_load_best_checkpoint(model):
    """
    Strictly load the configured validation-selected primary checkpoint.

    Evaluation never falls back to the last or in-memory model. Input hashes,
    condition order, model variant, and hazard binning are checked before the
    jointly trained from-scratch weights are accepted.
    """
    required_name = "best_model_by_val_loss.pt"  # minimum validation joint NLL
    configured_name = Path(
        str(getattr(cfg, "PRIMARY_CHECKPOINT", required_name))
    ).name
    if configured_name != required_name:
        raise ValueError(
            "Locked test evaluation requires the checkpoint selected by "
            "the best-validation-NLL checkpoint: PRIMARY_CHECKPOINT="
            f"{required_name!r}; got {configured_name!r}."
        )

    path = RUN_DIR / required_name
    if not path.exists():
        raise FileNotFoundError(
            "Locked test evaluation requires the validation-selected checkpoint: "
            f"{path}"
        )

    checkpoint = torch.load(path, map_location=DEVICE)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Checkpoint does not contain model_state_dict: {path}"
        )
    if checkpoint.get("model_input_hashes") != dict(MODEL_INPUT_HASHES):
        raise ValueError(
            "Checkpoint/input hash mismatch. Refusing locked test evaluation."
        )
    if int(checkpoint.get("K", -1)) != int(K):
        raise ValueError(
            f"Checkpoint condition count mismatch: {checkpoint.get('K')} != {K}"
        )
    if str(checkpoint.get("model_variant", "")) != str(cfg.MODEL_VARIANT):
        raise ValueError(
            "Checkpoint model-variant mismatch. Refusing weights from a "
            "different architecture."
        )

    checkpoint_conditions = [
        str(value) for value in checkpoint.get("condition_names", [])
    ]
    expected_conditions = [str(value) for value in condition_names]
    if checkpoint_conditions != expected_conditions:
        raise ValueError(
            "Checkpoint condition order mismatch. Refusing locked test evaluation."
        )

    checkpoint_binning = checkpoint.get("hazard_binning", {})
    expected_binning = {
        "BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS),
        "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS),
        "N_BINS": int(haz.N_BINS),
    }
    if checkpoint_binning != expected_binning:
        raise ValueError(
            "Checkpoint hazard-binning mismatch. Refusing locked test evaluation."
        )

    expected_architecture_source = ARCHITECTURE_SOURCE
    checkpoint_architecture_source = str(
        checkpoint.get("architecture_source", "")
    )
    if checkpoint_architecture_source != expected_architecture_source:
        raise ValueError(
            "Checkpoint architecture-source mismatch. Refusing locked test evaluation: "
            f"{checkpoint_architecture_source!r} != {expected_architecture_source!r}."
        )

    if checkpoint.get("delayed_entry_contract") != delayed_entry_contract():
        raise ValueError("Checkpoint delayed-entry contract mismatch; train a fresh matched run.")

    checkpoint_objective = checkpoint.get("objective", {})
    if checkpoint_objective.get("condition_identity_path") != bool(model.use_condition_identity_path):
        raise ValueError("Checkpoint condition-identity architecture setting differs from the active model.")
    if checkpoint_objective.get("reader_temporal_bias_mode") != FINAL_READER_TEMPORAL_BIAS_MODE:
        raise ValueError("Checkpoint reader-temporal mode differs from the active model.")
    if not bool(checkpoint_objective.get("shared_state_hazard_branch", False)):
        raise ValueError(
            "Checkpoint is not marked as using the direct shared-state hazard "
            "branch. Refusing locked evaluation."
        )

    training_origin = checkpoint.get("training_origin", None)
    if training_origin is not None:
        if not bool(training_origin.get("from_scratch", False)):
            raise ValueError(
                "A10 checkpoint is not marked as a from-scratch training run."
            )
        if bool(training_origin.get("external_checkpoint_loaded", True)):
            raise ValueError(
                "A10 checkpoint reports an external warm start; refusing locked evaluation."
            )
    else:
        # Backward compatibility for checkpoints produced by the first version of
        # this A10-on-locked-task notebook. Those checkpoints predate the explicit
        # training_origin field but already contain the locked input hashes, exact
        # condition order, model variant, hazard binning, and architecture source.
        # All of those fields have been validated above, and that notebook had no
        # external checkpoint-loading path before training.
        print(
            "Checkpoint predates explicit training_origin metadata; accepting the "
            "legacy A10 checkpoint after strict input/model/architecture validation."
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print("Loaded strict primary checkpoint for locked test evaluation:", path)
    return str(path)


checkpoint_used = maybe_load_best_checkpoint(model)

model.to(DEVICE)
model.eval()


# =============================================================================
# Extract locked test predictions
# =============================================================================
# A10 compatibility note: Cell 7 evaluates only outputs guaranteed by the
# locked A10 forward contract. Legacy V6/V7 reader-only diagnostics are not
# required for prediction or publication metrics and are intentionally omitted.


@torch.no_grad()
def extract_test_predictions():
    eids = []

    at_risk_all = []
    event_obs_all = []
    event_bin_all = []
    t_tilde_all = []

    C_months_all = []
    C_bin_all = []

    death_observed_post_index_all = []
    death_time_months_all = []
    death_bin_all = []

    risk_adj_by_h = {int(h): [] for h in EVAL_HORIZONS_MONTHS}
    risk_unadj_by_h = {int(h): [] for h in EVAL_HORIZONS_MONTHS}
    risk_death_by_h = {int(h): [] for h in EVAL_HORIZONS_MONTHS}

    start_bin = window_start_bin(ENTRY_DELAY_MONTHS)

    n_batches = 0

    for batch in test_loader:
        batch_cpu = batch
        batch_dev = move_batch_to_device(batch, DEVICE)

        out = model(batch_dev)

        B = int(batch_dev["C_months"].shape[0])

        if "eid" in batch_cpu:
            eids.extend([str(x) for x in batch_cpu["eid"]])
        else:
            # Fallback if dataloader does not carry eid.
            start = len(eids)
            eids.extend(
                test_samples.iloc[start : start + B]["eid"].astype(str).tolist()
            )

        at_risk_all.append(_safe_np(batch_cpu["at_risk"], np.float32))
        event_obs_all.append(_safe_np(batch_cpu["event_obs"], np.float32))
        event_bin_all.append(_safe_np(batch_cpu["event_bin"], np.int16))
        t_tilde_all.append(_safe_np(batch_cpu["t_tilde"], np.float32))

        C_months_all.append(_safe_np(batch_cpu["C_months"], np.float32))
        C_bin_all.append(_safe_np(batch_cpu["C_bin"], np.int16))

        death_observed_post_index_all.append(
            _safe_np(batch_cpu["death_observed_post_index"], np.float32)
        )
        death_time_months_all.append(
            _safe_np(batch_cpu["death_time_months"], np.float32)
        )
        death_bin_all.append(
            _safe_np(batch_cpu["death_bin"], np.int16)
        )

        for h in EVAL_HORIZONS_MONTHS:
            h = int(h)
            end_bin = month_to_haz_bin(ENTRY_DELAY_MONTHS + float(h))

            win = conditional_window_cif_local(
                p_condition=out["p_condition"],
                p_death_k=out["p_death_k"],
                p_neither=out["p_neither"],
                start_bin=start_bin,
                end_bin=end_bin,
            )

            risk_adj = win["window_cif_condition"]

            risk_unadj = unadjusted_window_risk_from_p_cond(
                p_cond_given_alive=out["p_cond_given_alive"],
                start_bin=start_bin,
                end_bin=end_bin,
            )

            risk_death = death_window_risk_from_p_death(
                p_death=out["p_death"],
                start_bin=start_bin,
                end_bin=end_bin,
            )

            risk_adj_by_h[h].append(_safe_np(risk_adj, np.float32))
            risk_unadj_by_h[h].append(_safe_np(risk_unadj, np.float32))
            risk_death_by_h[h].append(_safe_np(risk_death, np.float32))

        n_batches += 1

    pred = {
        "eid": np.asarray(eids, dtype=object),

        "at_risk": np.concatenate(at_risk_all, axis=0).astype(np.float32),
        "event_obs": np.concatenate(event_obs_all, axis=0).astype(np.float32),
        "event_bin": np.concatenate(event_bin_all, axis=0).astype(np.int16),
        "t_tilde": np.concatenate(t_tilde_all, axis=0).astype(np.float32),

        "C_months": np.concatenate(C_months_all, axis=0).astype(np.float32),
        "C_bin": np.concatenate(C_bin_all, axis=0).astype(np.int16),

        "death_observed_post_index": np.concatenate(death_observed_post_index_all, axis=0).astype(np.float32),
        "death_time_months": np.concatenate(death_time_months_all, axis=0).astype(np.float32),
        "death_bin": np.concatenate(death_bin_all, axis=0).astype(np.int16),

        "risk_adj_by_h": {
            int(h): np.concatenate(risk_adj_by_h[int(h)], axis=0).astype(np.float32)
            for h in EVAL_HORIZONS_MONTHS
        },
        "risk_unadj_by_h": {
            int(h): np.concatenate(risk_unadj_by_h[int(h)], axis=0).astype(np.float32)
            for h in EVAL_HORIZONS_MONTHS
        },
        "risk_death_by_h": {
            int(h): np.concatenate(risk_death_by_h[int(h)], axis=0).astype(np.float32)
            for h in EVAL_HORIZONS_MONTHS
        },
        "n_batches": int(n_batches),
    }

    return pred


print("\nExtracting locked test predictions...")
pred = extract_test_predictions()

N = int(pred["C_months"].shape[0])

print(
    "Prediction extraction complete:",
    {
        "N": N,
        "K": int(K),
        "n_batches": pred["n_batches"],
    },
)


# =============================================================================
# Save prediction arrays in Cell-8-compatible NPZ format
# =============================================================================

npz_payload = {
    "eid": pred["eid"],
    "condition_names": np.asarray(CONDITION_NAMES, dtype=object),
    "horizons_months": np.asarray(EVAL_HORIZONS_MONTHS, dtype=np.int32),
    "entry_delay_months": np.asarray(ENTRY_DELAY_MONTHS, dtype=np.float32),
    "gap_months": np.asarray(ENTRY_DELAY_MONTHS, dtype=np.float32),  # Legacy plotting alias for entry delay.
    "delayed_entry_schema_version": np.asarray(DELAYED_ENTRY_SCHEMA_VERSION),

    "at_risk": pred["at_risk"],
    "event_obs": pred["event_obs"],
    "event_bin": pred["event_bin"],
    "t_tilde": pred["t_tilde"],

    "C_months": pred["C_months"],
    "C_bin": pred["C_bin"],

    "death_observed_post_index": pred["death_observed_post_index"],
    "death_time_months": pred["death_time_months"],
    "death_bin": pred["death_bin"],
}

for h in EVAL_HORIZONS_MONTHS:
    h = int(h)
    npz_payload[f"risk_death_adjusted_h{h}"] = pred["risk_adj_by_h"][h]
    npz_payload[f"risk_unadjusted_h{h}"] = pred["risk_unadj_by_h"][h]
    npz_payload[f"risk_death_h{h}"] = pred["risk_death_by_h"][h]

np.savez_compressed(
    RUN_DIR / "test_prediction_arrays_death_adjusted.npz",
    **npz_payload,
)


# =============================================================================
# Point-estimate metric engine
# =============================================================================

def compute_metrics_for_indices(
    idx,
    include_per_condition=True,
    include_calibration=True,
    include_ctd=False,
    ctd_max_cases=1500,
    ctd_max_comparators=3000,
    rng=None,
):
    """
    Compute test metrics on a subset of rows.

    idx may contain duplicates, but this no-bootstrap cell uses it only for the
    full point-estimate set.
    """
    if rng is None:
        rng = np.random.default_rng(12345)

    idx = np.asarray(idx, dtype=int)

    micro_rows = []
    per_condition_rows = []
    calibration_rows = []
    calibration_error_rows = []
    death_rows = []
    summary_rows = []

    t_tilde = pred["t_tilde"][idx]
    at_risk = pred["at_risk"][idx] > 0.5
    C_months = pred["C_months"][idx]
    death_post = pred["death_observed_post_index"][idx] > 0.5
    death_time = pred["death_time_months"][idx]

    for h in EVAL_HORIZONS_MONTHS:
        h = int(h)

        R_by_type = {
            "death_adjusted": pred["risk_adj_by_h"][h][idx].astype(np.float64),
            "unadjusted": pred["risk_unadj_by_h"][h][idx].astype(np.float64),
        }

        R_death = pred["risk_death_by_h"][h][idx].astype(np.float64)

        obs = make_condition_window_observed_arrays(
            t_tilde=t_tilde,
            at_risk=at_risk,
            C_months=C_months,
            death_observed_post_index=death_post,
            death_time_months=death_time,
            horizon_months=float(h),
        )

        Y = obs["y"]
        known = obs["known"]
        eligible = obs["eligible"]
        status = obs["status"]
        time_rel = obs["time_rel"]
        censor_event = obs["censor_event"]

        # Precompute IPCW weights per condition.
        all_weights = np.zeros_like(Y, dtype=np.float64)

        for k in range(int(K)):
            all_weights[:, k] = ipcw_weights_for_binary_window(
                time_rel=time_rel[:, k],
                status=status[:, k],
                censor_event=censor_event[:, k],
                horizon_months=float(h),
                eligible=eligible[:, k],
                censoring_km=CELL7_CONDITION_CENSORING_KMS[k],
            )

        # ---------------------------------------------------------------------
        # Disease risk metrics: death-adjusted and unadjusted
        # ---------------------------------------------------------------------
        for risk_type, R in R_by_type.items():
            micro_discr = fixed_binary_metrics(
                y=Y.reshape(-1),
                pred=R.reshape(-1),
                mask=known.reshape(-1),
                weights=all_weights.reshape(-1),
            )

            micro_scores = weighted_brier_nbll(
                y=Y.reshape(-1),
                pred=R.reshape(-1),
                weights=(all_weights * known.astype(float)).reshape(-1),
                normalization_n=int(eligible.sum()),
            )

            micro_rows.append(
                {
                    "horizon_months": h,
                    "risk_type": risk_type,
                    **micro_discr,
                    **micro_scores,
                    "eligible_pairs": int(eligible.sum()),
                    "known_pairs": int(known.sum()),
                    "condition_events": int((status == 1).sum()),
                    "death_competing_events": int((status == 2).sum()),
                }
            )

            # Per-condition metrics.
            if include_per_condition:
                for k in range(int(K)):
                    known_k = known[:, k]
                    eligible_k = eligible[:, k]
                    y_k = Y[:, k]
                    r_k = R[:, k]
                    w_k = all_weights[:, k] * known_k.astype(float)

                    discr = fixed_binary_metrics(
                        y=y_k,
                        pred=r_k,
                        mask=known_k,
                        weights=all_weights[:, k],
                    )

                    scores = weighted_brier_nbll(
                        y=y_k,
                        pred=r_k,
                        weights=w_k,
                        normalization_n=int(eligible_k.sum()),
                    )

                    ctd = np.nan

                    per_condition_rows.append(
                        {
                            "horizon_months": h,
                            "risk_type": risk_type,
                            "condition_id": int(k),
                            "condition_name": CONDITION_NAMES[k],
                            "eligible_n": int(eligible_k.sum()),
                            "known_n": int(known_k.sum()),
                            "event_n": int(((status[:, k] == 1) & known_k).sum()),
                            "death_competing_n": int(((status[:, k] == 2) & known_k).sum()),
                            "admin_censored_n": int(((status[:, k] == 0) & known_k).sum()),
                            "mean_predicted_risk": float(np.nanmean(r_k[eligible_k])) if int(eligible_k.sum()) > 0 else np.nan,
                            "event_rate": float(np.mean(y_k[known_k])) if int(known_k.sum()) > 0 else np.nan,
                            **discr,
                            **scores,
                            "ctd_ipcw": ctd,
                            "ctd_status": "disabled_not_validly_implemented",
                        }
                    )

            # Aalen-Johansen calibration.
            if include_calibration:
                abs_errors = []
                abs_error_weights = []

                for k in range(int(K)):
                    eligible_k = eligible[:, k]

                    if int(eligible_k.sum()) < int(MIN_CAL_GROUP_N):
                        continue

                    r_k = R[:, k]
                    s = r_k[eligible_k].astype(np.float64)
                    t = time_rel[:, k][eligible_k].astype(np.float64)
                    st = status[:, k][eligible_k].astype(np.int16)

                    # Quantile bins. Robust to tied predictions.
                    edges = np.unique(np.quantile(s, np.linspace(0, 1, N_CAL_BINS + 1)))

                    if edges.size <= 2:
                        bin_id = np.zeros_like(s, dtype=np.int16)
                        n_bins_actual = 1
                    else:
                        bin_id = np.searchsorted(edges[1:-1], s, side="right").astype(np.int16)
                        n_bins_actual = int(bin_id.max()) + 1

                    for b in range(n_bins_actual):
                        m = bin_id == b

                        if int(m.sum()) < int(MIN_CAL_BIN_N):
                            continue

                        pred_mean = float(np.mean(s[m]))
                        obs_cif = aalen_johansen_cif_at_horizon(
                            time=t[m],
                            status=st[m],
                            horizon_months=float(h),
                        )

                        err = float(obs_cif - pred_mean) if np.isfinite(obs_cif) else np.nan

                        calibration_rows.append(
                            {
                                "horizon_months": h,
                                "risk_type": risk_type,
                                "condition_id": int(k),
                                "condition_name": CONDITION_NAMES[k],
                                "calibration_bin": int(b),
                                "n": int(m.sum()),
                                "predicted_mean": pred_mean,
                                "predicted_min": float(np.min(s[m])),
                                "predicted_max": float(np.max(s[m])),
                                "observed_aj_cif": obs_cif,
                                "observed_minus_predicted": err,
                            }
                        )

                        if np.isfinite(err):
                            abs_errors.append(abs(err))
                            abs_error_weights.append(int(m.sum()))

                if len(abs_errors) > 0:
                    ae = np.asarray(abs_errors, dtype=np.float64)
                    aw = np.asarray(abs_error_weights, dtype=np.float64)
                    calibration_error_rows.append(
                        {
                            "horizon_months": h,
                            "risk_type": risk_type,
                            "ici": float(np.average(ae, weights=aw)),
                            "e50": _weighted_quantile(ae, aw, 0.50),
                            "e90": _weighted_quantile(ae, aw, 0.90),
                            "n_calibration_bins": int(len(ae)),
                            "calibration_weight_n": int(aw.sum()),
                        }
                    )
                else:
                    calibration_error_rows.append(
                        {
                            "horizon_months": h,
                            "risk_type": risk_type,
                            "ici": np.nan,
                            "e50": np.nan,
                            "e90": np.nan,
                            "n_calibration_bins": 0,
                            "calibration_weight_n": 0,
                        }
                    )

        # ---------------------------------------------------------------------
        # Death-head metrics
        # ---------------------------------------------------------------------
        death_obs = make_death_window_observed_arrays(
            C_months=C_months,
            death_observed_post_index=death_post,
            death_time_months=death_time,
            horizon_months=float(h),
        )

        y_d = death_obs["y"]
        known_d = death_obs["known"]
        eligible_d = death_obs["eligible"]
        time_d = death_obs["time_rel"]
        status_d = death_obs["status"]
        censor_d = death_obs["censor_event"]

        w_d = ipcw_weights_for_binary_window(
            time_rel=time_d,
            status=status_d,
            censor_event=censor_d,
            horizon_months=float(h),
            eligible=eligible_d,
            censoring_km=CELL7_DEATH_CENSORING_KM,
        )

        death_discr = fixed_binary_metrics(
            y=y_d,
            pred=R_death,
            mask=known_d,
            weights=w_d,
        )

        death_scores = weighted_brier_nbll(
            y=y_d,
            pred=R_death,
            weights=w_d * known_d.astype(float),
            normalization_n=int(eligible_d.sum()),
        )

        death_ctd = np.nan

        death_rows.append(
            {
                "horizon_months": h,
                "eligible_n": int(eligible_d.sum()),
                "known_n": int(known_d.sum()),
                "death_event_n": int((y_d[known_d] == 1).sum()) if int(known_d.sum()) > 0 else 0,
                "mean_predicted_death_risk": float(np.nanmean(R_death[eligible_d])) if int(eligible_d.sum()) > 0 else np.nan,
                **death_discr,
                **death_scores,
                "ctd_ipcw": death_ctd,
                "ctd_status": "disabled_not_validly_implemented",
            }
        )

    micro_df = pd.DataFrame(micro_rows)
    per_condition_df = pd.DataFrame(per_condition_rows)
    calibration_df = pd.DataFrame(calibration_rows)
    calibration_error_df = pd.DataFrame(calibration_error_rows)
    death_df = pd.DataFrame(death_rows)

    # -------------------------------------------------------------------------
    # Publication summary
    # -------------------------------------------------------------------------
    for h in EVAL_HORIZONS_MONTHS:
        h = int(h)

        for risk_type in ["death_adjusted", "unadjusted"]:
            micro = micro_df[
                (micro_df["horizon_months"] == h)
                & (micro_df["risk_type"] == risk_type)
            ]

            sub = per_condition_df[
                (per_condition_df["horizon_months"] == h)
                & (per_condition_df["risk_type"] == risk_type)
            ]

            cal = calibration_error_df[
                (calibration_error_df["horizon_months"] == h)
                & (calibration_error_df["risk_type"] == risk_type)
            ]

            if len(micro):
                micro_row = micro.iloc[0].to_dict()
            else:
                micro_row = {}

            row = {
                "horizon_months": h,
                "risk_type": risk_type,
                "micro_auroc": _metric_float(micro_row.get("auroc", np.nan)),
                "micro_auprc": _metric_float(micro_row.get("auprc", np.nan)),
                "micro_brier_ipcw": _metric_float(micro_row.get("brier_ipcw", np.nan)),
                "micro_nbll_ipcw": _metric_float(micro_row.get("nbll_ipcw", np.nan)),
                "macro_auroc": float(sub["auroc"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_auprc": float(sub["auprc"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_brier_ipcw": float(sub["brier_ipcw"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_nbll_ipcw": float(sub["nbll_ipcw"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_ctd_ipcw": np.nan,
                "ctd_status": "disabled_not_validly_implemented",
                "mean_predicted_risk": float(sub["mean_predicted_risk"].mean(skipna=True)) if len(sub) else np.nan,
                "n_conditions_with_auroc": int(sub["auroc"].notna().sum()) if len(sub) else 0,
                "n_conditions": int(K),
                "ici": float(cal["ici"].iloc[0]) if len(cal) and "ici" in cal.columns else np.nan,
                "e50": float(cal["e50"].iloc[0]) if len(cal) and "e50" in cal.columns else np.nan,
                "e90": float(cal["e90"].iloc[0]) if len(cal) and "e90" in cal.columns else np.nan,
            }

            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Integrated mean over reported horizons.
    integrated_rows = []

    for risk_type in ["death_adjusted", "unadjusted"]:
        sub = summary_df[summary_df["risk_type"] == risk_type]

        integrated_rows.append(
            {
                "horizon_months": "integrated_mean_over_reported_horizons",
                "risk_type": risk_type,
                "micro_auroc": float(sub["micro_auroc"].mean(skipna=True)),
                "micro_auprc": float(sub["micro_auprc"].mean(skipna=True)),
                "micro_brier_ipcw": float(sub["micro_brier_ipcw"].mean(skipna=True)),
                "micro_nbll_ipcw": float(sub["micro_nbll_ipcw"].mean(skipna=True)),
                "macro_auroc": float(sub["macro_auroc"].mean(skipna=True)),
                "macro_auprc": float(sub["macro_auprc"].mean(skipna=True)),
                "macro_brier_ipcw": float(sub["macro_brier_ipcw"].mean(skipna=True)),
                "macro_nbll_ipcw": float(sub["macro_nbll_ipcw"].mean(skipna=True)),
                "macro_ctd_ipcw": np.nan,
                "ctd_status": "disabled_not_validly_implemented",
                "mean_predicted_risk": float(sub["mean_predicted_risk"].mean(skipna=True)),
                "n_conditions_with_auroc": int(sub["n_conditions_with_auroc"].mean(skipna=True)),
                "n_conditions": int(K),
                "ici": float(sub["ici"].mean(skipna=True)),
                "e50": float(sub["e50"].mean(skipna=True)),
                "e90": float(sub["e90"].mean(skipna=True)),
            }
        )

    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(integrated_rows),
        ],
        ignore_index=True,
    )

    return {
        "micro": micro_df,
        "per_condition": per_condition_df,
        "calibration": calibration_df,
        "calibration_error": calibration_error_df,
        "death": death_df,
        "summary": summary_df,
    }


# =============================================================================
# Compute locked point estimates
# =============================================================================

print("\nComputing locked test point estimates...")

point = compute_metrics_for_indices(
    idx=np.arange(N),
    include_per_condition=True,
    include_calibration=True,
    include_ctd=False,
    ctd_max_cases=EVAL_CTD_MAX_CASES,
    ctd_max_comparators=EVAL_CTD_MAX_COMPARATORS,
    rng=RNG,
)

micro_df = point["micro"]
per_condition_df = point["per_condition"]
calibration_df = point["calibration"]
calibration_error_df = point["calibration_error"]
death_df = point["death"]
summary_publication_df = point["summary"]

# Publication guardrail: calibration must be saved at 60 months in the same
# existing tables and with the same schemas.
for calibration_name, calibration_frame in [
    ("test_aj_calibration_by_horizon", calibration_df),
    ("test_calibration_error_summary", calibration_error_df),
]:
    if "horizon_months" not in calibration_frame.columns:
        raise RuntimeError(
            f"{calibration_name} is missing its existing horizon_months column."
        )
    if 60 not in set(
        pd.to_numeric(
            calibration_frame["horizon_months"], errors="coerce"
        ).dropna().astype(int)
    ):
        raise RuntimeError(
            f"{calibration_name} does not contain the required 60-month calibration."
        )


# =============================================================================
# Bootstrap CIs — disabled
# =============================================================================

print("\nBootstrap disabled. Saving no-CI compatibility files only.")

boot_replicates_df = pd.DataFrame()
boot_ci_df = pd.DataFrame()

summary_with_ci_df = summary_publication_df.copy()
death_with_ci_df = death_df.copy()

summary_with_ci_df["ci_computed"] = False
summary_with_ci_df["ci_lower"] = np.nan
summary_with_ci_df["ci_upper"] = np.nan

death_with_ci_df["ci_computed"] = False
death_with_ci_df["ci_lower"] = np.nan
death_with_ci_df["ci_upper"] = np.nan


# =============================================================================
# Save outputs using the same schema expected by Cell 8
# =============================================================================

micro_df.to_csv(
    RUN_DIR / "test_micro_metrics_by_horizon.csv",
    index=False,
)

per_condition_df.to_csv(
    RUN_DIR / "test_per_condition_metrics_by_horizon.csv",
    index=False,
)

death_df.to_csv(
    RUN_DIR / "test_death_head_metrics_by_horizon.csv",
    index=False,
)

calibration_df.to_csv(
    RUN_DIR / "test_aj_calibration_by_horizon.csv",
    index=False,
)

calibration_error_df.to_csv(
    RUN_DIR / "test_calibration_error_summary.csv",
    index=False,
)

summary_publication_df.to_csv(
    RUN_DIR / "test_summary_publication.csv",
    index=False,
)

summary_with_ci_df.to_csv(
    RUN_DIR / "test_summary_publication_with_ci.csv",
    index=False,
)

death_with_ci_df.to_csv(
    RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv",
    index=False,
)

boot_replicates_df.to_csv(
    RUN_DIR / "test_bootstrap_replicates_summary.csv",
    index=False,
)

boot_ci_df.to_csv(
    RUN_DIR / "test_bootstrap_summary_cis.csv",
    index=False,
)

# Backward-compatible filenames sometimes used by older notebooks.
micro_df.to_csv(
    RUN_DIR / "within_test_with_delayed_entry_death_adjusted.csv",
    index=False,
)

per_condition_df.to_csv(
    RUN_DIR / "condition_test_with_delayed_entry_death_adjusted.csv",
    index=False,
)


# =============================================================================
# Schema check for Cell 8 compatibility
# =============================================================================

required_files = {
    "micro": RUN_DIR / "test_micro_metrics_by_horizon.csv",
    "per_condition": RUN_DIR / "test_per_condition_metrics_by_horizon.csv",
    "death": RUN_DIR / "test_death_head_metrics_by_horizon.csv",
    "calibration": RUN_DIR / "test_aj_calibration_by_horizon.csv",
    "summary": RUN_DIR / "test_summary_publication.csv",
    "pred_arrays": RUN_DIR / "test_prediction_arrays_death_adjusted.npz",
}

missing_files = [str(p) for p in required_files.values() if not p.exists()]

if missing_files:
    raise FileNotFoundError(
        "Cell 7 failed to create required Cell 8 files:\n"
        + "\n".join(missing_files)
    )

required_summary_cols = {
    "horizon_months",
    "risk_type",
    "micro_auroc",
    "micro_auprc",
    "macro_auroc",
    "macro_auprc",
    "macro_ctd_ipcw",
    "macro_brier_ipcw",
    "macro_nbll_ipcw",
    "mean_predicted_risk",
    "n_conditions_with_auroc",
    "n_conditions",
}

required_death_cols = {
    "horizon_months",
    "eligible_n",
    "known_n",
    "death_event_n",
    "mean_predicted_death_risk",
    "auroc",
    "auprc",
    "ctd_ipcw",
    "brier_ipcw",
    "nbll_ipcw",
}

required_condition_cols = {
    "horizon_months",
    "risk_type",
    "condition_id",
    "condition_name",
    "known_n",
    "event_n",
    "death_competing_n",
    "admin_censored_n",
    "mean_predicted_risk",
    "event_rate",
    "auroc",
    "auprc",
    "ctd_ipcw",
    "brier_ipcw",
    "nbll_ipcw",
}

required_calibration_cols = {
    "horizon_months",
    "risk_type",
    "condition_id",
    "condition_name",
    "calibration_bin",
    "n",
    "predicted_mean",
    "predicted_min",
    "predicted_max",
    "observed_aj_cif",
    "observed_minus_predicted",
}

schema_checks = [
    ("summary", summary_publication_df, required_summary_cols),
    ("death", death_df, required_death_cols),
    ("per_condition", per_condition_df, required_condition_cols),
    ("calibration", calibration_df, required_calibration_cols),
]

for name, df_check, cols in schema_checks:
    missing_cols = sorted(cols - set(df_check.columns))

    if missing_cols:
        raise ValueError(f"{name} output missing required columns for Cell 8: {missing_cols}")

pred_npz_check = np.load(
    RUN_DIR / "test_prediction_arrays_death_adjusted.npz",
    allow_pickle=True,
)

required_npz_keys = {"condition_names"}

for h in EVAL_HORIZONS_MONTHS:
    h = int(h)
    required_npz_keys.update(
        {
            f"risk_death_adjusted_h{h}",
            f"risk_unadjusted_h{h}",
            f"risk_death_h{h}",
        }
    )

missing_npz_keys = sorted(required_npz_keys - set(pred_npz_check.files))

if missing_npz_keys:
    raise ValueError(f"Prediction NPZ missing Cell 8 keys: {missing_npz_keys}")


# =============================================================================
# Metadata
# =============================================================================

metadata = {
    "model_variant": str(getattr(cfg, "MODEL_VARIANT", "A10_death_adjusted")),
    "checkpoint_used": checkpoint_used,
    "n_test_rows": int(N),
    "K": int(K),
    "condition_names_n": int(len(CONDITION_NAMES)),
    "entry_delay_months": float(ENTRY_DELAY_MONTHS),
    "horizons_months": list(map(int, EVAL_HORIZONS_MONTHS)),
    "death_tie_policy": str(DEATH_TIE_POLICY),
    "bootstrap": {
        "enabled": False,
        "n": 0,
        "alpha": float(EVAL_BOOTSTRAP_ALPHA),
        "cluster": "eid",
        "include_ctd": False,
        "ctd_status": "disabled_not_validly_implemented",
        "censoring_km_source": "canonical_train_samples",
        "seed": int(EVAL_BOOTSTRAP_SEED),
        "ctd_max_cases_point": int(EVAL_CTD_MAX_CASES),
        "ctd_max_comparators_point": int(EVAL_CTD_MAX_COMPARATORS),
        "ctd_max_cases_bootstrap": 0,
        "ctd_max_comparators_bootstrap": 0,
    },
    "output_files": {
        "pred_arrays": str(RUN_DIR / "test_prediction_arrays_death_adjusted.npz"),
        "micro": str(RUN_DIR / "test_micro_metrics_by_horizon.csv"),
        "per_condition": str(RUN_DIR / "test_per_condition_metrics_by_horizon.csv"),
        "death": str(RUN_DIR / "test_death_head_metrics_by_horizon.csv"),
        "death_with_ci": str(RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv"),
        "calibration": str(RUN_DIR / "test_aj_calibration_by_horizon.csv"),
        "calibration_error": str(RUN_DIR / "test_calibration_error_summary.csv"),
        "summary": str(RUN_DIR / "test_summary_publication.csv"),
        "summary_with_ci": str(RUN_DIR / "test_summary_publication_with_ci.csv"),
        "bootstrap_replicates": str(RUN_DIR / "test_bootstrap_replicates_summary.csv"),
        "bootstrap_cis": str(RUN_DIR / "test_bootstrap_summary_cis.csv"),
    },
    "cell8_compatible_schema": True,
}

with open(RUN_DIR / "cell7_test_evaluation_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2, default=str)


# =============================================================================
# Print report
# =============================================================================

print("\nSaved Cell 7 artifacts:")
for _, p in metadata["output_files"].items():
    print(" ", p)

print(" ", RUN_DIR / "cell7_test_evaluation_metadata.json")

print("\nMain summary:")
print(summary_publication_df)

print("\nDeath-head metrics:")
print(death_df)

print(
    "\n✅ Cell 7 complete — locked test metrics generated with death-adjusted CIFs, "
    "unadjusted comparison risks, validation/test-coherent eligibility, "
    "Aalen-Johansen calibration, ICI/E50/E90, train-KM IPCW probability scores, "
    "Ctd explicitly disabled, shared death-head metrics, Cell-8-compatible output "
    "schema, and no-bootstrap point estimates."
)


# =============================================================================
# Notebook cell 20
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import average_precision_score, roc_auc_score


# =============================================================================
# Preconditions and locked configuration
# =============================================================================

required_globals = [
    "cfg",
    "haz",
    "RUN_DIR",
    "DEVICE",
    "model",
    "test_loader",
    "train_samples",
    "test_samples",
    "K",
    "maybe_load_best_checkpoint",
]
missing_globals = [
    name for name in required_globals if name not in globals()
]
if missing_globals:
    raise RuntimeError(
        "Run Cells 1–5 and Cell 7 before this cell. "
        f"Missing globals: {missing_globals}"
    )

RUN_DIR = Path(RUN_DIR).expanduser().resolve()
RUN_DIR.mkdir(parents=True, exist_ok=True)

ENTRY_DELAY_MONTHS_INTEGRATED = int(
    cfg.ENTRY_DELAY_MONTHS
)
REPORT_HORIZONS_MONTHS_INTEGRATED = tuple(
    int(value)
    for value in getattr(
        cfg,
        "EVAL_HORIZONS_MONTHS",
        (12, 24, 60, 120),
    )
)
BIN_WIDTH_MONTHS_INTEGRATED = int(haz.BIN_WIDTH_MONTHS)
INTEGRATION_START_MONTHS = int(
    getattr(cfg, "EVAL_INTEGRATION_START_MONTHS", 12)
)
INTEGRATION_END_MONTHS = int(
    getattr(
        cfg,
        "EVAL_INTEGRATION_END_MONTHS",
        max(REPORT_HORIZONS_MONTHS_INTEGRATED),
    )
)
CENSOR_SURVIVAL_FLOOR = float(
    getattr(cfg, "EVAL_CENSOR_SURVIVAL_FLOOR", 0.01)
)
TIME_EPS_INTEGRATED = 1e-8
PROB_EPS_INTEGRATED = float(
    getattr(cfg, "PROB_EPS", 1e-7)
)

DENSE_HORIZONS_MONTHS = tuple(
    range(
        INTEGRATION_START_MONTHS,
        INTEGRATION_END_MONTHS + 1,
        BIN_WIDTH_MONTHS_INTEGRATED,
    )
)

if INTEGRATION_START_MONTHS % BIN_WIDTH_MONTHS_INTEGRATED:
    raise ValueError(
        "Integration start must lie on a canonical bin boundary."
    )
if (
    ENTRY_DELAY_MONTHS_INTEGRATED + INTEGRATION_END_MONTHS
    > int(haz.MAX_HORIZON_MONTHS)
):
    raise ValueError(
        "Entry delay plus integration horizon exceeds the model axis."
    )
if DENSE_HORIZONS_MONTHS != tuple(range(12, 121, 6)):
    raise ValueError(
        "Expected the canonical dense grid 12, 18, ..., 120 months; "
        f"found {DENSE_HORIZONS_MONTHS}."
    )

if (
    "CONDITION_NAMES" in globals()
    and len(CONDITION_NAMES) == int(K)
):
    CONDITION_NAMES_INTEGRATED = [
        str(value) for value in CONDITION_NAMES
    ]
elif (
    "condition_names" in globals()
    and len(condition_names) == int(K)
):
    CONDITION_NAMES_INTEGRATED = [
        str(value) for value in condition_names
    ]
else:
    raise RuntimeError(
        "Cannot resolve the locked condition-name axis."
    )

CR_TARGET_CENSOR_INTEGRATED = int(
    getattr(cfg, "CR_TARGET_CENSOR", 0)
)
CR_TARGET_CONDITION_INTEGRATED = int(
    getattr(cfg, "CR_TARGET_CONDITION", 1)
)
CR_TARGET_DEATH_INTEGRATED = int(
    getattr(cfg, "CR_TARGET_DEATH", 2)
)


# =============================================================================
# Strict checkpoint reload
# =============================================================================

checkpoint_used_for_integrated = maybe_load_best_checkpoint(model)
model.to(DEVICE)
model.eval()


# =============================================================================
# Dense-grid prediction helpers
# =============================================================================

def integrated_month_to_bin(months):
    value = float(months)
    bin_index = (
        int(
            np.ceil(
                max(value, 0.0)
                / float(BIN_WIDTH_MONTHS_INTEGRATED)
            )
        )
        - 1
    )
    return int(
        max(0, min(bin_index, int(haz.N_BINS) - 1))
    )


def integrated_move_batch_to_device(batch, device):
    return {
        key: (
            value.to(device, non_blocking=True)
            if torch.is_tensor(value)
            else value
        )
        for key, value in batch.items()
    }


def integrated_window_cif(
    p_condition,
    p_death_k,
    p_neither,
    start_bin,
    end_bin,
):
    """
    Conditional competing-risk CIF over bins (start_bin, end_bin].
    """
    n_bins = int(p_condition.shape[-1])
    start = int(
        max(-1, min(int(start_bin), n_bins - 1))
    )
    end = int(
        max(0, min(int(end_bin), n_bins - 1))
    )

    if end <= start:
        return torch.zeros_like(p_condition[:, :, 0])

    p_condition_window = p_condition[
        :, :, start + 1 : end + 1
    ]
    p_neither_window = p_neither[
        :, :, start + 1 : end + 1
    ]

    survival_before = torch.cat(
        [
            torch.ones_like(
                p_neither_window[:, :, :1]
            ),
            torch.cumprod(
                p_neither_window[:, :, :-1],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    return torch.sum(
        survival_before * p_condition_window,
        dim=-1,
    ).clamp(0.0, 1.0)


@torch.no_grad()
def extract_dense_adjusted_predictions():
    eid_axis = []
    risk_chunks = {
        int(horizon): []
        for horizon in DENSE_HORIZONS_MONTHS
    }

    start_bin = window_start_bin(
        ENTRY_DELAY_MONTHS_INTEGRATED
    )

    model.eval()

    for batch in test_loader:
        batch_device = integrated_move_batch_to_device(
            batch,
            DEVICE,
        )
        outputs = model(batch_device)

        batch_eids = batch["eid"]
        eid_axis.extend(str(value) for value in batch_eids)

        for horizon in DENSE_HORIZONS_MONTHS:
            end_bin = integrated_month_to_bin(
                ENTRY_DELAY_MONTHS_INTEGRATED + int(horizon)
            )
            risk = integrated_window_cif(
                p_condition=outputs["p_condition"],
                p_death_k=outputs["p_death_k"],
                p_neither=outputs["p_neither"],
                start_bin=start_bin,
                end_bin=end_bin,
            )
            risk_chunks[int(horizon)].append(
                risk.detach().cpu().numpy().astype(
                    np.float32
                )
            )

    risk_cube = np.stack(
        [
            np.concatenate(
                risk_chunks[int(horizon)],
                axis=0,
            )
            for horizon in DENSE_HORIZONS_MONTHS
        ],
        axis=-1,
    ).astype(np.float32)

    return np.asarray(eid_axis, dtype=object), risk_cube


print(
    "Extracting V11 dense-grid test predictions:",
    DENSE_HORIZONS_MONTHS,
)
dense_eids, dense_adjusted_risk = (
    extract_dense_adjusted_predictions()
)

expected_test_eids = (
    test_samples["eid"].astype(str).to_numpy()
)
if dense_eids.astype(str).tolist() != expected_test_eids.tolist():
    raise AssertionError(
        "Dense predictions do not match canonical test-sample order."
    )

expected_dense_shape = (
    len(test_samples),
    int(K),
    len(DENSE_HORIZONS_MONTHS),
)
if dense_adjusted_risk.shape != expected_dense_shape:
    raise ValueError(
        "Dense risk cube has shape "
        f"{dense_adjusted_risk.shape}; expected "
        f"{expected_dense_shape}."
    )
if not np.isfinite(dense_adjusted_risk).all():
    raise ValueError(
        "Dense risk cube contains non-finite predictions."
    )
if np.any(
    (dense_adjusted_risk < -1e-7)
    | (dense_adjusted_risk > 1.0 + 1e-7)
):
    raise ValueError(
        "Dense risk cube contains values outside [0, 1]."
    )

dense_adjusted_risk = np.clip(
    dense_adjusted_risk,
    0.0,
    1.0,
)


# =============================================================================
# Canonical landmark competing-risk outcomes
# =============================================================================

def make_integrated_condition_outcomes(frame):
    """Canonical competing-risk outcomes conditional on delayed-entry eligibility."""
    return make_delayed_entry_outcomes_numpy(
        np.stack(frame["cr_at_risk"].to_numpy()),
        np.stack(frame["cr_event_type"].to_numpy()),
        np.stack(frame["cr_event_time_months"].to_numpy()),
        frame["C_months"].to_numpy(dtype=float),
    )


class IntegratedCensoringKM:
    """
    Kaplan-Meier estimator of administrative-censoring survival G.
    """

    def __init__(self, exit_time, censor_event):
        time = np.asarray(exit_time, dtype=float)
        event = np.asarray(
            censor_event,
            dtype=bool,
        )
        valid = np.isfinite(time) & (time >= 0)
        time = time[valid]
        event = event[valid]

        if time.size == 0:
            self.times = np.asarray([], dtype=float)
            self.survival_after = np.asarray(
                [],
                dtype=float,
            )
            return

        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order].astype(np.int64)

        unique, first, _ = np.unique(
            time,
            return_index=True,
            return_counts=True,
        )
        at_risk = time.size - first
        censor_counts = np.add.reduceat(
            event,
            first,
        )
        factors = (
            1.0
            - censor_counts
            / np.maximum(at_risk, 1)
        )

        self.times = unique
        self.survival_after = np.cumprod(
            np.clip(factors, 0.0, 1.0)
        )

    def _query(self, query, side, clip):
        query_array = np.asarray(query, dtype=float)
        indices = (
            np.searchsorted(
                self.times,
                query_array,
                side=side,
            )
            - 1
        )
        result = np.ones_like(
            query_array,
            dtype=float,
        )
        valid = indices >= 0
        result[valid] = self.survival_after[
            indices[valid]
        ]

        if clip:
            result = np.maximum(
                result,
                CENSOR_SURVIVAL_FLOOR,
            )
        return result

    def before_raw(self, query):
        return self._query(
            query,
            side="left",
            clip=False,
        )

    def before(self, query):
        return self._query(
            query,
            side="left",
            clip=True,
        )


def make_integrated_binary_window(
    eligible,
    status,
    exit_time,
    horizon,
    censoring_km,
):
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=float)
    horizon = float(horizon)

    case = (
        eligible
        & (status == 1)
        & (
            exit_time
            <= horizon + TIME_EPS_INTEGRATED
        )
    )
    competing = (
        eligible
        & (status == 2)
        & (
            exit_time
            <= horizon + TIME_EPS_INTEGRATED
        )
    )
    observed_through = (
        eligible
        & (
            exit_time
            >= horizon - TIME_EPS_INTEGRATED
        )
    )
    known = case | competing | observed_through
    y = case.astype(float)

    weights = np.zeros_like(
        exit_time,
        dtype=float,
    )
    event_or_competing = case | competing
    weights[event_or_competing] = (
        1.0
        / censoring_km.before(
            exit_time[event_or_competing]
        )
    )
    observed_control = (
        observed_through & (~event_or_competing)
    )
    weights[observed_control] = (
        1.0
        / float(censoring_km.before(horizon))
    )

    return {
        "eligible": eligible,
        "case": case,
        "competing": competing,
        "observed_through": observed_through,
        "known": known,
        "y": y,
        "weights": weights,
    }


def integrated_aalen_johansen_cif(
    exit_time,
    status,
    horizon,
):
    """
    Aalen-Johansen CIF for condition status=1 with death status=2.
    """
    time = np.asarray(exit_time, dtype=float)
    state = np.asarray(status, dtype=np.int8)
    valid = np.isfinite(time) & (time >= 0)
    time = time[valid]
    state = state[valid]

    if time.size == 0:
        return np.nan

    order = np.argsort(time, kind="mergesort")
    time = time[order]
    state = state[order]

    unique, first, _ = np.unique(
        time,
        return_index=True,
        return_counts=True,
    )
    at_risk = time.size - first
    primary = np.add.reduceat(
        (state == 1).astype(np.int64),
        first,
    )
    all_events = np.add.reduceat(
        (state != 0).astype(np.int64),
        first,
    )

    event_mask = (
        (unique <= float(horizon) + TIME_EPS_INTEGRATED)
        & (all_events > 0)
    )
    if not np.any(event_mask):
        return 0.0

    hazards_all = (
        all_events / np.maximum(at_risk, 1)
    )
    survival_after = np.cumprod(
        np.clip(
            1.0 - hazards_all,
            0.0,
            1.0,
        )
    )
    survival_before = np.concatenate(
        ([1.0], survival_after[:-1])
    )
    increments = (
        survival_before
        * primary
        / np.maximum(at_risk, 1)
    )
    return float(np.sum(increments[event_mask]))


# =============================================================================
# Canonical metric primitives
# =============================================================================

def integrated_ipcw_ht_scores(
    y,
    prediction,
    weights,
    eligible_n,
):
    y = np.asarray(y, dtype=float)
    prediction = np.clip(
        np.asarray(prediction, dtype=float),
        PROB_EPS_INTEGRATED,
        1.0 - PROB_EPS_INTEGRATED,
    )
    weights = np.asarray(weights, dtype=float)

    valid = (
        np.isfinite(y)
        & np.isfinite(prediction)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if eligible_n <= 0 or not valid.any():
        return {
            "brier_ipcw_ht": np.nan,
            "nbll_ipcw_ht": np.nan,
        }

    brier = (
        np.sum(
            weights[valid]
            * (
                y[valid]
                - prediction[valid]
            )
            ** 2
        )
        / int(eligible_n)
    )
    nbll = (
        -np.sum(
            weights[valid]
            * (
                y[valid]
                * np.log(prediction[valid])
                + (
                    1.0 - y[valid]
                )
                * np.log1p(
                    -prediction[valid]
                )
            )
        )
        / int(eligible_n)
    )

    return {
        "brier_ipcw_ht": float(brier),
        "nbll_ipcw_ht": float(nbll),
    }


def integrated_competing_risk_auc(
    window,
    prediction,
):
    known = np.asarray(
        window["known"],
        dtype=bool,
    )
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(
        window["weights"],
        dtype=float,
    )
    prediction = np.asarray(
        prediction,
        dtype=float,
    )

    valid = (
        known
        & np.isfinite(prediction)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if (
        not valid.any()
        or np.unique(y[valid].astype(int)).size < 2
    ):
        return {
            "auc_ipcw_cr": np.nan,
            "auprc_ipcw_cr": np.nan,
        }

    return {
        "auc_ipcw_cr": float(
            roc_auc_score(
                y[valid],
                prediction[valid],
                sample_weight=weights[valid],
            )
        ),
        "auprc_ipcw_cr": float(
            average_precision_score(
                y[valid],
                prediction[valid],
                sample_weight=weights[valid],
            )
        ),
    }


def integrate_dense_metric(values, horizons):
    values_array = np.asarray(values, dtype=float)
    horizon_array = np.asarray(
        horizons,
        dtype=float,
    )
    valid = (
        np.isfinite(values_array)
        & np.isfinite(horizon_array)
    )

    if valid.sum() < 2:
        return np.nan

    x = horizon_array[valid]
    y = values_array[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])

    if span <= 0:
        return np.nan

    trapezoid = getattr(
        np,
        "trapezoid",
        np.trapz,
    )
    return float(trapezoid(y, x) / span)


# =============================================================================
# Per-condition dense-grid evaluation and integration
# =============================================================================

train_outcomes_integrated = (
    make_integrated_condition_outcomes(train_samples)
)
test_outcomes_integrated = (
    make_integrated_condition_outcomes(test_samples)
)

integrated_rows = []
dense_point_rows = []

for condition_id in range(int(K)):
    train_selected = train_outcomes_integrated[
        "eligible"
    ][:, condition_id]
    test_selected = test_outcomes_integrated[
        "eligible"
    ][:, condition_id]

    censoring_km = IntegratedCensoringKM(
        exit_time=train_outcomes_integrated[
            "exit_time"
        ][train_selected, condition_id],
        censor_event=(
            train_outcomes_integrated[
                "status"
            ][train_selected, condition_id]
            == CR_TARGET_CENSOR_INTEGRATED
        ),
    )

    test_status = test_outcomes_integrated[
        "status"
    ][:, condition_id]
    test_exit_time = test_outcomes_integrated[
        "exit_time"
    ][:, condition_id]
    train_status = train_outcomes_integrated[
        "status"
    ][:, condition_id]
    train_exit_time = train_outcomes_integrated[
        "exit_time"
    ][:, condition_id]

    eligible_n = int(test_selected.sum())

    brier_values = []
    null_brier_values = []
    nbll_values = []
    auc_values = []
    calibration_values = []

    for grid_index, horizon in enumerate(
        DENSE_HORIZONS_MONTHS
    ):
        prediction = dense_adjusted_risk[
            :,
            condition_id,
            grid_index,
        ].astype(float)

        window = make_integrated_binary_window(
            eligible=test_selected,
            status=test_status,
            exit_time=test_exit_time,
            horizon=float(horizon),
            censoring_km=censoring_km,
        )
        scores = integrated_ipcw_ht_scores(
            y=window["y"],
            prediction=prediction,
            weights=window["weights"],
            eligible_n=eligible_n,
        )
        auc = integrated_competing_risk_auc(
            window=window,
            prediction=prediction,
        )

        train_null_risk = (
            integrated_aalen_johansen_cif(
                exit_time=train_exit_time[
                    train_selected
                ],
                status=train_status[
                    train_selected
                ],
                horizon=float(horizon),
            )
        )
        null_prediction = np.full_like(
            prediction,
            train_null_risk,
            dtype=float,
        )
        null_scores = integrated_ipcw_ht_scores(
            y=window["y"],
            prediction=null_prediction,
            weights=window["weights"],
            eligible_n=eligible_n,
        )

        observed_risk = integrated_aalen_johansen_cif(
            exit_time=test_exit_time[test_selected],
            status=test_status[test_selected],
            horizon=float(horizon),
        )
        mean_prediction = (
            float(np.mean(prediction[test_selected]))
            if eligible_n > 0
            else np.nan
        )
        absolute_calibration_error = (
            abs(mean_prediction - observed_risk)
            if (
                np.isfinite(mean_prediction)
                and np.isfinite(observed_risk)
            )
            else np.nan
        )

        brier_values.append(
            scores["brier_ipcw_ht"]
        )
        null_brier_values.append(
            null_scores["brier_ipcw_ht"]
        )
        nbll_values.append(
            scores["nbll_ipcw_ht"]
        )
        auc_values.append(
            auc["auc_ipcw_cr"]
        )
        calibration_values.append(
            absolute_calibration_error
        )

        dense_point_rows.append(
            {
                "model_name": (
                    str(cfg.MODEL_VARIANT)
                ),
                "risk_type": "death_adjusted",
                "condition_id": int(condition_id),
                "condition_name": (
                    CONDITION_NAMES_INTEGRATED[
                        condition_id
                    ]
                ),
                "horizon_months": int(horizon),
                "eligible_n": eligible_n,
                "known_n": int(
                    window["known"].sum()
                ),
                "event_n": int(
                    window["case"].sum()
                ),
                "death_competing_n": int(
                    window["competing"].sum()
                ),
                "mean_predicted_cif": (
                    mean_prediction
                ),
                "observed_aj_cif": observed_risk,
                "absolute_calibration_error": (
                    absolute_calibration_error
                ),
                "auc_ipcw_cr": (
                    auc["auc_ipcw_cr"]
                ),
                "auprc_ipcw_cr": (
                    auc["auprc_ipcw_cr"]
                ),
                "brier_ipcw_ht": (
                    scores["brier_ipcw_ht"]
                ),
                "brier_null_ipcw_ht": (
                    null_scores["brier_ipcw_ht"]
                ),
                "nbll_ipcw_ht": (
                    scores["nbll_ipcw_ht"]
                ),
            }
        )

    ibs = integrate_dense_metric(
        brier_values,
        DENSE_HORIZONS_MONTHS,
    )
    ibs_null = integrate_dense_metric(
        null_brier_values,
        DENSE_HORIZONS_MONTHS,
    )

    integrated_rows.append(
        {
            "model_name": (
                str(cfg.MODEL_VARIANT)
            ),
            "risk_type": "death_adjusted",
            "condition_id": int(condition_id),
            "condition_name": (
                CONDITION_NAMES_INTEGRATED[
                    condition_id
                ]
            ),
            "integration_start_months": int(
                DENSE_HORIZONS_MONTHS[0]
            ),
            "integration_end_months": int(
                DENSE_HORIZONS_MONTHS[-1]
            ),
            "grid_step_months": int(
                BIN_WIDTH_MONTHS_INTEGRATED
            ),
            "n_grid_points": int(
                len(DENSE_HORIZONS_MONTHS)
            ),
            "ibs_ipcw_ht": ibs,
            "ibs_null_ipcw_ht": ibs_null,
            "scaled_ibs_ipcw_ht": (
                1.0 - ibs / ibs_null
                if (
                    np.isfinite(ibs)
                    and np.isfinite(ibs_null)
                    and ibs_null > 0
                )
                else np.nan
            ),
            "inbll_ipcw_ht": integrate_dense_metric(
                nbll_values,
                DENSE_HORIZONS_MONTHS,
            ),
            "iauc_ipcw_cr": integrate_dense_metric(
                auc_values,
                DENSE_HORIZONS_MONTHS,
            ),
            "integrated_abs_calibration_error": (
                integrate_dense_metric(
                    calibration_values,
                    DENSE_HORIZONS_MONTHS,
                )
            ),
        }
    )


integrated_metrics_df = pd.DataFrame(
    integrated_rows
)
dense_point_metrics_df = pd.DataFrame(
    dense_point_rows
)


# =============================================================================
# Validate and save
# =============================================================================

required_integrated_columns = {
    "model_name",
    "risk_type",
    "condition_id",
    "condition_name",
    "integration_start_months",
    "integration_end_months",
    "grid_step_months",
    "n_grid_points",
    "ibs_ipcw_ht",
    "ibs_null_ipcw_ht",
    "scaled_ibs_ipcw_ht",
    "inbll_ipcw_ht",
    "iauc_ipcw_cr",
    "integrated_abs_calibration_error",
}
missing_integrated_columns = sorted(
    required_integrated_columns
    - set(integrated_metrics_df.columns)
)
if missing_integrated_columns:
    raise RuntimeError(
        "Integrated table is missing columns: "
        f"{missing_integrated_columns}"
    )

if len(integrated_metrics_df) != int(K):
    raise RuntimeError(
        "Expected exactly one death-adjusted integrated row "
        f"per condition ({K}); found "
        f"{len(integrated_metrics_df)}."
    )

if (
    integrated_metrics_df[
        "condition_name"
    ].astype(str).tolist()
    != CONDITION_NAMES_INTEGRATED
):
    raise AssertionError(
        "Integrated-table condition order changed."
    )

if not integrated_metrics_df[
    "risk_type"
].astype(str).eq("death_adjusted").all():
    raise AssertionError(
        "Integrated table contains an unexpected risk type."
    )

if not integrated_metrics_df[
    "n_grid_points"
].eq(len(DENSE_HORIZONS_MONTHS)).all():
    raise AssertionError(
        "Integrated table has the wrong dense-grid size."
    )

INTEGRATED_OUTPUT_PATH = (
    RUN_DIR
    / "test_integrated_metrics_by_condition.csv"
)
DENSE_POINT_OUTPUT_PATH = (
    RUN_DIR
    / "test_dense_metrics_by_condition_and_horizon.csv"
)
STANDARD_POINT_OUTPUT_PATH = (
    RUN_DIR
    / "test_per_condition_metrics_by_horizon.csv"
)
DENSE_PREDICTION_OUTPUT_PATH = (
    RUN_DIR
    / "test_dense_prediction_arrays_for_integrated_metrics.npz"
)
INTEGRATED_METADATA_PATH = (
    RUN_DIR
    / "test_integrated_metrics_metadata.json"
)

integrated_temp_path = (
    INTEGRATED_OUTPUT_PATH.with_suffix(".tmp.csv")
)
dense_point_temp_path = (
    DENSE_POINT_OUTPUT_PATH.with_suffix(".tmp.csv")
)

integrated_metrics_df.to_csv(
    integrated_temp_path,
    index=False,
)
dense_point_metrics_df.to_csv(
    dense_point_temp_path,
    index=False,
)
integrated_temp_path.replace(
    INTEGRATED_OUTPUT_PATH
)
dense_point_temp_path.replace(
    DENSE_POINT_OUTPUT_PATH
)

# Update the standard point table in place while preserving all legacy
# columns and both risk types. Canonical columns and aliases are replaced only
# for the primary death-adjusted rows at the four locked report horizons.
if not STANDARD_POINT_OUTPUT_PATH.is_file():
    raise FileNotFoundError(
        "Cell 7 did not create the standard per-condition point table: "
        f"{STANDARD_POINT_OUTPUT_PATH}"
    )

standard_point_df = pd.read_csv(
    STANDARD_POINT_OUTPUT_PATH
)
point_keys = [
    "condition_id",
    "condition_name",
    "horizon_months",
    "risk_type",
]
canonical_point_columns = [
    "eligible_n",
    "known_n",
    "event_n",
    "death_competing_n",
    "auc_ipcw_cr",
    "auprc_ipcw_cr",
    "brier_ipcw_ht",
    "brier_null_ipcw_ht",
    "nbll_ipcw_ht",
]
missing_standard_keys = sorted(
    set(point_keys) - set(standard_point_df.columns)
)
if missing_standard_keys:
    raise ValueError(
        "Standard point table is missing merge keys: "
        f"{missing_standard_keys}"
    )

canonical_report_df = dense_point_metrics_df[
    dense_point_metrics_df[
        "horizon_months"
    ].isin(REPORT_HORIZONS_MONTHS_INTEGRATED)
].copy()
canonical_report_df = canonical_report_df[
    point_keys + canonical_point_columns
].copy()
canonical_report_df = canonical_report_df.rename(
    columns={
        column: f"_canonical_{column}"
        for column in canonical_point_columns
    }
)

standard_point_df = standard_point_df.merge(
    canonical_report_df,
    on=point_keys,
    how="left",
    validate="many_to_one",
)
primary_rows = (
    standard_point_df["risk_type"]
    .astype(str)
    .eq("death_adjusted")
    & standard_point_df[
        "horizon_months"
    ].isin(REPORT_HORIZONS_MONTHS_INTEGRATED)
)

for column in canonical_point_columns:
    source = f"_canonical_{column}"
    if column not in standard_point_df.columns:
        standard_point_df[column] = np.nan
    standard_point_df.loc[
        primary_rows,
        column,
    ] = standard_point_df.loc[
        primary_rows,
        source,
    ].to_numpy()
    standard_point_df = standard_point_df.drop(
        columns=[source]
    )

alias_map = {
    "auroc": "auc_ipcw_cr",
    "auprc": "auprc_ipcw_cr",
    "brier_ipcw": "brier_ipcw_ht",
    "nbll_ipcw": "nbll_ipcw_ht",
}
for alias, canonical in alias_map.items():
    if alias not in standard_point_df.columns:
        standard_point_df[alias] = np.nan
    standard_point_df.loc[
        primary_rows,
        alias,
    ] = standard_point_df.loc[
        primary_rows,
        canonical,
    ].to_numpy()

if standard_point_df.loc[
    primary_rows,
    "auc_ipcw_cr",
].isna().all():
    raise RuntimeError(
        "Canonical point AUROC was not merged into the standard table."
    )

standard_point_temp_path = (
    STANDARD_POINT_OUTPUT_PATH.with_suffix(
        ".canonical.tmp.csv"
    )
)
standard_point_df.to_csv(
    standard_point_temp_path,
    index=False,
)
standard_point_temp_path.replace(
    STANDARD_POINT_OUTPUT_PATH
)

np.savez_compressed(
    DENSE_PREDICTION_OUTPUT_PATH,
    eid=dense_eids.astype(object),
    condition_names=np.asarray(
        CONDITION_NAMES_INTEGRATED,
        dtype=object,
    ),
    dense_horizons_months=np.asarray(
        DENSE_HORIZONS_MONTHS,
        dtype=np.int32,
    ),
    entry_delay_months=np.asarray(ENTRY_DELAY_MONTHS_INTEGRATED, dtype=np.float32),
    gap_months=np.asarray(ENTRY_DELAY_MONTHS_INTEGRATED, dtype=np.float32),  # Legacy plotting alias.
    delayed_entry_schema_version=np.asarray(DELAYED_ENTRY_SCHEMA_VERSION),
    risk_death_adjusted=dense_adjusted_risk.astype(
        np.float32
    ),
)

integrated_metadata = {
    "schema_version": (
        "ehr-competing-risk-canonical-integrated-evaluation.v1"
    ),
    "model_name": (
        str(cfg.MODEL_VARIANT)
    ),
    "risk_type": "death_adjusted",
    "checkpoint": str(
        checkpoint_used_for_integrated
    ),
    "entry_delay_months": int(
        ENTRY_DELAY_MONTHS_INTEGRATED
    ),
    "dense_horizons_months": [
        int(value)
        for value in DENSE_HORIZONS_MONTHS
    ],
    "integration_method": (
        "trapezoidal mean over the canonical six-month grid"
    ),
    "censoring_distribution": (
        "condition-specific administrative-censoring "
        "Kaplan-Meier estimated from canonical train_samples"
    ),
    "censor_survival_floor": float(
        CENSOR_SURVIVAL_FLOOR
    ),
    "estimand": (
        "Condition incidence during (6, 6+horizon] months from the original index, "
        "before death, conditional on being alive, condition-free, and observed "
        "strictly beyond the six-month delayed-entry landmark. Horizons are months after entry."
    ),
    "delayed_entry_contract": delayed_entry_contract(),
    "n_test_participants": int(
        len(test_samples)
    ),
    "n_conditions": int(K),
    "output_files": {
        "integrated_metrics": str(
            INTEGRATED_OUTPUT_PATH
        ),
        "dense_point_metrics": str(
            DENSE_POINT_OUTPUT_PATH
        ),
        "standard_point_metrics": str(
            STANDARD_POINT_OUTPUT_PATH
        ),
        "dense_predictions": str(
            DENSE_PREDICTION_OUTPUT_PATH
        ),
    },
}
with open(
    INTEGRATED_METADATA_PATH,
    "w",
) as handle:
    json.dump(
        integrated_metadata,
        handle,
        indent=2,
        default=str,
    )

# Final read-back validation of the exact Figure-1 file.
saved_integrated = pd.read_csv(
    INTEGRATED_OUTPUT_PATH
)
if len(saved_integrated) != int(K):
    raise RuntimeError(
        "Saved integrated table failed row-count validation."
    )
if not required_integrated_columns.issubset(
    saved_integrated.columns
):
    raise RuntimeError(
        "Saved integrated table failed schema validation."
    )

saved_standard_point = pd.read_csv(
    STANDARD_POINT_OUTPUT_PATH
)
required_standard_canonical = {
    "auc_ipcw_cr",
    "auprc_ipcw_cr",
    "brier_ipcw_ht",
    "nbll_ipcw_ht",
    "auroc",
    "auprc",
    "brier_ipcw",
    "nbll_ipcw",
}
if not required_standard_canonical.issubset(
    saved_standard_point.columns
):
    raise RuntimeError(
        "Saved standard point table failed canonical schema validation."
    )

print("=" * 88)
print("V11 canonical point/integrated metrics complete")
print("=" * 88)
print("Checkpoint:", checkpoint_used_for_integrated)
print("Dense horizons:", DENSE_HORIZONS_MONTHS)
print("Conditions:", len(saved_integrated))
print("Integrated table:", INTEGRATED_OUTPUT_PATH)
print("Dense point audit:", DENSE_POINT_OUTPUT_PATH)
print("Canonical standard point table:", STANDARD_POINT_OUTPUT_PATH)
print("Dense prediction audit:", DENSE_PREDICTION_OUTPUT_PATH)
print("Metadata:", INTEGRATED_METADATA_PATH)
print(
    saved_integrated[
        [
            "condition_name",
            "iauc_ipcw_cr",
            "scaled_ibs_ipcw_ht",
            "integrated_abs_calibration_error",
        ]
    ].head(10)
)
