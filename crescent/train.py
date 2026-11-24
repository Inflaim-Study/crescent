"""Stage 4: training (disease NLL + 0.10 x death NLL), EMA weights, checkpoint selection.

Extracted from the CRESCENT_Final notebook (cell 12). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 12
# =============================================================================

import json
import math
import time
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score

assert "model" in globals() and "MODEL_INPUT_HASHES" in globals(), "Run Cells 1-5 first."
RUN_DIR = Path(RUN_DIR)
DEVICE = torch.device(getattr(cfg, "DEVICE", "cuda:0"))
if str(DEVICE) != "cuda:0" or torch.cuda.current_device() != 0:
    raise RuntimeError("Training is locked to the scheduler-visible cuda:0 device.")

# -----------------------------------------------------------------------------
# Settings. Training objective: 1.0 * disease NLL + 0.10 * death NLL (+ optional time-smoothness, off by default).
# Primary checkpoint: minimum validation joint NLL (same weighted combination).
# -----------------------------------------------------------------------------
CR_TARGET_CENSOR = int(getattr(cfg, "CR_TARGET_CENSOR", 0))
CR_TARGET_CONDITION = int(getattr(cfg, "CR_TARGET_CONDITION", 1))
CR_TARGET_DEATH = int(getattr(cfg, "CR_TARGET_DEATH", 2))
ENTRY_DELAY_MONTHS = float(cfg.ENTRY_DELAY_MONTHS)
RATE_EPS = float(getattr(cfg, "RATE_EPS", 1e-8))
PROB_EPS = float(getattr(cfg, "PROB_EPS", 1e-7))
# Validation discrimination monitor: IPCW macro AUROC at 24 and 60 months after entry.
VAL_AUROC_HORIZONS_MONTHS = (24, 60)
VAL_AUROC_INTERVAL = 5000        # AUROC / top-bottom-5 evaluation every 5,000 updates (NLL + checkpointing stay every VALIDATION_INTERVAL)
VAL_MIN_CASES_FOR_RANKING = 20   # a condition needs >= this many validation cases at BOTH horizons to be ranked
VAL_TOP_BOTTOM_N = 5
if not set(VAL_AUROC_HORIZONS_MONTHS).issubset(set(int(x) for x in cfg.EVAL_HORIZONS_MONTHS)):
    raise ValueError("Validation AUROC horizons must be among EVAL_HORIZONS_MONTHS.")
LOSS_ALPHA_NLL = float(cfg.LOSS_ALPHA_NLL)
LOSS_GAMMA_DEATH = float(cfg.LOSS_GAMMA_DEATH)
if LOSS_ALPHA_NLL != 1.0 or LOSS_GAMMA_DEATH != 0.10:
    raise ValueError("Training loss must be 1.0*disease_nll + 0.10*death_nll.")
NORMALISE_DISEASE_CR_NLL_BY_CONDITION = bool(
    getattr(cfg, "NORMALISE_DISEASE_CR_NLL_BY_CONDITION", False)
)
EVAL_HORIZONS_MONTHS = tuple(int(x) for x in cfg.EVAL_HORIZONS_MONTHS)
GRAD_CLIP = float(cfg.GRAD_CLIP)
EMA_DECAY = float(cfg.EMA_DECAY)
USE_EMA = EMA_DECAY > 0.0
TIME_SMOOTHNESS_WEIGHT = float(cfg.TIME_SMOOTHNESS_WEIGHT)
MAX_ITERATIONS = int(cfg.MAX_ITERATIONS)
VALIDATION_INTERVAL = int(cfg.VALIDATION_INTERVAL)
LOG_INTERVAL = int(cfg.LOG_INTERVAL)
WARMUP_ITERATIONS = int(cfg.WARMUP_ITERATIONS)
PATIENCE_VALIDATIONS = int(cfg.PATIENCE_VALIDATIONS)
MIN_ITERATIONS_BEFORE_EARLY_STOP = int(cfg.MIN_ITERATIONS_BEFORE_EARLY_STOP)
MIN_DELTA = float(cfg.MIN_DELTA)
PRIMARY_CHECKPOINT_NAME = Path(str(cfg.PRIMARY_CHECKPOINT)).name  # best_model_by_val_loss.pt


def move_batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def month_to_haz_bin(months):
    m = float(months)
    b = int(np.ceil(max(m, 0.0) / float(haz.BIN_WIDTH_MONTHS)) - 1)
    return max(0, min(b, int(haz.N_BINS) - 1))


def _piecewise_exposure_months(stop_months, n_bins, bin_width_months):
    width = float(bin_width_months)
    starts = torch.arange(int(n_bins), device=stop_months.device, dtype=stop_months.dtype) * width
    return (stop_months.unsqueeze(-1) - starts).clamp(min=0.0, max=width)


def _event_bin_from_exact_months(event_time_months, n_bins, bin_width_months):
    width = float(bin_width_months)
    safe = torch.where(torch.isfinite(event_time_months), event_time_months, torch.zeros_like(event_time_months))
    return (torch.ceil(safe.clamp_min(0.0) / width).long() - 1).clamp(0, int(n_bins) - 1)


def disease_competing_risk_nll(condition_rate, cr_event_type, cr_event_time_months, cr_at_risk, C_months):
    """Condition cause-specific PE NLL conditional on eligibility at entry."""
    B, Kloc, T = condition_rate.shape
    entry = float(cfg.ENTRY_DELAY_MONTHS)
    eps = ENTRY_TIME_EPS_MONTHS
    max_horizon = float(T) * float(haz.BIN_WIDTH_MONTHS)
    types = cr_event_type.long()
    times = cr_event_time_months.to(condition_rate.dtype)
    active = cr_at_risk.float() > 0.5
    finite_event = torch.isfinite(times)
    pair_event = finite_event & ((types == CR_TARGET_CONDITION) | (types == CR_TARGET_DEATH))
    censor = C_months.to(condition_rate.dtype)[:, None].expand(B, Kloc).clamp(0.0, max_horizon)
    before_entry = pair_event & (times <= entry + eps)
    eligible = active & torch.isfinite(C_months)[:, None] & (censor > entry + eps) & (~before_entry)
    observed_pair_event = eligible & pair_event & (times > entry + eps) & (times <= censor + eps) & (times <= max_horizon + eps)
    safe_event_time = torch.where(finite_event, times, censor).clamp(0.0, max_horizon)
    stop = torch.where(observed_pair_event, torch.minimum(safe_event_time, censor), censor)
    entry_times = torch.full_like(stop, entry)
    exposure = (_piecewise_exposure_months(stop, T, haz.BIN_WIDTH_MONTHS) - _piecewise_exposure_months(entry_times, T, haz.BIN_WIDTH_MONTHS)).clamp_min(0.0)
    exposure = torch.where(eligible.unsqueeze(-1), exposure, torch.zeros_like(exposure))
    cumulative_hazard = (condition_rate * exposure).sum(dim=-1)
    event_bin = _event_bin_from_exact_months(times, T, haz.BIN_WIDTH_MONTHS)
    event_rate = condition_rate.gather(-1, event_bin.unsqueeze(-1)).squeeze(-1)
    observed_condition = observed_pair_event & (types == CR_TARGET_CONDITION)
    nll_pair = cumulative_hazard - observed_condition.float() * torch.log(event_rate.clamp_min(RATE_EPS))
    if not bool(eligible.any()):
        loss = condition_rate.sum() * 0.0
    elif NORMALISE_DISEASE_CR_NLL_BY_CONDITION:
        per_person = (nll_pair * eligible.float()).sum(dim=1) / eligible.float().sum(dim=1).clamp_min(1.0)
        loss = per_person[eligible.any(dim=1)].mean()
    else:
        loss = nll_pair[eligible].mean()
    with torch.no_grad():
        return loss, {
            "n_active_pairs": int(eligible.sum().item()),
            "n_baseline_active_pairs": int(active.sum().item()),
            "n_excluded_at_entry_pairs": int((active & (~eligible)).sum().item()),
            "n_condition_pairs": int(observed_condition.sum().item()),
            "n_death_pairs": int((observed_pair_event & (types == CR_TARGET_DEATH)).sum().item()),
            "n_censor_pairs": int((eligible & (~observed_pair_event)).sum().item()),
            "exact_exposure_months": float(exposure.sum().item()),
        }


def direct_death_nll(death_rate, death_observed_post_index, death_time_months, C_months):
    """One death PE contribution per participant eligible at delayed entry."""
    B, T = death_rate.shape
    entry = float(cfg.ENTRY_DELAY_MONTHS)
    eps = ENTRY_TIME_EPS_MONTHS
    max_horizon = float(T) * float(haz.BIN_WIDTH_MONTHS)
    times = death_time_months.to(death_rate.dtype)
    censor = C_months.to(death_rate.dtype).clamp(0.0, max_horizon)
    death_event = (death_observed_post_index.float() > 0.5) & torch.isfinite(times)
    eligible = torch.isfinite(C_months) & (censor > entry + eps) & (~(death_event & (times <= entry + eps)))
    observed = eligible & death_event & (times > entry + eps) & (times <= censor + eps) & (times <= max_horizon + eps)
    safe = torch.where(torch.isfinite(times), times, censor).clamp(0.0, max_horizon)
    stop = torch.where(observed, torch.minimum(safe, censor), censor)
    exposure = (_piecewise_exposure_months(stop, T, haz.BIN_WIDTH_MONTHS) - _piecewise_exposure_months(torch.full_like(stop, entry), T, haz.BIN_WIDTH_MONTHS)).clamp_min(0.0)
    exposure = torch.where(eligible.unsqueeze(-1), exposure, torch.zeros_like(exposure))
    cumulative_hazard = (death_rate * exposure).sum(dim=-1)
    event_bin = _event_bin_from_exact_months(times, T, haz.BIN_WIDTH_MONTHS)
    event_rate = death_rate.gather(-1, event_bin.view(B, 1)).squeeze(-1)
    nll = cumulative_hazard - observed.float() * torch.log(event_rate.clamp_min(RATE_EPS))
    loss = nll[eligible].mean() if bool(eligible.any()) else death_rate.sum() * 0.0
    return loss, {
        "n_eligible_participants": int(eligible.sum().item()),
        "n_excluded_at_entry_participants": int((~eligible).sum().item()),
        "n_death_events": int(observed.sum().item()),
        "n_death_censored": int((eligible & (~observed)).sum().item()),
        "exact_exposure_months": float(exposure.sum().item()),
    }


def validate_delayed_entry_likelihood():
    """CPU checks of exact exposure, exclusion, event boundaries, and gradients."""
    T = int(haz.N_BINS)
    rate = torch.full((7, 1, T), 0.01, dtype=torch.float64, requires_grad=True)
    types = torch.tensor([[1], [2], [1], [0], [2], [0], [0]])
    times = torch.tensor([[3.], [6.], [12.], [float("inf")], [9.], [float("inf")], [float("inf")]], dtype=torch.float64)
    censor = torch.tensor([24., 24., 24., 18., 24., 5., 6.], dtype=torch.float64)
    loss, diag = disease_competing_risk_nll(rate, types, times, torch.ones(7, 1), censor)
    expected = (0.01 * 21.0 - math.log(0.01)) / 3.0
    assert math.isclose(float(loss.detach()), expected, rel_tol=1e-12)
    assert diag["n_active_pairs"] == 3 and diag["exact_exposure_months"] == 21.0
    loss.backward()
    assert torch.count_nonzero(rate.grad[:, :, 0]).item() == 0
    assert torch.count_nonzero(rate.grad[[0, 1, 5, 6]]).item() == 0
    assert math.isclose(float(rate.grad[2, 0, 1]), (6.0 - 100.0) / 3.0, rel_tol=1e-12)

    death_rate = torch.full((6, T), 0.02, dtype=torch.float64, requires_grad=True)
    death_times = torch.tensor([3., 6., 12., float("inf"), float("inf"), float("inf")], dtype=torch.float64)
    death_censor = torch.tensor([24., 24., 24., 18., 5., 6.], dtype=torch.float64)
    death_loss, death_diag = direct_death_nll(death_rate, torch.tensor([1., 1., 1., 0., 0., 0.]), death_times, death_censor)
    assert math.isclose(float(death_loss.detach()), (0.02 * 18.0 - math.log(0.02)) / 2.0, rel_tol=1e-12)
    assert death_diag["n_eligible_participants"] == 2 and death_diag["exact_exposure_months"] == 18.0
    death_loss.backward()
    assert torch.count_nonzero(death_rate.grad[:, 0]).item() == 0
    assert torch.count_nonzero(death_rate.grad[[0, 1, 4, 5]]).item() == 0

    zero_rate = torch.full((1, 1, T), 0.01, dtype=torch.float64, requires_grad=True)
    zero_loss, _ = disease_competing_risk_nll(zero_rate, torch.tensor([[1]]), torch.tensor([[6.]], dtype=torch.float64), torch.ones(1, 1), torch.tensor([24.], dtype=torch.float64))
    zero_loss.backward()
    assert float(zero_loss.detach()) == 0.0 and torch.count_nonzero(zero_rate.grad).item() == 0
    print("Delayed-entry likelihood checks passed: month-six exclusions, post-entry exposure, and zero pre-entry gradients.")


validate_delayed_entry_likelihood()


def total_loss(out, batch):
    """1.0 * disease NLL + 0.10 * death NLL (exact piecewise-exponential, post-entry)."""
    disease_nll, disease_diag = disease_competing_risk_nll(
        out["condition_rate"], batch["cr_event_type"], batch["cr_event_time_months"],
        batch["cr_at_risk"], batch["C_months"],
    )
    death_nll, death_diag = direct_death_nll(
        out["death_rate"], batch["death_observed_post_index"],
        batch["death_time_months"], batch["C_months"],
    )
    total = LOSS_ALPHA_NLL * disease_nll + LOSS_GAMMA_DEATH * death_nll
    smoothness = None
    if TIME_SMOOTHNESS_WEIGHT > 0.0 and torch.is_grad_enabled():
        _resid = out["baseline_logits"] + out["history_logits"]            # (B, K, T)
        smoothness = (_resid[..., 2:] - 2.0 * _resid[..., 1:-1] + _resid[..., :-2]).square().mean()
        total = total + TIME_SMOOTHNESS_WEIGHT * smoothness
    shared = out["shared_state_logits"].detach()
    parts = {
        "total": float(total.detach().item()),
        "disease_cr_nll": float(disease_nll.detach().item()),
        "death_nll": float(death_nll.detach().item()),
        "shared_state_log_rate_mean": float(shared.mean().item()),
        "shared_state_log_rate_abs_mean": float(shared.abs().mean().item()),
        "shared_state_log_rate_rms": float(shared.square().mean().sqrt().item()),
        "condition_log_rate_saturation_fraction": float(
            out["condition_log_rate_saturation_fraction"].detach().item()
        ),
        "time_smoothness_penalty": float(smoothness.detach().item()) if smoothness is not None else 0.0,
    }
    parts.update({f"disease_{k}": v for k, v in disease_diag.items()})
    parts.update({f"death_{k}": v for k, v in death_diag.items()})
    return total, parts


@torch.no_grad()
def evaluate_validation_loss(loader):
    """Validation joint NLL, weighted exactly as in training."""
    model.eval()
    disease_num = 0.0; disease_den = 0
    death_num = 0.0; death_den = 0
    shared_sum = 0.0; shared_abs_sum = 0.0; shared_sq_sum = 0.0; shared_n = 0
    saturation_num = 0.0; saturation_den = 0
    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        out = model(batch)
        _, parts = total_loss(out, batch)
        n_pairs = int(parts["disease_n_active_pairs"])
        disease_num += float(parts["disease_cr_nll"]) * n_pairs
        disease_den += n_pairs
        n_death = int(parts["death_n_eligible_participants"])
        death_num += float(parts["death_nll"]) * n_death
        death_den += n_death
        shared = out["shared_state_logits"].detach().double()
        shared_sum += float(shared.sum().item())
        shared_abs_sum += float(shared.abs().sum().item())
        shared_sq_sum += float(shared.square().sum().item())
        shared_n += int(shared.numel())
        n_logits = int(out["condition_log_rate_unclamped"].numel())
        saturation_num += float(out["condition_log_rate_saturation_fraction"].item()) * n_logits
        saturation_den += n_logits
    d = disease_num / max(disease_den, 1)
    de = death_num / max(death_den, 1)
    return {
        "val_disease_cr_nll": float(d),
        "val_death_nll": float(de),
        "val_loss": float(LOSS_ALPHA_NLL * d + LOSS_GAMMA_DEATH * de),
        "val_shared_state_log_rate_mean": float(shared_sum / max(shared_n, 1)),
        "val_shared_state_log_rate_abs_mean": float(shared_abs_sum / max(shared_n, 1)),
        "val_shared_state_log_rate_rms": float(np.sqrt(shared_sq_sum / max(shared_n, 1))),
        "val_condition_log_rate_saturation_fraction": float(
            saturation_num / max(saturation_den, 1)
        ),
    }


# -----------------------------------------------------------------------------
# Validation discrimination: IPCW competing-risk AUROC at 24 and 60 months
# (death-adjusted condition CIF over the post-entry window; censoring KMs are
# fitted on the training split only, exactly as in the locked test evaluation).
# -----------------------------------------------------------------------------
VALIDATION_TIME_EPS = 1e-8
VALIDATION_CENSOR_SURVIVAL_FLOOR = float(cfg.EVAL_CENSOR_SURVIVAL_FLOOR)


def make_condition_outcomes(frame):
    """Competing-risk outcomes conditional on delayed-entry eligibility."""
    return make_delayed_entry_outcomes_numpy(
        np.stack(frame["cr_at_risk"].to_numpy()),
        np.stack(frame["cr_event_type"].to_numpy()),
        np.stack(frame["cr_event_time_months"].to_numpy()),
        frame["C_months"].to_numpy(dtype=float),
    )


class ValidationCensoringKM:
    """Training-split Kaplan-Meier estimate of censoring survival, G(t-)."""

    def __init__(self, exit_time, censor_event):
        t = np.asarray(exit_time, dtype=float)
        e = np.asarray(censor_event, dtype=bool)
        valid = np.isfinite(t) & (t >= 0.0)
        t, e = t[valid], e[valid]
        if t.size == 0:
            self.times = np.asarray([], dtype=float)
            self.survival_after = np.asarray([], dtype=float)
            return
        order = np.argsort(t, kind="mergesort")
        t, e = t[order], e[order].astype(np.int64)
        unique, first = np.unique(t, return_index=True)
        at_risk = t.size - first
        censor_count = np.add.reduceat(e, first)
        self.times = unique
        self.survival_after = np.cumprod(np.clip(1.0 - censor_count / np.maximum(at_risk, 1), 0.0, 1.0))

    def before(self, query):
        q = np.asarray(query, dtype=float)
        idx = np.searchsorted(self.times, q, side="left") - 1
        out = np.ones_like(q, dtype=float)
        ok = idx >= 0
        out[ok] = self.survival_after[idx[ok]]
        return np.maximum(out, VALIDATION_CENSOR_SURVIVAL_FLOOR)


def make_validation_binary_window(eligible, status, exit_time, horizon, censoring_km):
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=float)
    horizon = float(horizon)
    case = eligible & (status == 1) & (exit_time <= horizon + VALIDATION_TIME_EPS)
    competing = eligible & (status == 2) & (exit_time <= horizon + VALIDATION_TIME_EPS)
    observed_through = eligible & (exit_time >= horizon - VALIDATION_TIME_EPS)
    known = case | competing | observed_through
    weights = np.zeros_like(exit_time, dtype=float)
    event_or_competing = case | competing
    weights[event_or_competing] = 1.0 / censoring_km.before(exit_time[event_or_competing])
    controls = observed_through & (~event_or_competing)
    weights[controls] = 1.0 / float(censoring_km.before(horizon))
    return {"case": case, "known": known, "y": case.astype(float), "weights": weights}


VALIDATION_TRAIN_OUTCOMES = make_condition_outcomes(train_samples)
VALIDATION_OUTCOMES = make_condition_outcomes(val_samples)
VALIDATION_CONDITION_CENSORING_KMS = [
    ValidationCensoringKM(
        VALIDATION_TRAIN_OUTCOMES["exit_time"][VALIDATION_TRAIN_OUTCOMES["eligible"][:, k], k],
        VALIDATION_TRAIN_OUTCOMES["status"][VALIDATION_TRAIN_OUTCOMES["eligible"][:, k], k] == 0,
    )
    for k in range(int(K))
]


@torch.no_grad()
def collect_validation_risks(loader, horizons_months):
    """Death-adjusted post-entry condition CIFs for every validation participant."""
    model.eval()
    horizons = tuple(int(h) for h in horizons_months)
    collected = {h: [] for h in horizons}
    start_bin = window_start_bin(ENTRY_DELAY_MONTHS)
    for batch in loader:
        batch = move_batch_to_device(batch, DEVICE)
        out = model(batch)
        for h in horizons:
            end_bin = month_to_haz_bin(ENTRY_DELAY_MONTHS + float(h))
            win = conditional_window_cif_from_probs(
                out["p_condition"], out["p_death_k"], out["p_neither"], start_bin, end_bin
            )
            collected[h].append(win["window_cif_condition"].cpu().numpy().astype(np.float64))
    return {h: np.concatenate(collected[h], axis=0) for h in horizons}


def validation_per_condition_auroc(loader):
    """One row per condition: IPCW AUROC and case counts at 24 and 60 months."""
    risks = collect_validation_risks(loader, VAL_AUROC_HORIZONS_MONTHS)
    rows = []
    for k in range(int(K)):
        row = {"condition_id": int(k), "condition_name": str(condition_names[k])}
        for h in VAL_AUROC_HORIZONS_MONTHS:
            window = make_validation_binary_window(
                VALIDATION_OUTCOMES["eligible"][:, k], VALIDATION_OUTCOMES["status"][:, k],
                VALIDATION_OUTCOMES["exit_time"][:, k], float(h), VALIDATION_CONDITION_CENSORING_KMS[k],
            )
            risk = np.clip(risks[h][:, k], PROB_EPS, 1.0 - PROB_EPS)
            valid = window["known"] & np.isfinite(risk) & np.isfinite(window["weights"]) & (window["weights"] > 0.0)
            auc = np.nan
            if int(valid.sum()) >= 2 and np.unique(window["y"][valid].astype(int)).size == 2:
                auc = float(roc_auc_score(window["y"][valid], risk[valid], sample_weight=window["weights"][valid]))
            row[f"n_cases_{h}m"] = int(window["case"].sum())
            row[f"auc_{h}m"] = auc
        rows.append(row)
    frame = pd.DataFrame(rows)
    auc_columns = [f"auc_{h}m" for h in VAL_AUROC_HORIZONS_MONTHS]
    frame["auc_mean_24_60"] = np.where(
        np.isfinite(frame[auc_columns]).all(axis=1), frame[auc_columns].mean(axis=1), np.nan
    )
    return frame


def validation_macro_auroc_summary(per_condition):
    """Macro AUROC (mean over conditions with a computable AUROC) at each horizon."""
    summary = {}
    for h in VAL_AUROC_HORIZONS_MONTHS:
        values = per_condition[f"auc_{h}m"]
        summary[f"val_macro_auc_{h}m"] = float(values.mean(skipna=True)) if values.notna().any() else np.nan
        summary[f"val_n_macro_conditions_{h}m"] = int(values.notna().sum())
    summary["val_macro_auc_mean_24_60"] = float(
        np.nanmean([summary[f"val_macro_auc_{h}m"] for h in VAL_AUROC_HORIZONS_MONTHS])
    )
    return summary


def top_bottom_conditions(per_condition, n=VAL_TOP_BOTTOM_N, min_cases=VAL_MIN_CASES_FOR_RANKING):
    """Top-n and bottom-n conditions by mean of the 24- and 60-month AUROC."""
    ok = per_condition["auc_mean_24_60"].notna()
    for h in VAL_AUROC_HORIZONS_MONTHS:
        ok &= per_condition[f"n_cases_{h}m"] >= int(min_cases)
    ranked = per_condition[ok].sort_values("auc_mean_24_60", ascending=False).reset_index(drop=True)
    top = ranked.head(n).copy(); top.insert(0, "group", "top")
    bottom = ranked.tail(n).sort_values("auc_mean_24_60").copy(); bottom.insert(0, "group", "bottom")
    top.insert(1, "rank", np.arange(1, len(top) + 1))
    bottom.insert(1, "rank", np.arange(1, len(bottom) + 1))
    return pd.concat([top, bottom], ignore_index=True), int(ok.sum())


# -----------------------------------------------------------------------------
# Optimiser, schedule (linear warm-up then cosine decay), checkpoints
# -----------------------------------------------------------------------------
BASE_LR = float(cfg.LR)
MIN_LR = BASE_LR * float(cfg.SCHEDULER_MIN_LR_RATIO)


def learning_rate_for_iteration(zero_based_iteration):
    i = int(zero_based_iteration)
    if i < WARMUP_ITERATIONS:
        return BASE_LR * float(i + 1) / max(1, WARMUP_ITERATIONS)
    if i >= MAX_ITERATIONS:
        return MIN_LR
    decay_ratio = float(i - WARMUP_ITERATIONS) / max(1, MAX_ITERATIONS - WARMUP_ITERATIONS)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (BASE_LR - MIN_LR)


if bool(cfg.WEIGHT_DECAY_EXEMPT_1D):
    # REVISION: no decay on norm gains/biases, log-rate intercepts and scalar gates (ndim < 2); decaying the
    # intercepts pulls the baseline log-rate toward 0, i.e. a rate of one event per month.
    _decay = [p for p in model.parameters() if p.ndim >= 2]
    _no_decay = [p for p in model.parameters() if p.ndim < 2]
    assert len(_decay) + len(_no_decay) == len(list(model.parameters()))
    optimizer = torch.optim.AdamW(
        [{"params": _decay, "weight_decay": float(cfg.WEIGHT_DECAY)},
         {"params": _no_decay, "weight_decay": 0.0}],
        lr=BASE_LR,
    )
else:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=BASE_LR, weight_decay=float(cfg.WEIGHT_DECAY)
    )

# -----------------------------------------------------------------------------
# Exponential moving average of the weights. Validation, checkpoint selection (best_model_by_val_loss.pt,
# last_model.pt) and therefore the locked test evaluation all use the EMA weights; the raw-weight validation
# NLL is logged as val_loss_raw_weights for comparison.
# -----------------------------------------------------------------------------
from contextlib import contextmanager, nullcontext

ema_params = [p.detach().clone() for p in model.parameters()] if USE_EMA else []


@torch.no_grad()
def ema_update(iteration):
    if not USE_EMA:
        return
    decay = min(EMA_DECAY, (1.0 + float(iteration)) / (10.0 + float(iteration)))
    for e, p in zip(ema_params, model.parameters()):
        e.lerp_(p.detach(), 1.0 - decay)


@contextmanager
def ema_weights_in_model():
    """Temporarily load the EMA weights into `model` (no-op when EMA is disabled)."""
    if not USE_EMA:
        yield
        return
    backup = [p.detach().clone() for p in model.parameters()]
    with torch.no_grad():
        for p, e in zip(model.parameters(), ema_params):
            p.copy_(e)
    try:
        yield
    finally:
        with torch.no_grad():
            for p, b in zip(model.parameters(), backup):
                p.copy_(b)

best_loss_path = RUN_DIR / PRIMARY_CHECKPOINT_NAME
last_path = RUN_DIR / "last_model.pt"


def save_checkpoint(path, iteration, epoch, extra=None):
    torch.save(
        {
            "iteration": int(iteration),
            "global_step": int(iteration),
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": {
                "type": "linear_warmup_cosine_by_iteration",
                "last_iteration": int(iteration),
                "max_iterations": int(MAX_ITERATIONS),
                "warmup_iterations": int(WARMUP_ITERATIONS),
                "base_lr": float(BASE_LR),
                "min_lr": float(MIN_LR),
            },
            "K": int(K),
            "condition_names": [str(x) for x in condition_names],
            "model_variant": str(cfg.MODEL_VARIANT),
            "delayed_entry_contract": delayed_entry_contract(),
            "model_input_hashes": dict(MODEL_INPUT_HASHES),
            "hazard_binning": {
                "BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS),
                "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS),
                "N_BINS": int(haz.N_BINS),
            },
            "architecture_source": ARCHITECTURE_SOURCE,
            "training_origin": {"from_scratch": True, "external_checkpoint_loaded": False},
            "training_protocol": {
                "unit": "optimizer_iteration",
                "max_iterations": int(MAX_ITERATIONS),
                "validation_interval": int(VALIDATION_INTERVAL),
                "patience_validations": int(PATIENCE_VALIDATIONS),
                "validation_participants": int(len(val_samples)),
            },
            "objective": {
                "exact_piecewise_exponential_nll": True,
                "alpha_nll": LOSS_ALPHA_NLL,
                "gamma_death": LOSS_GAMMA_DEATH,
                "condition_identity_path": bool(model.use_condition_identity_path),
                "reader_temporal_bias_mode": FINAL_READER_TEMPORAL_BIAS_MODE,
                "shared_state_hazard_branch": True,
                "shared_state_source": "cls_vec",
                "shared_state_broadcast": "equal_weight_for_all_conditions",
                "crp_value_embedding": "linear_value_direction_plus_piecewise_linear_residual",
                "shared_state_final_projection_zero_initialised": True,
                "condition_residual_centering": False,
                "primary_checkpoint_selection": "joint_nll",
            },
            "extra": {} if extra is None else dict(extra),
        },
        path,
    )


def train_one_iteration(batch, iteration):
    model.train()
    learning_rate = learning_rate_for_iteration(int(iteration) - 1)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    batch = move_batch_to_device(batch, DEVICE)
    optimizer.zero_grad(set_to_none=True)
    out = model(batch)
    loss, parts = total_loss(out, batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss at iteration {iteration}: {loss}")
    loss.backward()

    shared_grad_sq = torch.zeros((), device=DEVICE)
    for parameter in model.shared_state_head.parameters():
        if parameter.grad is not None:
            shared_grad_sq = shared_grad_sq + parameter.grad.detach().square().sum()
    shared_state_grad_norm = float(shared_grad_sq.sqrt().item())
    if int(iteration) == 1 and int(parts.get("disease_n_active_pairs", 0)) > 0 and not (
        np.isfinite(shared_state_grad_norm) and shared_state_grad_norm > 0.0
    ):
        raise RuntimeError(
            "The zero-initialised shared-state head received no gradient on the "
            f"first iteration: {shared_state_grad_norm}."
        )
    parts["shared_state_grad_norm"] = shared_state_grad_norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optimizer.step()

    row = {
        "iteration": int(iteration),
        "loss": float(loss.detach()),
        "grad_norm": float(grad_norm.detach()),
        "lr": float(learning_rate),
    }
    row.update(parts)
    return row


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------
FULL_CONSOLE_LOG_PATH = RUN_DIR / "cell6_full_training_console.log"
ITERATION_LOG_PATH = RUN_DIR / "iteration_log_every_100.csv"
_console_log_handle = open(FULL_CONSOLE_LOG_PATH, mode="w", buffering=1, encoding="utf-8")


def log_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)
    print(*args, file=_console_log_handle, flush=True,
          **{k: v for k, v in kwargs.items() if k in {"sep", "end"}})


def write_csv_atomically(frame, path):
    path = Path(path)
    temporary_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary_path, index=False)
    temporary_path.replace(path)


MEAN_KEYS = [
    "loss", "disease_cr_nll", "death_nll", "shared_state_log_rate_mean",
    "shared_state_log_rate_abs_mean", "shared_state_log_rate_rms",
    "shared_state_grad_norm", "condition_log_rate_saturation_fraction", "grad_norm",
]

log_print("Architecture:", cfg.MODEL_VARIANT)
log_print("Objective: 1.0 * disease NLL + 0.10 * death NLL")
log_print("Revision settings:", {
    "batch_size": int(cfg.BATCH_SIZE), "lr": BASE_LR, "weight_decay": float(cfg.WEIGHT_DECAY),
    "wd_exempt_1d": bool(cfg.WEIGHT_DECAY_EXEMPT_1D), "ema_decay": EMA_DECAY,
    "dropout": float(cfg.DROPOUT_ENCODER), "event_mask_prob": float(cfg.EVENT_MASK_PROB),
    "n_heads": int(cfg.N_HEADS), "time_smoothness_weight": TIME_SMOOTHNESS_WEIGHT,
    "attained_age_term": bool(cfg.USE_ATTAINED_AGE_TERM),
    "static_cont_input_layernorm": bool(cfg.STATIC_CONT_INPUT_LAYERNORM),
    "reader_temporal_bias": bool(cfg.USE_READER_TEMPORAL_BIAS), "seed": int(cfg.SEED),
})
log_print("Primary checkpoint: minimum validation joint NLL ->", best_loss_path.name)
log_print(
    "Schedule:",
    {"max_iterations": MAX_ITERATIONS, "warmup_iterations": WARMUP_ITERATIONS,
     "validation_interval": VALIDATION_INTERVAL, "patience_validations": PATIENCE_VALIDATIONS},
)

if len(train_loader) < 1:
    raise RuntimeError("Training loader is empty.")

# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
history = []
iteration_log_rows = []
best_val_loss = np.inf
best_loss_iteration = -1
best_loss_epoch = -1
validations_without_improvement = 0
train_iterator = iter(train_loader)
current_epoch = 1
window_rows = []
training_start_time = time.time()

for iteration in range(1, MAX_ITERATIONS + 1):
    try:
        batch = next(train_iterator)
    except StopIteration:
        current_epoch += 1
        train_iterator = iter(train_loader)
        batch = next(train_iterator)

    train_row = train_one_iteration(batch, iteration)
    ema_update(iteration)
    window_rows.append(train_row)

    if iteration % LOG_INTERVAL == 0:
        recent = pd.DataFrame(window_rows[-LOG_INTERVAL:])
        log_row = {"iteration": int(iteration), "epoch": int(current_epoch)}
        log_row.update({k: float(recent[k].mean()) for k in MEAN_KEYS})
        log_row["lr"] = float(train_row["lr"])
        log_row["elapsed_minutes"] = float((time.time() - training_start_time) / 60.0)
        iteration_log_rows.append(log_row)
        write_csv_atomically(pd.DataFrame(iteration_log_rows), ITERATION_LOG_PATH)
        log_print(
            f"iteration {iteration:06d}/{MAX_ITERATIONS:06d} | epoch~{current_epoch:03d} "
            f"| loss={log_row['loss']:.4f} | disease={log_row['disease_cr_nll']:.4f} "
            f"| death={log_row['death_nll']:.4f} | lr={log_row['lr']:.3e} "
            f"| elapsed={log_row['elapsed_minutes']:.1f} min"
        )

    if not (iteration % VALIDATION_INTERVAL == 0 or iteration == MAX_ITERATIONS):
        continue

    window = pd.DataFrame(window_rows)
    window_rows = []
    auroc_due = (iteration % VAL_AUROC_INTERVAL == 0) or (iteration == MAX_ITERATIONS)
    val_auroc_summary = {}
    if auroc_due:
        with ema_weights_in_model():
            val_per_condition = validation_per_condition_auroc(val_loader)
        val_auroc_summary = validation_macro_auroc_summary(val_per_condition)
        val_top_bottom_i, _ = top_bottom_conditions(val_per_condition)
        val_per_condition.to_csv(RUN_DIR / f"validation_per_condition_auroc_iteration_{iteration:06d}.csv", index=False)
        val_top_bottom_i.to_csv(RUN_DIR / f"validation_top5_bottom5_iteration_{iteration:06d}.csv", index=False)
        log_print(f"\nValidation AUROC at iteration {iteration:06d}: {val_auroc_summary}")
        log_print(val_top_bottom_i[["group", "rank", "condition_name", "auc_24m", "auc_60m", "auc_mean_24_60"]].to_string(index=False))
    with ema_weights_in_model():
        _val_loss_row = evaluate_validation_loss(val_loader)
    row = {
        "iteration": int(iteration),
        "epoch": int(current_epoch),
        **{f"train_{k}": float(window[k].mean()) for k in MEAN_KEYS},
        **_val_loss_row,
        **val_auroc_summary,
        "lr": float(optimizer.param_groups[0]["lr"]),
        "elapsed_seconds": float(time.time() - training_start_time),
    }
    if USE_EMA:
        row["val_loss_raw_weights"] = float(evaluate_validation_loss(val_loader)["val_loss"])
    history.append(row)
    write_csv_atomically(pd.DataFrame(history), RUN_DIR / "training_history.csv")

    if np.isfinite(row["val_loss"]) and row["val_loss"] < best_val_loss - MIN_DELTA:
        best_val_loss = float(row["val_loss"])
        best_loss_iteration = int(iteration)
        best_loss_epoch = int(current_epoch)
        validations_without_improvement = 0
        with ema_weights_in_model():
            save_checkpoint(
                best_loss_path, iteration, current_epoch,
                {"selection": "minimum_exact_validation_joint_nll", "row": row,
                 "weights": "ema" if USE_EMA else "raw", "ema_decay": EMA_DECAY},
            )
    else:
        validations_without_improvement += 1

    with ema_weights_in_model():
        save_checkpoint(last_path, iteration, current_epoch,
                        {"selection": "last", "row": row, "weights": "ema" if USE_EMA else "raw", "ema_decay": EMA_DECAY})

    auroc_text = (
        f"| val macro AUROC 24m={row['val_macro_auc_24m']:.4f}, 60m={row['val_macro_auc_60m']:.4f} "
        if auroc_due else ""
    )
    log_print(
        f"\nValidation at iteration {iteration:06d} (epoch~{current_epoch}) "
        f"| train={row['train_loss']:.4f} | val_nll={row['val_loss']:.6f} "
        f"(disease={row['val_disease_cr_nll']:.6f}, death={row['val_death_nll']:.6f}) "
        f"{auroc_text}"
        f"| best={best_val_loss:.6f} @ {best_loss_iteration} "
        f"| no_improve={validations_without_improvement}\n"
    )

    if (
        PATIENCE_VALIDATIONS > 0
        and iteration >= MIN_ITERATIONS_BEFORE_EARLY_STOP
        and validations_without_improvement >= PATIENCE_VALIDATIONS
    ):
        log_print(f"Early stopping at iteration {iteration}.")
        break

# -----------------------------------------------------------------------------
# Restore the best-validation-NLL checkpoint and write the training summary
# -----------------------------------------------------------------------------
if not best_loss_path.exists():
    raise FileNotFoundError(f"Validation-NLL checkpoint was not created: {best_loss_path}")
ckpt = torch.load(best_loss_path, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
selected_checkpoint_row = dict(ckpt.get("extra", {}).get("row", {}))

# Validation performance of the restored (best-NLL) checkpoint:
# macro AUROC at 24 and 60 months, per-condition AUROC, and top/bottom 5 conditions.
val_per_condition_final = validation_per_condition_auroc(val_loader)
val_macro_final = validation_macro_auroc_summary(val_per_condition_final)
val_top_bottom, n_rankable_conditions = top_bottom_conditions(val_per_condition_final)
val_per_condition_final.to_csv(RUN_DIR / "validation_per_condition_auroc_24_60m.csv", index=False)
val_top_bottom.to_csv(RUN_DIR / "validation_top5_bottom5_conditions_24_60m.csv", index=False)
log_print("\nValidation macro AUROC of restored checkpoint:", val_macro_final)
log_print(
    f"\nTop {VAL_TOP_BOTTOM_N} / bottom {VAL_TOP_BOTTOM_N} conditions by mean 24/60-month AUROC "
    f"({n_rankable_conditions} conditions with >= {VAL_MIN_CASES_FOR_RANKING} cases at both horizons):"
)
log_print(val_top_bottom[["group", "rank", "condition_name", "n_cases_24m", "auc_24m", "n_cases_60m", "auc_60m", "auc_mean_24_60"]].to_string(index=False))

summary = {
    "training_unit": "optimizer_iteration",
    "max_iterations_requested": int(MAX_ITERATIONS),
    "completed_iterations": int(history[-1]["iteration"]) if history else 0,
    "validation_interval": int(VALIDATION_INTERVAL),
    "warmup_iterations": int(WARMUP_ITERATIONS),
    "n_validations": int(len(history)),
    "fixed_budget_training": bool(PATIENCE_VALIDATIONS == 0),
    "objective": "1.0*disease_nll + 0.10*death_nll",
    "primary_checkpoint_selection": "joint_nll",
    "best_val_loss": float(best_val_loss),
    "best_loss_iteration": int(best_loss_iteration),
    "best_loss_epoch": int(best_loss_epoch),
    "restored_checkpoint_path": str(best_loss_path),
    "last_path": str(last_path),
    "reported_horizons_months": list(EVAL_HORIZONS_MONTHS),
    "validation_macro_auroc_restored_checkpoint": val_macro_final,
    "validation_ranking_min_cases": int(VAL_MIN_CASES_FOR_RANKING),
    "validation_n_rankable_conditions": int(n_rankable_conditions),
    "validation_top_bottom_path": str(RUN_DIR / "validation_top5_bottom5_conditions_24_60m.csv"),
    "validation_per_condition_path": str(RUN_DIR / "validation_per_condition_auroc_24_60m.csv"),
    "validation_participants": int(len(val_samples)),
    "reader_temporal_bias_mode": FINAL_READER_TEMPORAL_BIAS_MODE,
    "weights_used_for_validation_and_checkpoint": "ema" if USE_EMA else "raw",
    "ema_decay": EMA_DECAY,
    "seed": int(cfg.SEED),
    "selected_checkpoint_val_disease_cr_nll": selected_checkpoint_row.get("val_disease_cr_nll"),
    "selected_checkpoint_val_death_nll": selected_checkpoint_row.get("val_death_nll"),
    "training_history_path": str(RUN_DIR / "training_history.csv"),
    "iteration_log_path": str(ITERATION_LOG_PATH),
    "full_console_log_path": str(FULL_CONSOLE_LOG_PATH),
}
with open(RUN_DIR / "cell6_training_summary_death_adjusted.json", "w") as f:
    json.dump(summary, f, indent=2)
log_print(
    f"Training complete. Restored {best_loss_path.name} "
    f"(val NLL {best_val_loss:.6f} at iteration {best_loss_iteration})."
)
_console_log_handle.flush()
_console_log_handle.close()
