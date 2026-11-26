# Gemma 4 E4B (frozen encoder) — Locked test evaluation, dense-grid metrics, participant bootstrap, and calibration export
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 25 ---
if PHASE=='evaluate':
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
    if not TEST_ONLY_CPU and (str(DEVICE) != "cuda:0" or torch.cuda.current_device() != 0):
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

    # Gemma arm: `maybe_load_best_checkpoint` is defined in the preceding cell (lock-aware loader).
    assert callable(maybe_load_best_checkpoint)


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

# --- Notebook cell 27 ---
if PHASE=='evaluate':
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

# --- Notebook cell 31 ---
if PHASE=='evaluate':
    import hashlib
    import json
    import math
    import os
    import platform
    import time
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import sklearn
    from sklearn.metrics import average_precision_score, roc_auc_score

    try:
        from joblib import Parallel, delayed
        JOBLIB_AVAILABLE = True
    except Exception:
        JOBLIB_AVAILABLE = False

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


    # =============================================================================
    # USER-EDITABLE CONFIGURATION
    # =============================================================================

    CI_BOOTSTRAP_N = int(getattr(cfg, "EVAL_BOOTSTRAP_N", 200))
    CI_BOOTSTRAP_SEED = int(getattr(cfg, "EVAL_BOOTSTRAP_SEED", 101))
    CI_ALPHA = float(getattr(cfg, "EVAL_BOOTSTRAP_ALPHA", 0.05))

    # Fixed-horizon discrimination estimates retained for main/supplementary use.
    CI_POINT_HORIZONS_MONTHS = tuple(
        int(value)
        for value in getattr(cfg, "EVAL_CI_POINT_HORIZONS_MONTHS", (24, 60))
    )

    # Wolbers-style C-index: rank by the predicted CIF at tau and include event-of-
    # interest cases observed by tau.
    CI_CINDEX_TAU_MONTHS = int(getattr(cfg, "EVAL_CINDEX_TAU_MONTHS", 60))

    # None means all conditions. To test the cell quickly, use a short tuple such
    # as (0, 1, 2), then restore None for the publication run.
    CI_CONDITION_IDS = None

    # Threading shares the large prediction arrays instead of copying them into
    # separate processes. Reduce this if the notebook server has limited CPUs.
    CI_N_JOBS = max(1, min(8, int(os.cpu_count() or 1)))
    CI_BOOTSTRAP_CHUNK_SIZE = 25

    # A condition-level CI is withheld if fewer than this fraction of bootstrap
    # estimates are finite. The counts and status remain in the output.
    CI_MIN_VALID_BOOTSTRAP_FRACTION = float(
        getattr(cfg, "EVAL_CI_MIN_VALID_BOOTSTRAP_FRACTION", 0.90)
    )

    # For a macro estimate, use a fixed point-valid condition set. A bootstrap
    # macro estimate is retained only if at least this fraction of that fixed set
    # has a finite estimate in the replicate.
    CI_MACRO_MIN_CONDITION_FRACTION = float(
        getattr(cfg, "EVAL_CI_MACRO_MIN_CONDITION_FRACTION", 0.90)
    )

    # Existing valid chunk files are reused. Set True only when deliberately
    # replacing chunks for this exact configuration.
    CI_FORCE_RECOMPUTE = False


    # =============================================================================
    # REQUIRED FILES AND LOCKED SETTINGS
    # =============================================================================

    if "RUN_DIR" not in globals():
        raise RuntimeError(
            "RUN_DIR is missing. Run the model setup cell before Cell 7B."
        )

    RUN_DIR = Path(RUN_DIR).expanduser().resolve()

    TRAIN_SAMPLES_PATH = RUN_DIR / "canonical_train_samples.pkl"
    TEST_SAMPLES_PATH = RUN_DIR / "canonical_test_samples.pkl"
    DENSE_PREDICTION_PATH = (
        RUN_DIR / "test_dense_prediction_arrays_for_integrated_metrics.npz"
    )
    DENSE_POINT_METRICS_PATH = (
        RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv"
    )
    INTEGRATED_POINT_METRICS_PATH = (
        RUN_DIR / "test_integrated_metrics_by_condition.csv"
    )

    required_paths = [
        TRAIN_SAMPLES_PATH,
        TEST_SAMPLES_PATH,
        DENSE_PREDICTION_PATH,
        DENSE_POINT_METRICS_PATH,
        INTEGRATED_POINT_METRICS_PATH,
    ]
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Cell 7B is missing required prior-cell artifacts:\n"
            + "\n".join(missing_paths)
        )

    if not (0.0 < float(CI_ALPHA) < 1.0):
        raise ValueError("CI_ALPHA must be strictly between 0 and 1.")
    if int(CI_BOOTSTRAP_N) < 1:
        raise ValueError("CI_BOOTSTRAP_N must be at least 1.")
    if int(CI_BOOTSTRAP_CHUNK_SIZE) < 1:
        raise ValueError("CI_BOOTSTRAP_CHUNK_SIZE must be at least 1.")
    if not (0.0 < float(CI_MIN_VALID_BOOTSTRAP_FRACTION) <= 1.0):
        raise ValueError(
            "CI_MIN_VALID_BOOTSTRAP_FRACTION must lie in (0, 1]."
        )
    if not (0.0 < float(CI_MACRO_MIN_CONDITION_FRACTION) <= 1.0):
        raise ValueError(
            "CI_MACRO_MIN_CONDITION_FRACTION must lie in (0, 1]."
        )

    MODEL_NAME_CI = str(
        getattr(
            globals().get("cfg", object()),
            "MODEL_VARIANT",
            "CRESCENT_ONE_MONTH_CAP_CRP_3LAYER_TEMPORAL_INTERACTION_NLL_SHARED_STATE",
        )
    )
    PROB_EPS_CI = float(
        getattr(globals().get("cfg", object()), "PROB_EPS", 1e-7)
    )
    CENSOR_SURVIVAL_FLOOR_CI = float(
        getattr(
            globals().get("cfg", object()),
            "EVAL_CENSOR_SURVIVAL_FLOOR",
            0.01,
        )
    )
    CR_TARGET_CENSOR_CI = int(
        getattr(globals().get("cfg", object()), "CR_TARGET_CENSOR", 0)
    )
    CR_TARGET_CONDITION_CI = int(
        getattr(globals().get("cfg", object()), "CR_TARGET_CONDITION", 1)
    )
    CR_TARGET_DEATH_CI = int(
        getattr(globals().get("cfg", object()), "CR_TARGET_DEATH", 2)
    )
    TIME_EPS_CI = 1e-8


    # =============================================================================
    # LOAD AND VALIDATE LOCKED ARRAYS
    # =============================================================================

    train_samples_ci = pd.read_pickle(TRAIN_SAMPLES_PATH)
    test_samples_ci = pd.read_pickle(TEST_SAMPLES_PATH)

    with np.load(DENSE_PREDICTION_PATH, allow_pickle=True) as dense_npz_ci:
        required_npz_keys = {
            "eid",
            "condition_names",
            "dense_horizons_months",
            "entry_delay_months",
            "risk_death_adjusted",
        }
        missing_npz_keys = sorted(required_npz_keys - set(dense_npz_ci.files))
        if missing_npz_keys:
            raise KeyError(
                "Dense prediction NPZ is missing keys: "
                f"{missing_npz_keys}"
            )

        dense_eids_ci = np.asarray(dense_npz_ci["eid"]).astype(str)
        condition_names_ci = [
            str(value)
            for value in np.asarray(
                dense_npz_ci["condition_names"],
                dtype=object,
            ).tolist()
        ]
        dense_horizons_ci = tuple(
            int(value)
            for value in np.asarray(
                dense_npz_ci["dense_horizons_months"],
                dtype=np.int64,
            ).tolist()
        )
        entry_delay_months_ci = float(
            np.asarray(dense_npz_ci["entry_delay_months"]).reshape(-1)[0]
        )
        dense_risk_ci = np.asarray(
            dense_npz_ci["risk_death_adjusted"],
            dtype=np.float64,
        )

    expected_integration_start_ci = int(
        getattr(cfg, "EVAL_INTEGRATION_START_MONTHS", 12)
    )
    expected_integration_end_ci = int(
        getattr(cfg, "EVAL_INTEGRATION_END_MONTHS", 120)
    )
    if (
        dense_horizons_ci[0] != expected_integration_start_ci
        or dense_horizons_ci[-1] != expected_integration_end_ci
    ):
        raise AssertionError(
            "Dense-grid integration window does not match the locked V11 "
            f"configuration: grid={dense_horizons_ci[0]}–{dense_horizons_ci[-1]}, "
            f"expected={expected_integration_start_ci}–{expected_integration_end_ci}."
        )

    test_eids_ci = test_samples_ci["eid"].astype(str).to_numpy()
    train_eids_ci = train_samples_ci["eid"].astype(str).to_numpy()

    if pd.Series(test_eids_ci).duplicated().any():
        raise AssertionError(
            "canonical_test_samples.pkl is not one row per participant."
        )
    if pd.Series(train_eids_ci).duplicated().any():
        raise AssertionError(
            "canonical_train_samples.pkl is not one row per participant."
        )
    if dense_eids_ci.tolist() != test_eids_ci.tolist():
        raise AssertionError(
            "Dense prediction rows do not match canonical test eid order."
        )

    n_test_ci = int(len(test_samples_ci))
    n_conditions_ci = int(len(condition_names_ci))
    n_dense_horizons_ci = int(len(dense_horizons_ci))

    expected_risk_shape_ci = (
        n_test_ci,
        n_conditions_ci,
        n_dense_horizons_ci,
    )
    if dense_risk_ci.shape != expected_risk_shape_ci:
        raise ValueError(
            "Dense risk cube has shape "
            f"{dense_risk_ci.shape}; expected {expected_risk_shape_ci}."
        )
    if not np.isfinite(dense_risk_ci).all():
        raise ValueError("Dense risk cube contains non-finite values.")
    if np.any((dense_risk_ci < -1e-7) | (dense_risk_ci > 1.0 + 1e-7)):
        raise ValueError("Dense risk cube contains values outside [0, 1].")
    dense_risk_ci = np.clip(dense_risk_ci, 0.0, 1.0)

    if tuple(sorted(set(dense_horizons_ci))) != dense_horizons_ci:
        raise AssertionError(
            "Dense horizons must be unique and strictly increasing."
        )

    requested_horizons_ci = tuple(
        int(value) for value in CI_POINT_HORIZONS_MONTHS
    )
    if len(set(requested_horizons_ci)) != len(requested_horizons_ci):
        raise ValueError("CI_POINT_HORIZONS_MONTHS contains duplicates.")
    missing_requested_horizons_ci = sorted(
        set(requested_horizons_ci) - set(dense_horizons_ci)
    )
    if missing_requested_horizons_ci:
        raise ValueError(
            "Requested point horizons are absent from the dense grid: "
            f"{missing_requested_horizons_ci}"
        )
    if int(CI_CINDEX_TAU_MONTHS) not in set(dense_horizons_ci):
        raise ValueError(
            "CI_CINDEX_TAU_MONTHS must be present in the dense prediction grid."
        )

    if CI_CONDITION_IDS is None:
        selected_condition_ids_ci = tuple(range(n_conditions_ci))
    else:
        selected_condition_ids_ci = tuple(
            int(value) for value in CI_CONDITION_IDS
        )
        if len(set(selected_condition_ids_ci)) != len(
            selected_condition_ids_ci
        ):
            raise ValueError("CI_CONDITION_IDS contains duplicates.")
        invalid_condition_ids_ci = [
            value
            for value in selected_condition_ids_ci
            if value < 0 or value >= n_conditions_ci
        ]
        if invalid_condition_ids_ci:
            raise ValueError(
                "CI_CONDITION_IDS contains out-of-range values: "
                f"{invalid_condition_ids_ci}"
            )

    dense_horizon_to_index_ci = {
        int(horizon): int(index)
        for index, horizon in enumerate(dense_horizons_ci)
    }


    # =============================================================================
    # COMPETING-RISK OUTCOMES AFTER THE DELAYED-ENTRY LANDMARK
    # =============================================================================

    def make_ci_condition_outcomes(frame):
        """Canonical competing-risk outcomes conditional on delayed-entry eligibility."""
        return make_delayed_entry_outcomes_numpy(
            np.stack(frame["cr_at_risk"].to_numpy()),
            np.stack(frame["cr_event_type"].to_numpy()),
            np.stack(frame["cr_event_time_months"].to_numpy()),
            frame["C_months"].to_numpy(dtype=float),
        )


    train_outcomes_ci = make_ci_condition_outcomes(train_samples_ci)
    test_outcomes_ci = make_ci_condition_outcomes(test_samples_ci)


    # =============================================================================
    # FROZEN TRAIN-SPLIT CENSORING DISTRIBUTIONS
    # =============================================================================

    class FrozenTrainingCensoringKM:
        """
        Marginal Kaplan–Meier estimator of administrative-censoring survival.

        before(t) returns G(t-); after(t) returns G(t). Both are estimated once
        from the canonical training split and are not refitted in test bootstraps.
        """

        def __init__(self, exit_time, censor_event, floor):
            time_values = np.asarray(exit_time, dtype=np.float64).reshape(-1)
            event_values = np.asarray(censor_event, dtype=bool).reshape(-1)
            valid = np.isfinite(time_values) & (time_values >= 0.0)
            time_values = time_values[valid]
            event_values = event_values[valid]
            self.floor = float(floor)

            if time_values.size == 0:
                self.times = np.asarray([], dtype=np.float64)
                self.survival_after = np.asarray([], dtype=np.float64)
                return

            order = np.argsort(time_values, kind="mergesort")
            time_values = time_values[order]
            event_values = event_values[order].astype(np.int64)

            unique_time, first = np.unique(
                time_values,
                return_index=True,
            )
            at_risk = time_values.size - first
            censor_count = np.add.reduceat(event_values, first)
            factors = 1.0 - censor_count / np.maximum(at_risk, 1)

            self.times = unique_time
            self.survival_after = np.cumprod(
                np.clip(factors, 0.0, 1.0)
            )

        def _query(self, query, side):
            query_array = np.asarray(query, dtype=np.float64)
            original_shape = query_array.shape
            flat_query = query_array.reshape(-1)
            indices = (
                np.searchsorted(self.times, flat_query, side=side) - 1
            )
            result = np.ones_like(flat_query, dtype=np.float64)
            valid = indices >= 0
            result[valid] = self.survival_after[indices[valid]]
            result = np.maximum(result, self.floor)

            if original_shape == ():
                return float(result[0])
            return result.reshape(original_shape)

        def before(self, query):
            return self._query(query, side="left")

        def after(self, query):
            return self._query(query, side="right")


    condition_censoring_kms_ci = []
    condition_cindex_censoring_kms_ci = []
    for condition_id_ci in range(n_conditions_ci):
        selected_train_ci = train_outcomes_ci["eligible"][
            :, condition_id_ci
        ]
        status_train_ci = train_outcomes_ci["status"][
            selected_train_ci,
            condition_id_ci,
        ]
        time_train_ci = train_outcomes_ci["exit_time"][
            selected_train_ci,
            condition_id_ci,
        ]
        condition_censoring_kms_ci.append(
            FrozenTrainingCensoringKM(
                exit_time=time_train_ci,
                censor_event=(status_train_ci == 0),
                floor=CENSOR_SURVIVAL_FLOOR_CI,
            )
        )
        condition_cindex_censoring_kms_ci.append(
            FrozenTrainingCensoringKM(
                exit_time=(
                    train_outcomes_ci["exit_time_untruncated"][
                        selected_train_ci,
                        condition_id_ci,
                    ]
                ),
                censor_event=(status_train_ci == 0),
                floor=CENSOR_SURVIVAL_FLOOR_CI,
            )
        )


    # =============================================================================
    # CANONICAL METRIC PRIMITIVES
    # =============================================================================

    def make_ci_binary_window(
        eligible,
        status,
        exit_time,
        horizon,
        censoring_km,
    ):
        eligible = np.asarray(eligible, dtype=bool)
        status = np.asarray(status, dtype=np.int8)
        exit_time = np.asarray(exit_time, dtype=np.float64)
        horizon = float(horizon)

        case = (
            eligible
            & (status == 1)
            & (exit_time <= horizon + TIME_EPS_CI)
        )
        competing = (
            eligible
            & (status == 2)
            & (exit_time <= horizon + TIME_EPS_CI)
        )
        observed_through = (
            eligible
            & (exit_time >= horizon - TIME_EPS_CI)
        )
        known = case | competing | observed_through
        y = case.astype(np.float64)

        weights = np.zeros_like(exit_time, dtype=np.float64)
        event_or_competing = case | competing
        if bool(event_or_competing.any()):
            weights[event_or_competing] = (
                1.0
                / censoring_km.before(exit_time[event_or_competing])
            )
        observed_control = observed_through & (~event_or_competing)
        if bool(observed_control.any()):
            weights[observed_control] = (
                1.0 / float(censoring_km.before(horizon))
            )

        weights[~np.isfinite(weights)] = 0.0
        return {
            "case": case,
            "competing": competing,
            "observed_through": observed_through,
            "known": known,
            "y": y,
            "weights": weights,
        }


    def ci_ipcw_auc_auprc(window, prediction):
        prediction = np.asarray(prediction, dtype=np.float64)
        known = np.asarray(window["known"], dtype=bool)
        y = np.asarray(window["y"], dtype=np.float64)
        weights = np.asarray(window["weights"], dtype=np.float64)

        valid = (
            known
            & np.isfinite(prediction)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        if (
            not bool(valid.any())
            or np.unique(y[valid].astype(np.int8)).size < 2
        ):
            return np.nan, np.nan

        auc = roc_auc_score(
            y[valid],
            prediction[valid],
            sample_weight=weights[valid],
        )
        auprc = average_precision_score(
            y[valid],
            prediction[valid],
            sample_weight=weights[valid],
        )
        return float(auc), float(auprc)


    def ci_ipcw_ht_brier(y, prediction, weights, eligible_n):
        y = np.asarray(y, dtype=np.float64)
        prediction = np.clip(
            np.asarray(prediction, dtype=np.float64),
            PROB_EPS_CI,
            1.0 - PROB_EPS_CI,
        )
        weights = np.asarray(weights, dtype=np.float64)

        valid = (
            np.isfinite(y)
            & np.isfinite(prediction)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        if int(eligible_n) <= 0 or not bool(valid.any()):
            return np.nan

        return float(
            np.sum(
                weights[valid]
                * np.square(y[valid] - prediction[valid])
            )
            / float(int(eligible_n))
        )


    def ci_aalen_johansen_cif(exit_time, status, horizon):
        """
        Aalen–Johansen cumulative incidence for condition status=1 with competing
        death status=2. Duplicated rows are retained, so a bootstrap sample is
        evaluated as its empirical resample.
        """

        time_values = np.asarray(exit_time, dtype=np.float64).reshape(-1)
        state_values = np.asarray(status, dtype=np.int8).reshape(-1)
        valid = np.isfinite(time_values) & (time_values >= 0.0)
        time_values = time_values[valid]
        state_values = state_values[valid]

        if time_values.size == 0:
            return np.nan

        order = np.argsort(time_values, kind="mergesort")
        time_values = time_values[order]
        state_values = state_values[order]

        unique_time, first = np.unique(
            time_values,
            return_index=True,
        )
        at_risk = time_values.size - first
        primary_events = np.add.reduceat(
            (state_values == 1).astype(np.int64),
            first,
        )
        all_events = np.add.reduceat(
            (state_values != 0).astype(np.int64),
            first,
        )

        event_mask = (
            (unique_time <= float(horizon) + TIME_EPS_CI)
            & (all_events > 0)
        )
        if not bool(event_mask.any()):
            return 0.0

        all_cause_hazard = all_events / np.maximum(at_risk, 1)
        survival_after = np.cumprod(
            np.clip(1.0 - all_cause_hazard, 0.0, 1.0)
        )
        survival_before = np.concatenate(
            [np.asarray([1.0]), survival_after[:-1]]
        )
        cif_increments = (
            survival_before
            * primary_events
            / np.maximum(at_risk, 1)
        )
        return float(np.sum(cif_increments[event_mask]))


    def ci_integrate(values, horizons):
        value_array = np.asarray(values, dtype=np.float64)
        horizon_array = np.asarray(horizons, dtype=np.float64)
        valid = np.isfinite(value_array) & np.isfinite(horizon_array)
        if int(valid.sum()) < 2:
            return np.nan

        x = horizon_array[valid]
        y = value_array[valid]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        span = float(x[-1] - x[0])
        if span <= 0.0:
            return np.nan

        trapezoid = getattr(np, "trapezoid", np.trapz)
        return float(trapezoid(y, x) / span)


    # =============================================================================
    # WOLBERS-STYLE TRUNCATED IPCW COMPETING-RISK C-INDEX
    #
    # Comparable pair types follow Wolbers et al., Biostatistics 2014:
    # A: i has the event of interest by tau and T_i < T_j (or j is censored at
    #    the same time), weighted by 1 / {G(T_i-) G(T_i)}.
    # B: i has the event of interest by tau and j has a competing event at or
    #    before T_i, weighted by 1 / {G(T_i-) G(T_j-)}.
    # Prediction ties contribute one half.
    # =============================================================================

    @njit(cache=False, nogil=True)
    def _ci_fenwick_add(tree, index, value):
        size = tree.shape[0]
        while index < size:
            tree[index] += value
            index += index & -index


    @njit(cache=False, nogil=True)
    def _ci_fenwick_sum(tree, index):
        total = 0.0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total


    @njit(cache=False, nogil=True)
    def _wolbers_cindex_core(
        exit_time,
        status,
        risk_rank,
        g_before,
        g_after,
        tau,
        n_risk_ranks,
    ):
        n_rows = exit_time.shape[0]
        order = np.argsort(exit_time)

        weighted_concordant = 0.0
        weighted_pairs = 0.0

        # Pair type A: later observed time, plus censoring tied at the case time.
        tree_a = np.zeros(n_risk_ranks + 1, dtype=np.float64)
        total_a = 0.0
        end = n_rows

        while end > 0:
            start = end - 1
            current_time = exit_time[order[start]]
            while (
                start > 0
                and exit_time[order[start - 1]] == current_time
            ):
                start -= 1

            # Same-time censored comparators are usable for pair type A.
            for position in range(start, end):
                row = order[position]
                if status[row] == 0:
                    _ci_fenwick_add(
                        tree_a,
                        risk_rank[row],
                        1.0,
                    )
                    total_a += 1.0

            if current_time <= tau:
                for position in range(start, end):
                    row = order[position]
                    if status[row] == 1:
                        rank = risk_rank[row]
                        lower = _ci_fenwick_sum(tree_a, rank - 1)
                        through = _ci_fenwick_sum(tree_a, rank)
                        tied = through - lower
                        pair_weight = (
                            1.0 / (g_before[row] * g_after[row])
                        )
                        weighted_pairs += pair_weight * total_a
                        weighted_concordant += pair_weight * (
                            lower + 0.5 * tied
                        )

            # All non-censored observations at this time become later comparators
            # for cases at earlier times.
            for position in range(start, end):
                row = order[position]
                if status[row] != 0:
                    _ci_fenwick_add(
                        tree_a,
                        risk_rank[row],
                        1.0,
                    )
                    total_a += 1.0

            end = start

        # Pair type B: competing event at or before the case time.
        tree_b = np.zeros(n_risk_ranks + 1, dtype=np.float64)
        total_b = 0.0
        start = 0

        while start < n_rows:
            end = start + 1
            current_time = exit_time[order[start]]
            while (
                end < n_rows
                and exit_time[order[end]] == current_time
            ):
                end += 1

            # Same-time competing events are usable for pair type B.
            for position in range(start, end):
                row = order[position]
                if status[row] == 2:
                    comparator_weight = 1.0 / g_before[row]
                    _ci_fenwick_add(
                        tree_b,
                        risk_rank[row],
                        comparator_weight,
                    )
                    total_b += comparator_weight

            if current_time <= tau:
                for position in range(start, end):
                    row = order[position]
                    if status[row] == 1:
                        rank = risk_rank[row]
                        lower = _ci_fenwick_sum(tree_b, rank - 1)
                        through = _ci_fenwick_sum(tree_b, rank)
                        tied = through - lower
                        case_weight = 1.0 / g_before[row]
                        weighted_pairs += case_weight * total_b
                        weighted_concordant += case_weight * (
                            lower + 0.5 * tied
                        )

            start = end

        if weighted_pairs <= 0.0:
            return np.nan, weighted_concordant, weighted_pairs

        return (
            weighted_concordant / weighted_pairs,
            weighted_concordant,
            weighted_pairs,
        )


    def _wolbers_cindex_naive(
        exit_time,
        status,
        risk,
        g_before,
        g_after,
        tau,
    ):
        weighted_concordant = 0.0
        weighted_pairs = 0.0
        n_rows = int(len(exit_time))

        for i in range(n_rows):
            if status[i] != 1 or exit_time[i] > float(tau):
                continue

            for j in range(n_rows):
                if i == j:
                    continue

                if (
                    exit_time[i] < exit_time[j]
                    or (
                        exit_time[i] == exit_time[j]
                        and status[j] == 0
                    )
                ):
                    weight = 1.0 / (
                        g_before[i] * g_after[i]
                    )
                elif status[j] == 2:
                    weight = 1.0 / (
                        g_before[i] * g_before[j]
                    )
                else:
                    continue

                weighted_pairs += weight
                if risk[i] > risk[j]:
                    weighted_concordant += weight
                elif risk[i] == risk[j]:
                    weighted_concordant += 0.5 * weight

        estimate = (
            weighted_concordant / weighted_pairs
            if weighted_pairs > 0.0
            else np.nan
        )
        return estimate, weighted_concordant, weighted_pairs


    def ci_wolbers_cindex(
        exit_time,
        status,
        risk,
        eligible,
        censoring_km,
        tau,
    ):
        exit_time = np.asarray(exit_time, dtype=np.float64)
        status = np.asarray(status, dtype=np.int8)
        risk = np.asarray(risk, dtype=np.float64)
        eligible = np.asarray(eligible, dtype=bool)

        valid = (
            eligible
            & np.isfinite(exit_time)
            & np.isfinite(risk)
            & np.isin(status, [0, 1, 2])
        )
        exit_time = exit_time[valid]
        status = status[valid]
        risk = risk[valid]

        if exit_time.size < 2:
            return {
                "estimate": np.nan,
                "weighted_concordant": 0.0,
                "weighted_pairs": 0.0,
            }

        _, inverse_rank = np.unique(risk, return_inverse=True)
        risk_rank = inverse_rank.astype(np.int64) + 1
        n_risk_ranks = int(risk_rank.max())
        g_before = np.asarray(
            censoring_km.before(exit_time),
            dtype=np.float64,
        )
        g_after = np.asarray(
            censoring_km.after(exit_time),
            dtype=np.float64,
        )

        estimate, weighted_concordant, weighted_pairs = (
            _wolbers_cindex_core(
                exit_time=exit_time,
                status=status,
                risk_rank=risk_rank,
                g_before=g_before,
                g_after=g_after,
                tau=float(tau),
                n_risk_ranks=n_risk_ranks,
            )
        )
        return {
            "estimate": float(estimate),
            "weighted_concordant": float(weighted_concordant),
            "weighted_pairs": float(weighted_pairs),
        }


    def validate_wolbers_implementation():
        test_time = np.asarray(
            [2.0, 4.0, 4.0, 5.0, 1.0, 3.0, 6.0, 2.0],
            dtype=np.float64,
        )
        test_status = np.asarray(
            [1, 0, 2, 1, 2, 0, 1, 1],
            dtype=np.int8,
        )
        test_risk = np.asarray(
            [0.8, 0.2, 0.2, 0.6, 0.1, 0.4, 0.6, 0.8],
            dtype=np.float64,
        )
        test_g_before = np.asarray(
            [0.95, 0.90, 0.90, 0.85, 1.00, 0.95, 0.80, 0.95],
            dtype=np.float64,
        )
        test_g_after = np.asarray(
            [0.95, 0.85, 0.85, 0.85, 1.00, 0.90, 0.80, 0.95],
            dtype=np.float64,
        )
        _, inverse_rank = np.unique(test_risk, return_inverse=True)

        fast = _wolbers_cindex_core(
            exit_time=test_time,
            status=test_status,
            risk_rank=inverse_rank.astype(np.int64) + 1,
            g_before=test_g_before,
            g_after=test_g_after,
            tau=5.0,
            n_risk_ranks=int(inverse_rank.max()) + 1,
        )
        slow = _wolbers_cindex_naive(
            exit_time=test_time,
            status=test_status,
            risk=test_risk,
            g_before=test_g_before,
            g_after=test_g_after,
            tau=5.0,
        )

        if not np.allclose(
            np.asarray(fast, dtype=np.float64),
            np.asarray(slow, dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AssertionError(
                "Optimized Wolbers C-index failed its naive-reference test: "
                f"fast={fast}, slow={slow}"
            )


    validate_wolbers_implementation()

    if not NUMBA_AVAILABLE:
        warnings.warn(
            "numba is unavailable. The exact C-index fallback is correct but may "
            "be slow. Install numba before the 2,000-replicate publication run.",
            RuntimeWarning,
        )
        if int(CI_BOOTSTRAP_N) >= 100:
            raise ImportError(
                "The publication bootstrap requires numba for the exact "
                "Wolbers C-index. Install numba in the notebook environment, "
                "restart the kernel if required, and rerun Cell 7B. For a small "
                "logic smoke test only, CI_BOOTSTRAP_N may be set below 100."
            )


    # =============================================================================
    # FIXED TRAIN-SPLIT NULL RISKS
    # =============================================================================

    train_null_risk_ci = np.full(
        (n_conditions_ci, n_dense_horizons_ci),
        np.nan,
        dtype=np.float64,
    )

    for condition_id_ci in selected_condition_ids_ci:
        selected_train_ci = train_outcomes_ci["eligible"][
            :, condition_id_ci
        ]
        train_time_ci = train_outcomes_ci["exit_time"][
            selected_train_ci,
            condition_id_ci,
        ]
        train_status_ci = train_outcomes_ci["status"][
            selected_train_ci,
            condition_id_ci,
        ]

        for grid_index_ci, horizon_ci in enumerate(dense_horizons_ci):
            train_null_risk_ci[
                condition_id_ci,
                grid_index_ci,
            ] = ci_aalen_johansen_cif(
                exit_time=train_time_ci,
                status=train_status_ci,
                horizon=float(horizon_ci),
            )


    # =============================================================================
    # ONE POINT/BOOTSTRAP SAMPLE
    # =============================================================================

    def ci_metric_row(
        bootstrap_id,
        condition_id,
        metric,
        estimate,
        horizon_months=np.nan,
        integration_start_months=np.nan,
        integration_end_months=np.nan,
        eligible_n=0,
        event_n=0,
        comparable_pair_weight=np.nan,
    ):
        return {
            "bootstrap_id": int(bootstrap_id),
            "model_name": MODEL_NAME_CI,
            "risk_type": "death_adjusted",
            "condition_id": int(condition_id),
            "condition_name": condition_names_ci[int(condition_id)],
            "metric": str(metric),
            "estimate": (
                float(estimate) if np.isfinite(estimate) else np.nan
            ),
            "horizon_months": (
                int(horizon_months)
                if np.isfinite(horizon_months)
                else np.nan
            ),
            "integration_start_months": (
                int(integration_start_months)
                if np.isfinite(integration_start_months)
                else np.nan
            ),
            "integration_end_months": (
                int(integration_end_months)
                if np.isfinite(integration_end_months)
                else np.nan
            ),
            "eligible_n": int(eligible_n),
            "event_n": int(event_n),
            "comparable_pair_weight": (
                float(comparable_pair_weight)
                if np.isfinite(comparable_pair_weight)
                else np.nan
            ),
        }


    def evaluate_ci_sample(index, bootstrap_id):
        index = np.asarray(index, dtype=np.int64)
        rows = []

        for condition_id in selected_condition_ids_ci:
            eligible = test_outcomes_ci["eligible"][
                index,
                condition_id,
            ]
            status = test_outcomes_ci["status"][
                index,
                condition_id,
            ]
            exit_time = test_outcomes_ci["exit_time"][
                index,
                condition_id,
            ]
            exit_time_untruncated = test_outcomes_ci[
                "exit_time_untruncated"
            ][
                index,
                condition_id,
            ]
            prediction_by_horizon = dense_risk_ci[
                index,
                condition_id,
                :,
            ]
            censoring_km = condition_censoring_kms_ci[
                condition_id
            ]
            eligible_n = int(eligible.sum())

            # Fixed-horizon AUROC and AUPRC.
            for horizon in requested_horizons_ci:
                grid_index = dense_horizon_to_index_ci[horizon]
                prediction = prediction_by_horizon[:, grid_index]
                window = make_ci_binary_window(
                    eligible=eligible,
                    status=status,
                    exit_time=exit_time,
                    horizon=float(horizon),
                    censoring_km=censoring_km,
                )
                auc, auprc = ci_ipcw_auc_auprc(
                    window=window,
                    prediction=prediction,
                )
                event_n = int(window["case"].sum())

                rows.append(
                    ci_metric_row(
                        bootstrap_id=bootstrap_id,
                        condition_id=condition_id,
                        metric="auc_ipcw_cr",
                        estimate=auc,
                        horizon_months=horizon,
                        eligible_n=eligible_n,
                        event_n=event_n,
                    )
                )
                rows.append(
                    ci_metric_row(
                        bootstrap_id=bootstrap_id,
                        condition_id=condition_id,
                        metric="auprc_ipcw_cr",
                        estimate=auprc,
                        horizon_months=horizon,
                        eligible_n=eligible_n,
                        event_n=event_n,
                    )
                )

            # Dense-grid Brier and calibration components.
            brier_values = []
            null_brier_values = []
            calibration_values = []

            for grid_index, horizon in enumerate(dense_horizons_ci):
                prediction = prediction_by_horizon[:, grid_index]
                window = make_ci_binary_window(
                    eligible=eligible,
                    status=status,
                    exit_time=exit_time,
                    horizon=float(horizon),
                    censoring_km=censoring_km,
                )
                brier = ci_ipcw_ht_brier(
                    y=window["y"],
                    prediction=prediction,
                    weights=window["weights"],
                    eligible_n=eligible_n,
                )

                null_prediction = np.full_like(
                    prediction,
                    train_null_risk_ci[condition_id, grid_index],
                    dtype=np.float64,
                )
                null_brier = ci_ipcw_ht_brier(
                    y=window["y"],
                    prediction=null_prediction,
                    weights=window["weights"],
                    eligible_n=eligible_n,
                )

                observed_risk = ci_aalen_johansen_cif(
                    exit_time=exit_time[eligible],
                    status=status[eligible],
                    horizon=float(horizon),
                )
                mean_prediction = (
                    float(np.mean(prediction[eligible]))
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

                brier_values.append(brier)
                null_brier_values.append(null_brier)
                calibration_values.append(
                    absolute_calibration_error
                )

            ibs = ci_integrate(brier_values, dense_horizons_ci)
            ibs_null = ci_integrate(
                null_brier_values,
                dense_horizons_ci,
            )
            scaled_ibs = (
                1.0 - ibs / ibs_null
                if (
                    np.isfinite(ibs)
                    and np.isfinite(ibs_null)
                    and ibs_null > 0.0
                )
                else np.nan
            )
            integrated_abs_calibration_error = ci_integrate(
                calibration_values,
                dense_horizons_ci,
            )

            end_horizon = int(dense_horizons_ci[-1])
            event_n_end = int(
                (
                    eligible
                    & (status == 1)
                    & (
                        exit_time
                        <= end_horizon + TIME_EPS_CI
                    )
                ).sum()
            )

            for metric, estimate in [
                ("ibs_ipcw_ht", ibs),
                ("ibs_null_ipcw_ht", ibs_null),
                ("scaled_ibs_ipcw_ht", scaled_ibs),
                (
                    "integrated_abs_calibration_error",
                    integrated_abs_calibration_error,
                ),
            ]:
                rows.append(
                    ci_metric_row(
                        bootstrap_id=bootstrap_id,
                        condition_id=condition_id,
                        metric=metric,
                        estimate=estimate,
                        integration_start_months=dense_horizons_ci[0],
                        integration_end_months=dense_horizons_ci[-1],
                        eligible_n=eligible_n,
                        event_n=event_n_end,
                    )
                )

            # Genuine truncated competing-risk concordance.
            cindex_tau = int(CI_CINDEX_TAU_MONTHS)
            cindex_grid_index = dense_horizon_to_index_ci[
                cindex_tau
            ]
            cindex_result = ci_wolbers_cindex(
                exit_time=exit_time_untruncated,
                status=status,
                risk=prediction_by_horizon[:, cindex_grid_index],
                eligible=eligible,
                censoring_km=(
                    condition_cindex_censoring_kms_ci[
                        condition_id
                    ]
                ),
                tau=float(cindex_tau),
            )
            cindex_event_n = int(
                (
                    eligible
                    & (status == 1)
                    & (
                        exit_time_untruncated
                        <= cindex_tau + TIME_EPS_CI
                    )
                ).sum()
            )
            rows.append(
                ci_metric_row(
                    bootstrap_id=bootstrap_id,
                    condition_id=condition_id,
                    metric="cindex_ipcw_cr",
                    estimate=cindex_result["estimate"],
                    horizon_months=cindex_tau,
                    eligible_n=eligible_n,
                    event_n=cindex_event_n,
                    comparable_pair_weight=(
                        cindex_result["weighted_pairs"]
                    ),
                )
            )

        return pd.DataFrame(rows)


    # =============================================================================
    # LOCKED POINT ESTIMATES AND CROSS-CHECK AGAINST EXISTING TABLES
    # =============================================================================

    print("=" * 88)
    print("CELL 7B — participant-bootstrap confidence intervals")
    print("=" * 88)
    print(
        {
            "model_name": MODEL_NAME_CI,
            "test_participants": n_test_ci,
            "conditions_selected": len(selected_condition_ids_ci),
            "bootstrap_replicates": int(CI_BOOTSTRAP_N),
            "point_horizons_months": requested_horizons_ci,
            "cindex_tau_months": int(CI_CINDEX_TAU_MONTHS),
            "dense_horizons_months": dense_horizons_ci,
            "n_jobs": int(CI_N_JOBS),
            "numba_available": bool(NUMBA_AVAILABLE),
        }
    )

    cell7b_start_time = time.perf_counter()

    point_metrics_ci = evaluate_ci_sample(
        index=np.arange(n_test_ci, dtype=np.int64),
        bootstrap_id=-1,
    )


    def _point_lookup(frame, metric, condition_id, horizon=np.nan):
        selected = (
            frame["metric"].astype(str).eq(str(metric))
            & frame["condition_id"].eq(int(condition_id))
        )
        if np.isfinite(horizon):
            selected &= frame["horizon_months"].eq(int(horizon))
        else:
            selected &= frame["horizon_months"].isna()

        values = frame.loc[selected, "estimate"]
        if len(values) != 1:
            raise AssertionError(
                "Point metric lookup was not one-to-one for "
                f"metric={metric}, condition={condition_id}, horizon={horizon}."
            )
        return float(values.iloc[0])


    dense_point_reference_ci = pd.read_csv(DENSE_POINT_METRICS_PATH)
    integrated_point_reference_ci = pd.read_csv(
        INTEGRATED_POINT_METRICS_PATH
    )

    if "risk_type" in dense_point_reference_ci.columns:
        dense_point_reference_ci = dense_point_reference_ci[
            dense_point_reference_ci["risk_type"]
            .astype(str)
            .eq("death_adjusted")
        ].copy()
    if "risk_type" in integrated_point_reference_ci.columns:
        integrated_point_reference_ci = (
            integrated_point_reference_ci[
                integrated_point_reference_ci["risk_type"]
                .astype(str)
                .eq("death_adjusted")
            ].copy()
        )

    validation_differences_ci = []

    for condition_id_ci in selected_condition_ids_ci:
        for horizon_ci in requested_horizons_ci:
            reference_row_ci = dense_point_reference_ci[
                dense_point_reference_ci["condition_id"].eq(
                    int(condition_id_ci)
                )
                & dense_point_reference_ci["horizon_months"].eq(
                    int(horizon_ci)
                )
            ]
            if len(reference_row_ci) != 1:
                raise AssertionError(
                    "Dense point-reference lookup was not one-to-one for "
                    f"condition={condition_id_ci}, horizon={horizon_ci}."
                )

            for metric_ci, reference_column_ci in [
                ("auc_ipcw_cr", "auc_ipcw_cr"),
                ("auprc_ipcw_cr", "auprc_ipcw_cr"),
            ]:
                calculated_ci = _point_lookup(
                    point_metrics_ci,
                    metric_ci,
                    condition_id_ci,
                    horizon_ci,
                )
                reference_ci = float(
                    reference_row_ci[reference_column_ci].iloc[0]
                )
                if np.isfinite(calculated_ci) and np.isfinite(reference_ci):
                    difference_ci = abs(calculated_ci - reference_ci)
                    validation_differences_ci.append(difference_ci)
                    if not np.isclose(
                        calculated_ci,
                        reference_ci,
                        rtol=5e-5,
                        atol=1e-7,
                    ):
                        raise AssertionError(
                            "Cell 7B discrimination point estimate does not "
                            "match the canonical dense table: "
                            f"condition={condition_id_ci}, "
                            f"horizon={horizon_ci}, metric={metric_ci}, "
                            f"calculated={calculated_ci}, "
                            f"reference={reference_ci}."
                        )
                elif not (
                    np.isnan(calculated_ci) and np.isnan(reference_ci)
                ):
                    raise AssertionError(
                        "Cell 7B and canonical dense table disagree on a "
                        "finite/non-finite point estimate."
                    )

        reference_integrated_row_ci = (
            integrated_point_reference_ci[
                integrated_point_reference_ci["condition_id"].eq(
                    int(condition_id_ci)
                )
            ]
        )
        if len(reference_integrated_row_ci) != 1:
            raise AssertionError(
                "Integrated point-reference lookup was not one-to-one for "
                f"condition={condition_id_ci}."
            )

        for metric_ci, reference_column_ci in [
            ("ibs_ipcw_ht", "ibs_ipcw_ht"),
            ("ibs_null_ipcw_ht", "ibs_null_ipcw_ht"),
            ("scaled_ibs_ipcw_ht", "scaled_ibs_ipcw_ht"),
            (
                "integrated_abs_calibration_error",
                "integrated_abs_calibration_error",
            ),
        ]:
            calculated_ci = _point_lookup(
                point_metrics_ci,
                metric_ci,
                condition_id_ci,
                np.nan,
            )
            reference_ci = float(
                reference_integrated_row_ci[
                    reference_column_ci
                ].iloc[0]
            )
            if np.isfinite(calculated_ci) and np.isfinite(reference_ci):
                difference_ci = abs(calculated_ci - reference_ci)
                validation_differences_ci.append(difference_ci)
                if not np.isclose(
                    calculated_ci,
                    reference_ci,
                    rtol=5e-5,
                    atol=1e-7,
                ):
                    raise AssertionError(
                        "Cell 7B integrated point estimate does not match "
                        "the canonical integrated table: "
                        f"condition={condition_id_ci}, metric={metric_ci}, "
                        f"calculated={calculated_ci}, "
                        f"reference={reference_ci}."
                    )
            elif not (
                np.isnan(calculated_ci) and np.isnan(reference_ci)
            ):
                raise AssertionError(
                    "Cell 7B and canonical integrated table disagree on a "
                    "finite/non-finite point estimate."
                )

    max_point_validation_difference_ci = (
        float(max(validation_differences_ci))
        if validation_differences_ci
        else 0.0
    )
    print(
        "Point-estimate validation passed; maximum absolute difference:",
        max_point_validation_difference_ci,
    )


    # =============================================================================
    # REPRODUCIBLE, RESUMABLE PARTICIPANT BOOTSTRAP
    # =============================================================================

    def sha256_text_ci(values):
        digest = hashlib.sha256()
        for value in values:
            digest.update(str(value).encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()


    eid_axis_sha256_ci = sha256_text_ci(test_eids_ci)
    condition_axis_sha256_ci = sha256_text_ci(condition_names_ci)

    configuration_ci = {
        "schema_version": (
            "crescent-participant-bootstrap-ci.v3"
        ),
        "implementation_version": (
            "2026-09-30-delayed-entry-contract-v3"
        ),
        "model_name": MODEL_NAME_CI,
        "bootstrap_n": int(CI_BOOTSTRAP_N),
        "bootstrap_seed": int(CI_BOOTSTRAP_SEED),
        "alpha": float(CI_ALPHA),
        "ci_method": "percentile",
        "resampling_unit": "eid",
        "point_horizons_months": list(requested_horizons_ci),
        "cindex_tau_months": int(CI_CINDEX_TAU_MONTHS),
        "dense_horizons_months": list(dense_horizons_ci),
        "selected_condition_ids": list(selected_condition_ids_ci),
        "entry_delay_months": float(entry_delay_months_ci),
        "test_n": int(n_test_ci),
        "condition_n": int(n_conditions_ci),
        "prediction_sha256": sha256_file(DENSE_PREDICTION_PATH),
        "training_samples_sha256": sha256_file(TRAIN_SAMPLES_PATH),
        "test_samples_sha256": sha256_file(TEST_SAMPLES_PATH),
        "censor_survival_floor": float(CENSOR_SURVIVAL_FLOOR_CI),
        "delayed_entry_contract": delayed_entry_contract(),
        "eid_axis_sha256": eid_axis_sha256_ci,
        "condition_axis_sha256": condition_axis_sha256_ci,
        "censoring_source": "canonical_train_samples.pkl",
        "censoring_frozen_across_bootstraps": True,
        "model_frozen_across_bootstraps": True,
        "integrated_auroc_included": False,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    configuration_json_ci = json.dumps(
        configuration_ci,
        sort_keys=True,
        separators=(",", ":"),
    )
    configuration_signature_ci = hashlib.sha256(
        configuration_json_ci.encode("utf-8")
    ).hexdigest()[:16]

    chunk_directory_ci = (
        RUN_DIR
        / f"test_bootstrap_ci_chunks_{configuration_signature_ci}"
    )
    chunk_directory_ci.mkdir(parents=True, exist_ok=True)

    chunk_configuration_path_ci = (
        chunk_directory_ci / "configuration.json"
    )


    def write_json_atomic_ci(payload, path):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)
        temporary.replace(path)


    if chunk_configuration_path_ci.is_file():
        with open(
            chunk_configuration_path_ci,
            "r",
            encoding="utf-8",
        ) as handle:
            existing_configuration_ci = json.load(handle)
        if existing_configuration_ci != configuration_ci:
            raise RuntimeError(
                "Existing bootstrap chunks have a different configuration. "
                "Do not combine them."
            )
    else:
        write_json_atomic_ci(
            configuration_ci,
            chunk_configuration_path_ci,
        )


    def evaluate_bootstrap_replicate_ci(bootstrap_id):
        # This seed rule can be repeated for comparator models after their eid
        # order has been aligned and verified by eid_axis_sha256.
        seed_sequence = np.random.SeedSequence(
            [int(CI_BOOTSTRAP_SEED), int(bootstrap_id)]
        )
        rng = np.random.default_rng(seed_sequence)
        bootstrap_index = rng.integers(
            low=0,
            high=n_test_ci,
            size=n_test_ci,
            dtype=np.int64,
        )
        return evaluate_ci_sample(
            index=bootstrap_index,
            bootstrap_id=int(bootstrap_id),
        )


    def read_valid_chunk_ci(path, expected_ids):
        if not Path(path).is_file():
            return None
        frame = pd.read_csv(path)
        observed_ids = sorted(
            frame["bootstrap_id"].astype(int).unique().tolist()
        )
        if observed_ids != sorted(int(value) for value in expected_ids):
            raise RuntimeError(
                f"Bootstrap chunk has unexpected replicate IDs: {path}"
            )
        expected_rows = (
            len(expected_ids)
            * len(selected_condition_ids_ci)
            * (
                2 * len(requested_horizons_ci)
                + 4
                + 1
            )
        )
        if len(frame) != expected_rows:
            raise RuntimeError(
                f"Bootstrap chunk has {len(frame)} rows; "
                f"expected {expected_rows}: {path}"
            )
        return frame


    bootstrap_chunk_frames_ci = []

    for chunk_start_ci in range(
        0,
        int(CI_BOOTSTRAP_N),
        int(CI_BOOTSTRAP_CHUNK_SIZE),
    ):
        chunk_stop_ci = min(
            chunk_start_ci + int(CI_BOOTSTRAP_CHUNK_SIZE),
            int(CI_BOOTSTRAP_N),
        )
        chunk_ids_ci = list(range(chunk_start_ci, chunk_stop_ci))
        chunk_path_ci = (
            chunk_directory_ci
            / (
                f"bootstrap_{chunk_start_ci:05d}_"
                f"{chunk_stop_ci - 1:05d}.csv.gz"
            )
        )

        existing_chunk_ci = None
        if not bool(CI_FORCE_RECOMPUTE):
            existing_chunk_ci = read_valid_chunk_ci(
                chunk_path_ci,
                chunk_ids_ci,
            )

        if existing_chunk_ci is not None:
            bootstrap_chunk_frames_ci.append(existing_chunk_ci)
            print(
                f"Reused bootstrap replicates "
                f"{chunk_start_ci + 1}–{chunk_stop_ci} "
                f"of {CI_BOOTSTRAP_N}."
            )
            continue

        if JOBLIB_AVAILABLE and int(CI_N_JOBS) > 1:
            results_ci = Parallel(
                n_jobs=int(CI_N_JOBS),
                prefer="threads",
            )(
                delayed(evaluate_bootstrap_replicate_ci)(
                    bootstrap_id_ci
                )
                for bootstrap_id_ci in chunk_ids_ci
            )
        else:
            results_ci = [
                evaluate_bootstrap_replicate_ci(
                    bootstrap_id_ci
                )
                for bootstrap_id_ci in chunk_ids_ci
            ]

        chunk_frame_ci = pd.concat(
            results_ci,
            ignore_index=True,
        )
        expected_rows_ci = (
            len(chunk_ids_ci)
            * len(selected_condition_ids_ci)
            * (
                2 * len(requested_horizons_ci)
                + 4
                + 1
            )
        )
        if len(chunk_frame_ci) != expected_rows_ci:
            raise RuntimeError(
                "Bootstrap worker returned an unexpected number of rows: "
                f"{len(chunk_frame_ci)} != {expected_rows_ci}."
            )

        temporary_chunk_path_ci = chunk_path_ci.with_name(
            chunk_path_ci.name + ".tmp"
        )
        chunk_frame_ci.to_csv(
            temporary_chunk_path_ci,
            index=False,
            compression="gzip",
        )
        temporary_chunk_path_ci.replace(chunk_path_ci)
        bootstrap_chunk_frames_ci.append(chunk_frame_ci)

        print(
            f"Completed bootstrap replicates "
            f"{chunk_start_ci + 1}–{chunk_stop_ci} "
            f"of {CI_BOOTSTRAP_N}."
        )

    bootstrap_replicates_ci = pd.concat(
        bootstrap_chunk_frames_ci,
        ignore_index=True,
    )

    observed_bootstrap_ids_ci = sorted(
        bootstrap_replicates_ci[
            "bootstrap_id"
        ].astype(int).unique().tolist()
    )
    expected_bootstrap_ids_ci = list(
        range(int(CI_BOOTSTRAP_N))
    )
    if observed_bootstrap_ids_ci != expected_bootstrap_ids_ci:
        raise RuntimeError(
            "Combined bootstrap table does not contain the exact requested "
            "replicate ID set."
        )


    # =============================================================================
    # PER-CONDITION PERCENTILE CONFIDENCE INTERVALS
    # =============================================================================

    key_columns_ci = [
        "condition_id",
        "condition_name",
        "metric",
        "horizon_key",
        "integration_start_key",
        "integration_end_key",
    ]


    def add_ci_group_keys(frame):
        frame = frame.copy()
        frame["horizon_key"] = (
            pd.to_numeric(frame["horizon_months"], errors="coerce")
            .fillna(-1)
            .astype(int)
        )
        frame["integration_start_key"] = (
            pd.to_numeric(
                frame["integration_start_months"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
        )
        frame["integration_end_key"] = (
            pd.to_numeric(
                frame["integration_end_months"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
        )
        return frame


    point_keyed_ci = add_ci_group_keys(point_metrics_ci)
    replicate_keyed_ci = add_ci_group_keys(
        bootstrap_replicates_ci
    )

    grouped_replicates_ci = (
        replicate_keyed_ci
        .groupby(key_columns_ci, sort=False)["estimate"]
        .agg(
            n_bootstrap_total="size",
            n_valid_bootstrap="count",
            ci_lower=lambda values: values.quantile(
                float(CI_ALPHA) / 2.0
            ),
            ci_upper=lambda values: values.quantile(
                1.0 - float(CI_ALPHA) / 2.0
            ),
        )
        .reset_index()
    )

    point_columns_for_merge_ci = (
        key_columns_ci
        + [
            "model_name",
            "risk_type",
            "estimate",
            "horizon_months",
            "integration_start_months",
            "integration_end_months",
            "eligible_n",
            "event_n",
            "comparable_pair_weight",
        ]
    )
    per_condition_ci = point_keyed_ci[
        point_columns_for_merge_ci
    ].merge(
        grouped_replicates_ci,
        on=key_columns_ci,
        how="left",
        validate="one_to_one",
    )

    per_condition_ci["valid_bootstrap_fraction"] = (
        per_condition_ci["n_valid_bootstrap"]
        / per_condition_ci["n_bootstrap_total"]
    )
    per_condition_ci["ci_level"] = 1.0 - float(CI_ALPHA)
    per_condition_ci["ci_method"] = (
        "participant_bootstrap_percentile"
    )
    per_condition_ci["ci_status"] = np.where(
        per_condition_ci["valid_bootstrap_fraction"]
        >= float(CI_MIN_VALID_BOOTSTRAP_FRACTION),
        "ok",
        "insufficient_valid_replicates",
    )

    insufficient_ci = per_condition_ci["ci_status"].ne("ok")
    per_condition_ci.loc[
        insufficient_ci,
        ["ci_lower", "ci_upper"],
    ] = np.nan

    per_condition_ci = per_condition_ci.drop(
        columns=[
            "horizon_key",
            "integration_start_key",
            "integration_end_key",
        ]
    )


    # =============================================================================
    # MACRO ESTIMATES OVER A FIXED POINT-VALID CONDITION SET
    # =============================================================================

    macro_key_columns_ci = [
        "metric",
        "horizon_key",
        "integration_start_key",
        "integration_end_key",
    ]

    macro_point_rows_ci = []
    macro_replicate_rows_ci = []

    for macro_key_ci, point_group_ci in point_keyed_ci.groupby(
        macro_key_columns_ci,
        sort=False,
    ):
        (
            macro_metric_ci,
            macro_horizon_key_ci,
            macro_start_key_ci,
            macro_end_key_ci,
        ) = macro_key_ci

        point_valid_ci = point_group_ci[
            np.isfinite(point_group_ci["estimate"])
        ].copy()
        fixed_condition_ids_ci = sorted(
            point_valid_ci["condition_id"].astype(int).tolist()
        )
        fixed_condition_n_ci = int(len(fixed_condition_ids_ci))

        if fixed_condition_n_ci == 0:
            macro_point_estimate_ci = np.nan
        else:
            macro_point_estimate_ci = float(
                point_valid_ci["estimate"].mean()
            )

        macro_point_rows_ci.append(
            {
                "model_name": MODEL_NAME_CI,
                "risk_type": "death_adjusted",
                "metric": str(macro_metric_ci),
                "horizon_months": (
                    int(macro_horizon_key_ci)
                    if int(macro_horizon_key_ci) >= 0
                    else np.nan
                ),
                "integration_start_months": (
                    int(macro_start_key_ci)
                    if int(macro_start_key_ci) >= 0
                    else np.nan
                ),
                "integration_end_months": (
                    int(macro_end_key_ci)
                    if int(macro_end_key_ci) >= 0
                    else np.nan
                ),
                "estimate": macro_point_estimate_ci,
                "fixed_condition_n": fixed_condition_n_ci,
                "fixed_condition_ids": "|".join(
                    str(value) for value in fixed_condition_ids_ci
                ),
            }
        )

        replicate_group_ci = replicate_keyed_ci[
            replicate_keyed_ci["metric"].astype(str).eq(
                str(macro_metric_ci)
            )
            & replicate_keyed_ci["horizon_key"].eq(
                int(macro_horizon_key_ci)
            )
            & replicate_keyed_ci["integration_start_key"].eq(
                int(macro_start_key_ci)
            )
            & replicate_keyed_ci["integration_end_key"].eq(
                int(macro_end_key_ci)
            )
            & replicate_keyed_ci["condition_id"].isin(
                fixed_condition_ids_ci
            )
        ].copy()

        required_finite_condition_n_ci = (
            int(
                math.ceil(
                    float(CI_MACRO_MIN_CONDITION_FRACTION)
                    * fixed_condition_n_ci
                )
            )
            if fixed_condition_n_ci > 0
            else 1
        )

        if fixed_condition_n_ci == 0:
            grouped_macro_replicates_ci = [
                (bootstrap_id_ci, None)
                for bootstrap_id_ci in range(int(CI_BOOTSTRAP_N))
            ]
        else:
            grouped_macro_replicates_ci = list(
                replicate_group_ci.groupby(
                    "bootstrap_id",
                    sort=False,
                )
            )

        for bootstrap_id_ci, bootstrap_group_ci in (
            grouped_macro_replicates_ci
        ):
            if bootstrap_group_ci is None:
                finite_values_ci = pd.Series([], dtype=float)
            else:
                finite_values_ci = bootstrap_group_ci.loc[
                    np.isfinite(bootstrap_group_ci["estimate"]),
                    "estimate",
                ]
            finite_condition_n_ci = int(len(finite_values_ci))
            macro_estimate_ci = (
                float(finite_values_ci.mean())
                if (
                    finite_condition_n_ci
                    >= required_finite_condition_n_ci
                )
                else np.nan
            )
            macro_replicate_rows_ci.append(
                {
                    "bootstrap_id": int(bootstrap_id_ci),
                    "model_name": MODEL_NAME_CI,
                    "risk_type": "death_adjusted",
                    "aggregation_level": "macro",
                    "condition_id": -1,
                    "condition_name": "MACRO",
                    "metric": str(macro_metric_ci),
                    "estimate": macro_estimate_ci,
                    "horizon_months": (
                        int(macro_horizon_key_ci)
                        if int(macro_horizon_key_ci) >= 0
                        else np.nan
                    ),
                    "integration_start_months": (
                        int(macro_start_key_ci)
                        if int(macro_start_key_ci) >= 0
                        else np.nan
                    ),
                    "integration_end_months": (
                        int(macro_end_key_ci)
                        if int(macro_end_key_ci) >= 0
                        else np.nan
                    ),
                    "fixed_condition_n": fixed_condition_n_ci,
                    "finite_condition_n": finite_condition_n_ci,
                }
            )

    macro_point_ci = pd.DataFrame(macro_point_rows_ci)
    macro_replicates_ci = pd.DataFrame(
        macro_replicate_rows_ci
    )

    macro_summary_rows_ci = []
    for macro_row_ci in macro_point_ci.itertuples(index=False):
        selected_macro_replicates_ci = macro_replicates_ci[
            macro_replicates_ci["metric"].astype(str).eq(
                str(macro_row_ci.metric)
            )
            & (
                pd.to_numeric(
                    macro_replicates_ci["horizon_months"],
                    errors="coerce",
                )
                .fillna(-1)
                .astype(int)
                == (
                    int(macro_row_ci.horizon_months)
                    if np.isfinite(macro_row_ci.horizon_months)
                    else -1
                )
            )
            & (
                pd.to_numeric(
                    macro_replicates_ci[
                        "integration_start_months"
                    ],
                    errors="coerce",
                )
                .fillna(-1)
                .astype(int)
                == (
                    int(macro_row_ci.integration_start_months)
                    if np.isfinite(
                        macro_row_ci.integration_start_months
                    )
                    else -1
                )
            )
            & (
                pd.to_numeric(
                    macro_replicates_ci[
                        "integration_end_months"
                    ],
                    errors="coerce",
                )
                .fillna(-1)
                .astype(int)
                == (
                    int(macro_row_ci.integration_end_months)
                    if np.isfinite(
                        macro_row_ci.integration_end_months
                    )
                    else -1
                )
            )
        ]["estimate"]

        valid_macro_ci = selected_macro_replicates_ci[
            np.isfinite(selected_macro_replicates_ci)
        ]
        n_valid_macro_ci = int(len(valid_macro_ci))
        valid_fraction_macro_ci = (
            n_valid_macro_ci / int(CI_BOOTSTRAP_N)
        )
        macro_status_ci = (
            "ok"
            if valid_fraction_macro_ci
            >= float(CI_MIN_VALID_BOOTSTRAP_FRACTION)
            else "insufficient_valid_replicates"
        )

        macro_summary_rows_ci.append(
            {
                **macro_row_ci._asdict(),
                "ci_lower": (
                    float(
                        valid_macro_ci.quantile(
                            float(CI_ALPHA) / 2.0
                        )
                    )
                    if macro_status_ci == "ok"
                    else np.nan
                ),
                "ci_upper": (
                    float(
                        valid_macro_ci.quantile(
                            1.0 - float(CI_ALPHA) / 2.0
                        )
                    )
                    if macro_status_ci == "ok"
                    else np.nan
                ),
                "n_bootstrap_total": int(CI_BOOTSTRAP_N),
                "n_valid_bootstrap": n_valid_macro_ci,
                "valid_bootstrap_fraction": (
                    valid_fraction_macro_ci
                ),
                "ci_level": 1.0 - float(CI_ALPHA),
                "ci_method": (
                    "participant_bootstrap_percentile"
                ),
                "ci_status": macro_status_ci,
                "minimum_condition_fraction_per_replicate": (
                    float(CI_MACRO_MIN_CONDITION_FRACTION)
                ),
            }
        )

    macro_ci = pd.DataFrame(macro_summary_rows_ci)


    # =============================================================================
    # SAVE FINAL OUTPUTS
    # =============================================================================

    PER_CONDITION_CI_PATH = (
        RUN_DIR / "test_per_condition_metrics_with_ci.csv"
    )
    POINT_HORIZON_CI_PATH = (
        RUN_DIR / "test_point_horizon_metrics_with_ci.csv"
    )
    CINDEX_CI_PATH = (
        RUN_DIR / "test_cindex_by_condition_with_ci.csv"
    )
    INTEGRATED_CI_PATH = (
        RUN_DIR / "test_integrated_metrics_with_ci.csv"
    )
    MACRO_CI_PATH = (
        RUN_DIR / "test_macro_metrics_with_ci.csv"
    )
    BOOTSTRAP_REPLICATES_PATH = (
        RUN_DIR / "test_bootstrap_replicates_metrics.csv.gz"
    )
    BOOTSTRAP_METADATA_PATH = (
        RUN_DIR / "test_bootstrap_ci_metadata.json"
    )


    def write_csv_atomic_ci(frame, path, compression=None):
        path = Path(path)
        temporary = path.with_name(path.name + ".tmp")
        frame.to_csv(
            temporary,
            index=False,
            compression=compression,
        )
        temporary.replace(path)


    point_horizon_ci = per_condition_ci[
        per_condition_ci["metric"].isin(
            ["auc_ipcw_cr", "auprc_ipcw_cr"]
        )
    ].copy()
    cindex_ci = per_condition_ci[
        per_condition_ci["metric"].eq("cindex_ipcw_cr")
    ].copy()
    integrated_ci = per_condition_ci[
        per_condition_ci["metric"].isin(
            [
                "ibs_ipcw_ht",
                "ibs_null_ipcw_ht",
                "scaled_ibs_ipcw_ht",
                "integrated_abs_calibration_error",
            ]
        )
    ].copy()

    write_csv_atomic_ci(
        per_condition_ci,
        PER_CONDITION_CI_PATH,
    )
    write_csv_atomic_ci(
        point_horizon_ci,
        POINT_HORIZON_CI_PATH,
    )
    write_csv_atomic_ci(
        cindex_ci,
        CINDEX_CI_PATH,
    )
    write_csv_atomic_ci(
        integrated_ci,
        INTEGRATED_CI_PATH,
    )
    write_csv_atomic_ci(
        macro_ci,
        MACRO_CI_PATH,
    )

    replicate_export_ci = bootstrap_replicates_ci.copy()
    replicate_export_ci.insert(
        3,
        "aggregation_level",
        "condition",
    )

    macro_replicate_export_ci = macro_replicates_ci.copy()
    replicate_export_columns_ci = list(replicate_export_ci.columns)
    for column_ci in macro_replicate_export_ci.columns:
        if column_ci not in replicate_export_columns_ci:
            replicate_export_columns_ci.append(column_ci)

    for missing_column_ci in (
        set(replicate_export_columns_ci)
        - set(replicate_export_ci.columns)
    ):
        replicate_export_ci[missing_column_ci] = np.nan
    for missing_column_ci in (
        set(replicate_export_columns_ci)
        - set(macro_replicate_export_ci.columns)
    ):
        macro_replicate_export_ci[missing_column_ci] = np.nan
    macro_replicate_export_ci = macro_replicate_export_ci[
        replicate_export_columns_ci
    ]
    replicate_export_ci = replicate_export_ci[
        replicate_export_columns_ci
    ]

    replicate_export_ci = pd.concat(
        [
            replicate_export_ci,
            macro_replicate_export_ci,
        ],
        ignore_index=True,
    )
    write_csv_atomic_ci(
        replicate_export_ci,
        BOOTSTRAP_REPLICATES_PATH,
        compression="gzip",
    )

    elapsed_seconds_ci = float(
        time.perf_counter() - cell7b_start_time
    )
    metadata_ci = {
        **configuration_ci,
        "status": "completed",
        "configuration_signature": configuration_signature_ci,
        "bootstrap_seed_rule": (
            "default_rng(SeedSequence([bootstrap_seed, bootstrap_id]))"
            ".integers(0, test_n, size=test_n)"
        ),
        "point_estimate_validation": {
            "passed": True,
            "maximum_absolute_difference": (
                max_point_validation_difference_ci
            ),
            "reference_dense_point_file": str(
                DENSE_POINT_METRICS_PATH
            ),
            "reference_integrated_point_file": str(
                INTEGRATED_POINT_METRICS_PATH
            ),
        },
        "cindex_definition": {
            "name": (
                "Wolbers-style truncated IPCW competing-risk C-index"
            ),
            "tau_months_after_entry": int(
                CI_CINDEX_TAU_MONTHS
            ),
            "prediction_time_months_after_entry": int(
                CI_CINDEX_TAU_MONTHS
            ),
            "prediction": "death-adjusted condition CIF",
            "censoring_model": (
                "condition-specific marginal KM from canonical train split "
                "using untruncated observed follow-up times"
            ),
            "prediction_ties": "counted_as_one_half",
            "reference": (
                "Wolbers M et al. Biostatistics. 2014;15(3):526-539. "
                "doi:10.1093/biostatistics/kxt059"
            ),
        },
        "calibration_definition": {
            "observed_risk": (
                "Aalen-Johansen CIF recomputed within each test bootstrap"
            ),
            "error": (
                "absolute difference between bootstrap mean predicted CIF "
                "and bootstrap Aalen-Johansen CIF at each dense horizon"
            ),
            "integration": (
                "trapezoidal mean over dense horizon grid"
            ),
        },
        "validity_thresholds": {
            "minimum_valid_bootstrap_fraction": float(
                CI_MIN_VALID_BOOTSTRAP_FRACTION
            ),
            "macro_minimum_condition_fraction_per_replicate": float(
                CI_MACRO_MIN_CONDITION_FRACTION
            ),
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds_ci,
            "n_jobs": int(CI_N_JOBS),
            "joblib_available": bool(JOBLIB_AVAILABLE),
            "numba_available": bool(NUMBA_AVAILABLE),
            "chunk_size": int(CI_BOOTSTRAP_CHUNK_SIZE),
            "chunk_directory": str(chunk_directory_ci),
        },
        "output_files": {
            "per_condition_ci": str(PER_CONDITION_CI_PATH),
            "point_horizon_ci": str(POINT_HORIZON_CI_PATH),
            "cindex_ci": str(CINDEX_CI_PATH),
            "integrated_ci": str(INTEGRATED_CI_PATH),
            "macro_ci": str(MACRO_CI_PATH),
            "bootstrap_replicates": str(
                BOOTSTRAP_REPLICATES_PATH
            ),
            "metadata": str(BOOTSTRAP_METADATA_PATH),
        },
    }
    write_json_atomic_ci(metadata_ci, BOOTSTRAP_METADATA_PATH)


    # =============================================================================
    # FINAL CHECKS AND COMPACT DISPLAY
    # =============================================================================

    expected_per_condition_rows_ci = (
        len(selected_condition_ids_ci)
        * (
            2 * len(requested_horizons_ci)
            + 4
            + 1
        )
    )
    if len(per_condition_ci) != expected_per_condition_rows_ci:
        raise RuntimeError(
            "Final per-condition CI table has an unexpected row count."
        )
    if len(cindex_ci) != len(selected_condition_ids_ci):
        raise RuntimeError(
            "Final C-index CI table must contain one row per selected condition."
        )
    if per_condition_ci["n_bootstrap_total"].ne(
        int(CI_BOOTSTRAP_N)
    ).any():
        raise RuntimeError(
            "At least one CI row does not contain all requested replicates."
        )
    if not BOOTSTRAP_METADATA_PATH.is_file():
        raise RuntimeError("Bootstrap metadata file was not saved.")

    print("=" * 88)
    print("CELL 7B COMPLETE")
    print("=" * 88)
    print(
        f"Elapsed: {elapsed_seconds_ci / 60.0:.1f} minutes | "
        f"replicates: {CI_BOOTSTRAP_N:,} | "
        f"conditions: {len(selected_condition_ids_ci):,}"
    )
    print("Per-condition CI:", PER_CONDITION_CI_PATH)
    print("Point-horizon CI:", POINT_HORIZON_CI_PATH)
    print("C-index CI:", CINDEX_CI_PATH)
    print("Integrated CI:", INTEGRATED_CI_PATH)
    print("Macro CI:", MACRO_CI_PATH)
    print("Bootstrap replicates:", BOOTSTRAP_REPLICATES_PATH)
    print("Metadata:", BOOTSTRAP_METADATA_PATH)
    print(
        macro_ci[
            [
                "metric",
                "horizon_months",
                "estimate",
                "ci_lower",
                "ci_upper",
                "n_valid_bootstrap",
                "ci_status",
            ]
        ]
    )

# --- Notebook cell 33 ---
if PHASE=='evaluate':
    from pathlib import Path
    import hashlib
    import json

    import numpy as np
    import pandas as pd


    # =============================================================================
    # Configuration
    # =============================================================================

    assert "RUN_DIR" in globals(), "RUN_DIR is missing. Run the model notebook first."

    RUN_DIR = Path(RUN_DIR).expanduser().resolve()
    FIG2_EXPORT_MODEL_NAME = "Gemma4_E4B_B_frozen_CRESCENT_decoder"
    FIG2_RISK_TYPE = "death_adjusted"
    FIG2_HORIZONS_MONTHS = (12, 24, 60, 120)
    FIG2_PRIMARY_HORIZON_MONTHS = 24
    FIG2_TIME_EPS = 1e-8

    _cfg_for_export = globals().get("cfg", None)
    FIG2_N_CALIBRATION_BINS = int(
        getattr(_cfg_for_export, "EVAL_N_CALIBRATION_BINS", 10)
    )
    FIG2_MIN_CALIBRATION_GROUP_N = int(
        getattr(_cfg_for_export, "EVAL_MIN_CALIBRATION_GROUP_N", 100)
    )
    FIG2_MIN_CALIBRATION_BIN_N = int(
        getattr(_cfg_for_export, "EVAL_MIN_CALIBRATION_BIN_N", 50)
    )

    if FIG2_N_CALIBRATION_BINS < 2:
        raise ValueError("FIG2_N_CALIBRATION_BINS must be at least 2.")

    PREDICTION_PATH = RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
    TEST_SAMPLES_PATH = RUN_DIR / "canonical_test_samples.pkl"
    PER_CONDITION_METRICS_PATH = (
        RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
    )
    FIG2_INPUT_DIR = RUN_DIR / "figure2_revised_inputs_2yr"
    FIG2_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_locked_paths = [
        PREDICTION_PATH,
        TEST_SAMPLES_PATH,
        PER_CONDITION_METRICS_PATH,
    ]
    missing_locked_paths = [
        str(path) for path in required_locked_paths if not path.exists()
    ]
    if missing_locked_paths:
        raise FileNotFoundError(
            "Cell 7/canonical export has not produced all required locked files:\n  - "
            + "\n  - ".join(missing_locked_paths)
        )


    # =============================================================================
    # Helpers
    # =============================================================================

    def _sha256_file(path, chunk_size=1024 * 1024):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()


    def _normalise_text_sex(values):
        values = pd.Series(values).astype(str).str.strip().str.lower()
        mapping = {
            "m": "Male",
            "male": "Male",
            "man": "Male",
            "men": "Male",
            "f": "Female",
            "female": "Female",
            "woman": "Female",
            "women": "Female",
        }
        return values.map(mapping).fillna("Unknown").to_numpy(dtype=object)


    def _resolve_sex(meta):
        for column in ["sex", "sex_label", "gender", "gender_label"]:
            if column in meta.columns:
                labels = _normalise_text_sex(meta[column])
                if np.isin(labels, ["Male", "Female"]).any():
                    return labels, f"text_column:{column}"

        if "gender_idx" not in meta.columns:
            raise KeyError(
                "canonical_test_samples.pkl contains no sex/gender field."
            )

        # Figure-2 plotting contract:
        #   0 = Female, 1 = Male, 2 = Unknown.
        mapping = {0: "Female", 1: "Male", 2: "Unknown"}
        numeric = pd.to_numeric(meta["gender_idx"], errors="coerce")
        labels = (
            numeric.map(mapping)
            .fillna("Unknown")
            .astype(str)
            .to_numpy(dtype=object)
        )
        return labels, "gender_idx_mapping_0_female_1_male_2_unknown"


    def _aalen_johansen_cif(time, status, horizon_months):
        """
        Aalen–Johansen cumulative incidence for cause 1.

        status:
          0 = no event / administrative censoring
          1 = condition event
          2 = death competing event
        """
        time = np.asarray(time, dtype=np.float64).reshape(-1)
        status = np.asarray(status, dtype=np.int16).reshape(-1)
        horizon = float(horizon_months)

        valid = np.isfinite(time) & (time >= 0.0)
        time = time[valid]
        status = status[valid]

        if time.size == 0:
            return np.nan

        status_h = np.where(time <= horizon + FIG2_TIME_EPS, status, 0)
        time_h = np.minimum(time, horizon)
        event_mask = (status_h == 1) | (status_h == 2)

        if not bool(event_mask.any()):
            return 0.0

        event_times = np.unique(time_h[event_mask])
        sorted_times = np.sort(time_h)
        n_risk = (
            len(sorted_times)
            - np.searchsorted(sorted_times, event_times, side="left")
        ).astype(np.float64)

        cause1_index = np.searchsorted(
            event_times, time_h[status_h == 1], side="left"
        )
        cause2_index = np.searchsorted(
            event_times, time_h[status_h == 2], side="left"
        )
        d1 = np.bincount(
            cause1_index, minlength=len(event_times)
        ).astype(np.float64)
        d2 = np.bincount(
            cause2_index, minlength=len(event_times)
        ).astype(np.float64)

        valid_risk = n_risk > 0
        if not bool(valid_risk.all()):
            event_times = event_times[valid_risk]
            n_risk = n_risk[valid_risk]
            d1 = d1[valid_risk]
            d2 = d2[valid_risk]

        total_hazard = np.clip((d1 + d2) / n_risk, 0.0, 1.0)
        survival_before = np.ones(len(event_times), dtype=np.float64)
        if len(event_times) > 1:
            survival_before[1:] = np.cumprod(1.0 - total_hazard[:-1])

        cif = np.sum(survival_before * (d1 / n_risk))
        return float(np.clip(cif, 0.0, 1.0))


    def _weighted_quantile(values, weights, quantile):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        valid = (
            np.isfinite(values)
            & np.isfinite(weights)
            & (weights > 0.0)
        )
        values = values[valid]
        weights = weights[valid]

        if values.size == 0:
            return np.nan

        order = np.argsort(values, kind="mergesort")
        values = values[order]
        weights = weights[order]
        cumulative = np.cumsum(weights)
        cutoff = float(quantile) * float(cumulative[-1])
        position = int(np.searchsorted(cumulative, cutoff, side="left"))
        position = int(np.clip(position, 0, len(values) - 1))
        return float(values[position])


    def _make_observed_arrays(
        event_obs,
        t_tilde,
        at_risk,
        C_months,
        death_observed_post_index,
        death_time_months,
        entry_delay_months,
        horizon_months,
    ):
        """
        Reproduce the fixed-horizon outcome construction used by comparison Cell 4A.

        event_obs is already the canonical death-aware condition indicator produced
        by data preparation, so death-before-condition and configured ties remain
        resolved exactly as in the locked targets.
        """
        horizon = float(horizon_months)
        endpoint = float(entry_delay_months) + horizon

        C_mat = C_months[:, None]
        death_time_mat = death_time_months[:, None]
        death_obs_mat = death_observed_post_index[:, None]

        condition_before_entry = (
            event_obs
            & np.isfinite(t_tilde)
            & (t_tilde <= float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
        )
        death_before_entry = (
            death_obs_mat
            & np.isfinite(death_time_mat)
            & (death_time_mat <= float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
        )

        eligible = (
            at_risk
            & (C_mat > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
            & (~condition_before_entry)
            & (~death_before_entry)
        )

        condition_event = (
            eligible
            & event_obs
            & np.isfinite(t_tilde)
            & (t_tilde > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
            & (t_tilde <= endpoint + FIG2_TIME_EPS)
        )

        death_competing = (
            eligible
            & death_obs_mat
            & np.isfinite(death_time_mat)
            & (death_time_mat > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
            & (death_time_mat <= endpoint + FIG2_TIME_EPS)
            & (~condition_event)
        )

        observed_through = eligible & (C_mat >= endpoint - FIG2_TIME_EPS)
        known = condition_event | death_competing | observed_through

        status = np.zeros(event_obs.shape, dtype=np.int16)
        status[condition_event] = 1
        status[death_competing] = 2

        time_rel = np.full(event_obs.shape, horizon, dtype=np.float64)
        time_rel[condition_event] = (
            t_tilde[condition_event] - float(entry_delay_months)
        )

        if bool(death_competing.any()):
            death_rel = np.broadcast_to(
                death_time_mat - float(entry_delay_months), event_obs.shape
            )
            time_rel[death_competing] = death_rel[death_competing]

        censor_event = (
            eligible
            & (~condition_event)
            & (~death_competing)
            & (~observed_through)
        )
        if bool(censor_event.any()):
            censor_rel = np.broadcast_to(
                C_mat - float(entry_delay_months), event_obs.shape
            )
            time_rel[censor_event] = censor_rel[censor_event]

        time_rel = np.clip(time_rel, 0.0, horizon)

        return {
            "eligible": eligible,
            "known": known,
            "condition_event": condition_event,
            "death_competing": death_competing,
            "status": status,
            "time_rel": time_rel,
        }


    # =============================================================================
    # Load and validate the locked cohort and prediction arrays
    # =============================================================================

    test_meta = pd.read_pickle(TEST_SAMPLES_PATH).reset_index(drop=True)
    required_meta_columns = {"eid", "index_age_years", "gender_idx"}
    missing_meta_columns = sorted(required_meta_columns - set(test_meta.columns))
    if missing_meta_columns:
        raise KeyError(
            "canonical_test_samples.pkl is missing required Figure-2 columns: "
            f"{missing_meta_columns}"
        )

    with np.load(PREDICTION_PATH, allow_pickle=True) as locked_npz:
        npz_payload = {key: locked_npz[key] for key in locked_npz.files}

    required_base_npz = {
        "eid",
        "condition_names",
        "horizons_months",
        "entry_delay_months",
        "at_risk",
        "event_obs",
        "t_tilde",
        "C_months",
        "death_observed_post_index",
        "death_time_months",
    }
    missing_base_npz = sorted(required_base_npz - set(npz_payload))
    if missing_base_npz:
        raise KeyError(
            "The locked prediction NPZ is missing required arrays: "
            f"{missing_base_npz}"
        )

    pred_eid = np.asarray(npz_payload["eid"], dtype=object).astype(str)
    meta_eid = test_meta["eid"].astype(str).to_numpy()
    if pred_eid.tolist() != meta_eid.tolist():
        raise AssertionError(
            "Prediction NPZ and canonical_test_samples.pkl have different "
            "participant ordering."
        )

    index_age_years = pd.to_numeric(
        test_meta["index_age_years"], errors="coerce"
    ).to_numpy(dtype=np.float32)
    if index_age_years.shape != (len(pred_eid),):
        raise ValueError("Canonical index_age_years has an unexpected shape.")
    if not np.isfinite(index_age_years).all():
        raise ValueError("Canonical index_age_years contains non-finite values.")

    npz_augmented = "index_age_years" not in npz_payload
    if "index_age_years" in npz_payload:
        existing_index_age = np.asarray(
            npz_payload["index_age_years"], dtype=np.float32
        ).reshape(-1)
        if existing_index_age.shape != index_age_years.shape:
            raise ValueError(
                "Existing NPZ index_age_years has the wrong shape."
            )
        if not np.allclose(
            existing_index_age,
            index_age_years,
            rtol=0.0,
            atol=1e-5,
            equal_nan=False,
        ):
            raise AssertionError(
                "Existing NPZ index_age_years does not match canonical test samples."
            )
    else:
        npz_payload["index_age_years"] = index_age_years
        temporary_npz_path = PREDICTION_PATH.with_name(
            PREDICTION_PATH.stem + ".figure2_tmp.npz"
        )
        np.savez_compressed(temporary_npz_path, **npz_payload)
        temporary_npz_path.replace(PREDICTION_PATH)

    condition_names = [
        str(value).strip()
        for value in np.asarray(
            npz_payload["condition_names"], dtype=object
        ).tolist()
    ]
    available_horizons = {
        int(value)
        for value in np.asarray(
            npz_payload["horizons_months"]
        ).reshape(-1).tolist()
    }
    missing_horizons = sorted(
        set(FIG2_HORIZONS_MONTHS) - available_horizons
    )
    if missing_horizons:
        raise ValueError(
            "Locked predictions are missing Figure-2 horizons: "
            f"{missing_horizons}"
        )

    entry_delay_months = float(
        np.asarray(npz_payload["entry_delay_months"], dtype=np.float64).reshape(-1)[0]
    )
    N = int(len(pred_eid))
    K = int(len(condition_names))

    at_risk = np.asarray(npz_payload["at_risk"], dtype=np.float64) > 0.5
    event_obs = np.asarray(npz_payload["event_obs"], dtype=np.float64) > 0.5
    t_tilde = np.asarray(npz_payload["t_tilde"], dtype=np.float64)
    C_months = np.asarray(
        npz_payload["C_months"], dtype=np.float64
    ).reshape(-1)
    death_post = np.asarray(
        npz_payload["death_observed_post_index"], dtype=np.float64
    ).reshape(-1) > 0.5
    death_time = np.asarray(
        npz_payload["death_time_months"], dtype=np.float64
    ).reshape(-1)

    for name, array in {
        "at_risk": at_risk,
        "event_obs": event_obs,
        "t_tilde": t_tilde,
    }.items():
        if array.shape != (N, K):
            raise ValueError(
                f"{name} has shape {array.shape}; expected {(N, K)}."
            )

    for name, array in {
        "C_months": C_months,
        "death_observed_post_index": death_post,
        "death_time_months": death_time,
    }.items():
        if array.shape != (N,):
            raise ValueError(
                f"{name} has shape {array.shape}; expected {(N,)}."
            )

    risk_by_horizon = {}
    for horizon in FIG2_HORIZONS_MONTHS:
        key = f"risk_death_adjusted_h{int(horizon)}"
        if key not in npz_payload:
            raise KeyError(f"Locked prediction NPZ is missing {key}.")
        risk = np.asarray(npz_payload[key], dtype=np.float64)
        if risk.shape != (N, K):
            raise ValueError(
                f"{key} has shape {risk.shape}; expected {(N, K)}."
            )
        if not np.isfinite(risk).all():
            raise ValueError(f"{key} contains non-finite predictions.")
        if np.any((risk < -1e-7) | (risk > 1.0 + 1e-7)):
            raise ValueError(f"{key} contains predictions outside [0, 1].")
        risk_by_horizon[int(horizon)] = np.clip(risk, 0.0, 1.0)

    sex, sex_source = _resolve_sex(test_meta)
    sex_counts = pd.Series(sex).value_counts(dropna=False).to_dict()
    if int(np.sum(sex == "Male")) == 0:
        raise ValueError("No Male participants were resolved.")
    if int(np.sum(sex == "Female")) == 0:
        raise ValueError("No Female participants were resolved.")

    strata = {
        "All": np.ones(N, dtype=bool),
        "Male": sex == "Male",
        "Female": sex == "Female",
    }


    # =============================================================================
    # Build condition × horizon × sex-stratum AJ calibration
    # =============================================================================

    decile_rows = []
    summary_rows = []

    for horizon in FIG2_HORIZONS_MONTHS:
        horizon = int(horizon)
        risk = risk_by_horizon[horizon]
        observed = _make_observed_arrays(
            event_obs=event_obs,
            t_tilde=t_tilde,
            at_risk=at_risk,
            C_months=C_months,
            death_observed_post_index=death_post,
            death_time_months=death_time,
            entry_delay_months=entry_delay_months,
            horizon_months=horizon,
        )

        for condition_id, condition_name in enumerate(condition_names):
            for stratum_name, participant_mask in strata.items():
                eligible_mask = (
                    participant_mask
                    & observed["eligible"][:, condition_id]
                )
                eligible_n = int(eligible_mask.sum())

                if eligible_n < int(FIG2_MIN_CALIBRATION_GROUP_N):
                    continue

                scores = risk[eligible_mask, condition_id].astype(np.float64)
                times = observed["time_rel"][
                    eligible_mask, condition_id
                ].astype(np.float64)
                statuses = observed["status"][
                    eligible_mask, condition_id
                ].astype(np.int16)
                known_values = observed["known"][
                    eligible_mask, condition_id
                ]
                condition_values = observed["condition_event"][
                    eligible_mask, condition_id
                ]
                death_values = observed["death_competing"][
                    eligible_mask, condition_id
                ]

                quantiles = np.linspace(
                    0.0, 1.0, FIG2_N_CALIBRATION_BINS + 1
                )
                edges = np.unique(np.quantile(scores, quantiles))

                if edges.size <= 2:
                    bin_id = np.zeros(eligible_n, dtype=np.int16)
                    n_bins_actual = 1
                else:
                    bin_id = np.searchsorted(
                        edges[1:-1], scores, side="right"
                    ).astype(np.int16)
                    n_bins_actual = int(bin_id.max()) + 1

                condition_decile_rows = []
                for calibration_bin in range(n_bins_actual):
                    in_bin = bin_id == calibration_bin
                    bin_n = int(in_bin.sum())
                    if bin_n < int(FIG2_MIN_CALIBRATION_BIN_N):
                        continue

                    predicted_mean = float(np.mean(scores[in_bin]))
                    observed_aj_cif = _aalen_johansen_cif(
                        time=times[in_bin],
                        status=statuses[in_bin],
                        horizon_months=horizon,
                    )
                    error = (
                        float(observed_aj_cif - predicted_mean)
                        if np.isfinite(observed_aj_cif)
                        else np.nan
                    )

                    row = {
                        "Model": FIG2_EXPORT_MODEL_NAME,
                        "model": FIG2_EXPORT_MODEL_NAME,
                        "endpoint_type": "condition",
                        "risk_type": FIG2_RISK_TYPE,
                        "condition_id": int(condition_id),
                        "condition_name": condition_name,
                        "horizon_months": horizon,
                        "calibration_stratum": stratum_name,
                        "sex": stratum_name,
                        "calibration_bin": int(calibration_bin),
                        "n": bin_n,
                        "eligible_n": bin_n,
                        "known_n": int(known_values[in_bin].sum()),
                        "event_n": int(condition_values[in_bin].sum()),
                        "death_competing_n": int(death_values[in_bin].sum()),
                        "predicted_mean": predicted_mean,
                        "predicted_min": float(np.min(scores[in_bin])),
                        "predicted_max": float(np.max(scores[in_bin])),
                        "observed_aj_cif": observed_aj_cif,
                        "observed_minus_predicted": error,
                        "absolute_error": (
                            float(abs(error)) if np.isfinite(error) else np.nan
                        ),
                    }
                    decile_rows.append(row)
                    condition_decile_rows.append(row)

                finite_rows = [
                    row
                    for row in condition_decile_rows
                    if np.isfinite(row["absolute_error"])
                ]
                absolute_errors = np.asarray(
                    [row["absolute_error"] for row in finite_rows],
                    dtype=np.float64,
                )
                bin_weights = np.asarray(
                    [row["n"] for row in finite_rows],
                    dtype=np.float64,
                )

                if absolute_errors.size:
                    ici = float(
                        np.average(absolute_errors, weights=bin_weights)
                    )
                    e50 = _weighted_quantile(
                        absolute_errors, bin_weights, 0.50
                    )
                    e90 = _weighted_quantile(
                        absolute_errors, bin_weights, 0.90
                    )
                    ici_unweighted = float(np.mean(absolute_errors))
                    e50_unweighted = float(
                        np.quantile(absolute_errors, 0.50)
                    )
                    e90_unweighted = float(
                        np.quantile(absolute_errors, 0.90)
                    )
                else:
                    ici = np.nan
                    e50 = np.nan
                    e90 = np.nan
                    ici_unweighted = np.nan
                    e50_unweighted = np.nan
                    e90_unweighted = np.nan

                summary_rows.append(
                    {
                        "Model": FIG2_EXPORT_MODEL_NAME,
                        "model": FIG2_EXPORT_MODEL_NAME,
                        "endpoint_type": "condition",
                        "risk_type": FIG2_RISK_TYPE,
                        "condition_id": int(condition_id),
                        "condition_name": condition_name,
                        "horizon_months": horizon,
                        "calibration_stratum": stratum_name,
                        "sex": stratum_name,
                        "ici": ici,
                        "e50": e50,
                        "e90": e90,
                        "ici_unweighted_bins": ici_unweighted,
                        "e50_unweighted_bins": e50_unweighted,
                        "e90_unweighted_bins": e90_unweighted,
                        "n_calibration_bins": int(len(finite_rows)),
                        "total_eligible_n": eligible_n,
                        "total_known_n": int(known_values.sum()),
                        "total_event_n": int(condition_values.sum()),
                        "total_death_competing_n": int(death_values.sum()),
                        "mean_predicted_risk": float(np.mean(scores)),
                    }
                )

    deciles_all = pd.DataFrame(decile_rows)
    summary_all = pd.DataFrame(summary_rows)

    if deciles_all.empty:
        raise RuntimeError("No Figure-2 calibration-decile rows were generated.")
    if summary_all.empty:
        raise RuntimeError("No Figure-2 calibration-summary rows were generated.")

    deciles_all = deciles_all.sort_values(
        [
            "horizon_months",
            "condition_id",
            "calibration_stratum",
            "calibration_bin",
        ]
    ).reset_index(drop=True)
    summary_all = summary_all.sort_values(
        [
            "horizon_months",
            "condition_id",
            "calibration_stratum",
        ]
    ).reset_index(drop=True)

    deciles_2yr = deciles_all[
        deciles_all["horizon_months"].eq(
            FIG2_PRIMARY_HORIZON_MONTHS
        )
    ].copy()
    summary_2yr = summary_all[
        summary_all["horizon_months"].eq(
            FIG2_PRIMARY_HORIZON_MONTHS
        )
    ].copy()


    # =============================================================================
    # Save canonical Figure-2 calibration files and compatibility aliases
    # =============================================================================

    DECILES_2YR_PATH = (
        FIG2_INPUT_DIR
        / "figure2_revised_panelH_calibration_deciles_2yr.csv"
    )
    SUMMARY_2YR_PATH = (
        FIG2_INPUT_DIR
        / "figure2_revised_panelH_calibration_summary_2yr.csv"
    )
    DECILES_ALL_PATH = (
        FIG2_INPUT_DIR
        / "figure2_revised_panelH_calibration_deciles_all_horizons.csv"
    )
    SUMMARY_ALL_ROOT_PATH = (
        RUN_DIR
        / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
    )
    SUMMARY_ALL_FIG2_PATH = (
        FIG2_INPUT_DIR
        / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
    )
    SUMMARY_ALL_ALIAS_ROOT_PATH = (
        RUN_DIR / "test_calibration_error_by_horizon_condition_sex.csv"
    )
    SUMMARY_ALL_ALIAS_FIG2_PATH = (
        FIG2_INPUT_DIR
        / "test_calibration_error_by_horizon_condition_sex.csv"
    )

    deciles_2yr.to_csv(DECILES_2YR_PATH, index=False)
    summary_2yr.to_csv(SUMMARY_2YR_PATH, index=False)
    deciles_all.to_csv(DECILES_ALL_PATH, index=False)
    summary_all.to_csv(SUMMARY_ALL_ROOT_PATH, index=False)
    summary_all.to_csv(SUMMARY_ALL_FIG2_PATH, index=False)
    summary_all.to_csv(SUMMARY_ALL_ALIAS_ROOT_PATH, index=False)
    summary_all.to_csv(SUMMARY_ALL_ALIAS_FIG2_PATH, index=False)


    # =============================================================================
    # Hard compatibility checks for Figures 1 and 2
    # =============================================================================

    figure1_metrics = pd.read_csv(PER_CONDITION_METRICS_PATH)
    required_figure1_columns = {
        "horizon_months",
        "condition_name",
        "risk_type",
        "event_n",
        "auroc",
    }
    missing_figure1_columns = sorted(
        required_figure1_columns - set(figure1_metrics.columns)
    )
    if missing_figure1_columns:
        raise ValueError(
            "Figure-1 CRESCENT metrics are missing columns: "
            f"{missing_figure1_columns}"
        )

    required_decile_columns = {
        "condition_id",
        "condition_name",
        "horizon_months",
        "calibration_stratum",
        "sex",
        "calibration_bin",
        "n",
        "known_n",
        "event_n",
        "predicted_mean",
        "observed_aj_cif",
    }
    required_summary_columns = {
        "condition_id",
        "condition_name",
        "horizon_months",
        "calibration_stratum",
        "sex",
        "ici",
        "e50",
        "e90",
        "total_event_n",
    }

    missing_decile_columns = sorted(
        required_decile_columns - set(deciles_2yr.columns)
    )
    missing_summary_columns = sorted(
        required_summary_columns - set(summary_all.columns)
    )
    if missing_decile_columns:
        raise ValueError(
            "Figure-2 calibration-decile export is missing columns: "
            f"{missing_decile_columns}"
        )
    if missing_summary_columns:
        raise ValueError(
            "Figure-2 calibration-summary export is missing columns: "
            f"{missing_summary_columns}"
        )

    export_horizons = sorted(
        pd.to_numeric(
            summary_all["horizon_months"], errors="coerce"
        ).dropna().astype(int).unique().tolist()
    )
    if export_horizons != sorted(FIG2_HORIZONS_MONTHS):
        raise ValueError(
            "All-horizon calibration export has the wrong horizons: "
            f"{export_horizons}"
        )

    export_strata = set(summary_all["calibration_stratum"].astype(str))
    if not {"All", "Male", "Female"}.issubset(export_strata):
        raise ValueError(
            "Calibration export does not contain All, Male, and Female strata."
        )

    if deciles_2yr.empty or summary_2yr.empty:
        raise RuntimeError("The required 2-year calibration exports are empty.")

    with np.load(PREDICTION_PATH, allow_pickle=True) as final_npz:
        required_figure2_npz = {
            "eid",
            "condition_names",
            "horizons_months",
            "entry_delay_months",
            "index_age_years",
            "at_risk",
            "event_obs",
            "t_tilde",
            "C_months",
            "death_observed_post_index",
            "death_time_months",
            *{
                f"risk_death_adjusted_h{int(horizon)}"
                for horizon in FIG2_HORIZONS_MONTHS
            },
        }
        missing_figure2_npz = sorted(
            required_figure2_npz - set(final_npz.files)
        )
        if missing_figure2_npz:
            raise KeyError(
                "Final Figure-2 prediction NPZ is missing arrays: "
                f"{missing_figure2_npz}"
            )


    # =============================================================================
    # Metadata / source audit
    # =============================================================================

    output_paths = {
        "prediction_npz": PREDICTION_PATH,
        "figure1_per_condition_metrics": PER_CONDITION_METRICS_PATH,
        "calibration_deciles_2yr": DECILES_2YR_PATH,
        "calibration_summary_2yr": SUMMARY_2YR_PATH,
        "calibration_deciles_all_horizons": DECILES_ALL_PATH,
        "calibration_summary_all_horizons_root": SUMMARY_ALL_ROOT_PATH,
        "calibration_summary_all_horizons_figure2": SUMMARY_ALL_FIG2_PATH,
        "calibration_summary_alias_root": SUMMARY_ALL_ALIAS_ROOT_PATH,
        "calibration_summary_alias_figure2": SUMMARY_ALL_ALIAS_FIG2_PATH,
    }

    metadata = {
        "model": FIG2_EXPORT_MODEL_NAME,
        "risk_type": FIG2_RISK_TYPE,
        "run_dir": str(RUN_DIR),
        "n_test_participants": N,
        "n_conditions": K,
        "horizons_months": list(FIG2_HORIZONS_MONTHS),
        "primary_horizon_months": FIG2_PRIMARY_HORIZON_MONTHS,
        "entry_delay_months": entry_delay_months,
        "n_calibration_bins_requested": FIG2_N_CALIBRATION_BINS,
        "minimum_calibration_group_n": FIG2_MIN_CALIBRATION_GROUP_N,
        "minimum_calibration_bin_n": FIG2_MIN_CALIBRATION_BIN_N,
        "sex_source": sex_source,
        "gender_index_to_sex": {
            "0": "Female",
            "1": "Male",
            "2": "Unknown",
        },
        "sex_counts": {
            str(key): int(value) for key, value in sex_counts.items()
        },
        "npz_index_age_years_added": bool(npz_augmented),
        "observed_calibration": (
            "Aalen-Johansen condition CIF with death as competing event, "
            "using canonical death-aware event_obs and the Cell-7 entry delay"
        ),
        "calibration_error_definition": (
            "ICI/E50/E90 are participant-count-weighted absolute errors across "
            "quantile calibration bins within condition, horizon, and stratum"
        ),
        "outputs": {
            key: str(path) for key, path in output_paths.items()
        },
        "sha256": {
            key: _sha256_file(path) for key, path in output_paths.items()
        },
        "figure1_crescent_inputs_ready": True,
        "figure2_crescent_inputs_ready_for_comparison_cell4a": True,
    }

    METADATA_PATH = RUN_DIR / "figure2_calibration_export_metadata.json"
    with open(METADATA_PATH, "w") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print("=" * 88)
    print("A10/CRESCENT Figure-2 calibration export complete")
    print("=" * 88)
    print("RUN_DIR:", RUN_DIR)
    print("Participants:", N)
    print("Conditions:", K)
    print("Sex counts:", sex_counts)
    print("Added index_age_years to NPZ:", npz_augmented)
    print("Figure 1 CRESCENT input: READY")
    print("Figure 2 CRESCENT calibration inputs: READY for comparison Cell 4A")
    print("Calibration rows:", len(deciles_all))
    print("Calibration summary rows:", len(summary_all))
    print("Metadata:", METADATA_PATH)
    for label, path in output_paths.items():
        print(f"  {label}: {path}")

# --- Notebook cell 36 ---
if PHASE=='evaluate':
    # =============================================================================
    # COMPLETE FIGURE 2 MODEL-DEPENDENT INPUT EXPORT — DELAYED ENTRY
    #
    # Panel H still requires an independent comparator with the same landmark,
    # delayed-entry design, target definitions, and test participants. This block exports only
    # CRESCENT-derived data from the locked predictions and canonical test rows.
    # =============================================================================

    import os
    import shutil
    import warnings
    from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

    FIG2_TARGET_MODEL = FIG2_EXPORT_MODEL_NAME
    CRESCENT_FIG2_DIR = FIG2_INPUT_DIR
    CRESCENT_RUN_DIR = RUN_DIR
    FIG2_AGE_BIN_WIDTH_YEARS = 5
    FIG2_MIN_AGE_BIN_ELIGIBLE = 10
    GENDER_INDEX_TO_SEX = {0: "Female", 1: "Male", 2: "Unknown"}

    FIG2_SELECTED_CONDITIONS = (
        "Cerebrovascular disease",
        "Chronic kidney disease",
        "Chronic lower respiratory diseases",
        "Dementia",
        "Dermatitis and eczema",
        "Diabetes",
        "Hypertensive diseases",
        "Ischaemic heart diseases",
        "Inflammatory polyarthropathies",
        "Mood disorders",
    )


    required_horizons = [int(h) for h in FIG2_HORIZONS_MONTHS]
    H = deciles_2yr
    Hsummary = summary_2yr
    ALL_CAL = summary_all
    test_meta = test_meta.reset_index(drop=True)
    death_time = death_time.astype(np.float64)
    death_post = death_post.astype(bool)
    index_age_years = index_age_years.astype(np.float64)
    project_root = (RUN_DIR.parents[2] if RUN_DIR.parent.name == "runs"
                    and RUN_DIR.parent.parent.name.startswith("Models") else Path.cwd())
    color_source_setting = os.getenv("FIG2_COLOR_KEY_SOURCE", "").strip()
    if color_source_setting:
        FIG2_COLOR_KEY_SOURCE = Path(color_source_setting).expanduser().resolve()
    else:
        FIG2_COLOR_KEY_SOURCE = (
            project_root / "runs"
            / "RevisedModel_death_adjusted_reader_temporal_kernel_V1"
            / "figure2_revised_inputs_2yr"
            / "figure2_revised_condition_color_key.csv"
        ).resolve()

    color_source = FIG2_COLOR_KEY_SOURCE

    def make_observed_arrays(horizon_months):
        output = _make_observed_arrays(
            event_obs=event_obs, t_tilde=t_tilde, at_risk=at_risk,
            C_months=C_months, death_observed_post_index=death_post,
            death_time_months=death_time, entry_delay_months=entry_delay_months,
            horizon_months=horizon_months,
        )
        output["y"] = output["condition_event"].astype(np.int8)
        return output

    def binary_metrics(y, risk, known):
        valid = np.asarray(known, dtype=bool) & np.isfinite(risk)
        values = np.asarray(y, dtype=np.int8)[valid]
        scores = np.asarray(risk, dtype=np.float64)[valid]
        if values.size < 2 or np.unique(values).size != 2:
            return {"auroc": np.nan, "auprc": np.nan}
        return {"auroc": float(roc_auc_score(values, scores)),
                "auprc": float(average_precision_score(values, scores))}

    aalen_johansen_cif = _aalen_johansen_cif
    observed_by_horizon = {
        int(h): make_observed_arrays(int(h)) for h in required_horizons
    }

    # This is the same complete-case plotting estimand as the supplied Figure 2
    # script. The main model's primary IPCW AUROC remains in the evaluation CSV.
    # Static condition metadata / colour key
    # =============================================================================

    condition_base = pd.DataFrame(
        {
            "condition_id": np.arange(K, dtype=int),
            "condition_name": condition_names,
        }
    )

    color_source = Path(FIG2_COLOR_KEY_SOURCE).expanduser().resolve()
    if color_source.exists():
        external_color = pd.read_csv(color_source)
        if "condition_name" not in external_color.columns:
            warnings.warn(
                f"Ignoring colour key without condition_name: {color_source}"
            )
            external_color = pd.DataFrame(columns=["condition_name"])
    else:
        external_color = pd.DataFrame(columns=["condition_name"])

    for column in ["ICD_Chapter", "Chapter_Name"]:
        if column not in external_color.columns:
            external_color[column] = "Unmapped"

    external_color = (
        external_color[["condition_name", "ICD_Chapter", "Chapter_Name"]]
        .drop_duplicates("condition_name")
        .copy()
    )
    external_color["condition_name"] = external_color["condition_name"].astype(str).str.strip()

    color_key = condition_base.merge(external_color, on="condition_name", how="left")
    color_key["ICD_Chapter"] = color_key["ICD_Chapter"].fillna("Unmapped").astype(str)
    color_key["Chapter_Name"] = color_key["Chapter_Name"].fillna("Unmapped").astype(str)
    color_key_path = CRESCENT_FIG2_DIR / "figure2_revised_condition_color_key.csv"
    color_key.to_csv(color_key_path, index=False)

    chapter_lookup = color_key.set_index("condition_name")[
        ["ICD_Chapter", "Chapter_Name"]
    ].to_dict("index")


    # =============================================================================
    # Per-condition sex-stratified AUROC/AUPRC tables
    # =============================================================================

    strata = {
        "All": np.ones(N, dtype=bool),
        "Male": sex == "Male",
        "Female": sex == "Female",
    }

    obs_2yr = observed_by_horizon[int(FIG2_PRIMARY_HORIZON_MONTHS)]
    event_n_2yr_all = obs_2yr["condition_event"].sum(axis=0).astype(int)

    performance_rows = []
    for horizon_months in required_horizons:
        obs = observed_by_horizon[int(horizon_months)]
        risk = risk_by_horizon[int(horizon_months)]

        for condition_id, condition_name in enumerate(condition_names):
            meta = chapter_lookup.get(
                condition_name,
                {"ICD_Chapter": "Unmapped", "Chapter_Name": "Unmapped"},
            )

            for sex_label, participant_mask in strata.items():
                participant_mask = np.asarray(participant_mask, dtype=bool)
                eligible = participant_mask & obs["eligible"][:, condition_id]
                known = participant_mask & obs["known"][:, condition_id]
                cases = participant_mask & obs["condition_event"][:, condition_id]
                deaths = participant_mask & obs["death_competing"][:, condition_id]
                metrics = binary_metrics(
                    obs["y"][:, condition_id],
                    risk[:, condition_id],
                    known,
                )

                performance_rows.append(
                    {
                        "Model": FIG2_TARGET_MODEL,
                        "model": FIG2_TARGET_MODEL,
                        "endpoint_type": "condition",
                        "risk_type": "death_adjusted",
                        "condition_id": int(condition_id),
                        "condition_name": condition_name,
                        "horizon_months": int(horizon_months),
                        "sex": sex_label,
                        "eligible_n": int(eligible.sum()),
                        "known_n": int(known.sum()),
                        "event_n": int(cases.sum()),
                        "death_competing_n": int(deaths.sum()),
                        "control_n": int(max(int(known.sum()) - int(cases.sum()), 0)),
                        "auroc": metrics["auroc"],
                        "auprc": metrics["auprc"],
                        "frequency_for_plot": int(event_n_2yr_all[condition_id]),
                        "event_n_2yr_all": int(event_n_2yr_all[condition_id]),
                        "ICD_Chapter": meta["ICD_Chapter"],
                        "Chapter_Name": meta["Chapter_Name"],
                    }
                )

    performance = pd.DataFrame(performance_rows)

    B = performance[
        performance["horizon_months"].eq(int(FIG2_PRIMARY_HORIZON_MONTHS))
    ].copy()
    B_path = CRESCENT_FIG2_DIR / "figure2_revised_panelB_frequency_sex_2yr.csv"
    B.to_csv(B_path, index=False)

    C = B[B["sex"].isin(["Male", "Female"])].copy()
    C_path = CRESCENT_FIG2_DIR / "figure2_revised_panelC_chapter_sex_auroc_2yr.csv"
    C.to_csv(C_path, index=False)

    pair = (
        C.pivot_table(
            index=[
                "condition_id",
                "condition_name",
                "ICD_Chapter",
                "Chapter_Name",
                "frequency_for_plot",
            ],
            columns="sex",
            values="auroc",
            aggfunc="first",
        )
        .reset_index()
    )
    for column in ["Male", "Female"]:
        if column not in pair.columns:
            pair[column] = np.nan
    pair["auroc_female_minus_male"] = pair["Female"] - pair["Male"]
    Cpair = pair
    Cpair_path = (
        CRESCENT_FIG2_DIR
        / "figure2_revised_panelC_chapter_sex_pair_deltas_2yr.csv"
    )
    Cpair.to_csv(Cpair_path, index=False)

    E = performance.copy()
    E_path = CRESCENT_FIG2_DIR / "figure2_revised_panelE_horizon_sex_auroc.csv"
    E.to_csv(E_path, index=False)


    # =============================================================================
    # Selected endpoints: fixed prespecified condition list
    # =============================================================================

    calibrated_2yr_all = H[
        pd.to_numeric(H["horizon_months"], errors="coerce").eq(
            int(FIG2_PRIMARY_HORIZON_MONTHS)
        )
        & H["calibration_stratum"].astype(str).eq("All")
        & pd.to_numeric(H["predicted_mean"], errors="coerce").notna()
        & pd.to_numeric(H["observed_aj_cif"], errors="coerce").notna()
    ]
    calibration_conditions = set(calibrated_2yr_all["condition_name"].astype(str).str.strip())

    duplicate_condition_names = condition_base[
        condition_base["condition_name"].duplicated(keep=False)
    ]["condition_name"].unique().tolist()
    if duplicate_condition_names:
        raise RuntimeError(
            "The CRESCENT condition axis contains duplicate names: "
            + ", ".join(duplicate_condition_names[:10])
        )

    condition_to_id = condition_base.set_index("condition_name")["condition_id"].to_dict()
    missing_selected = [
        condition
        for condition in FIG2_SELECTED_CONDITIONS
        if condition not in condition_to_id
    ]
    if missing_selected:
        raise RuntimeError(
            "The following prespecified Figure 2 conditions are absent from the "
            "CRESCENT prediction arrays:\n  - "
            + "\n  - ".join(missing_selected)
        )

    selection = pd.DataFrame(
        {
            "selection_rank": np.arange(
                1, len(FIG2_SELECTED_CONDITIONS) + 1, dtype=int
            ),
            "condition_name": list(FIG2_SELECTED_CONDITIONS),
        }
    )
    selection["condition_id"] = selection["condition_name"].map(condition_to_id).astype(int)
    selection["event_n_2yr_all"] = selection["condition_id"].map(
        lambda condition_id: int(event_n_2yr_all[int(condition_id)])
    )
    selection["has_2yr_calibration"] = selection["condition_name"].isin(
        calibration_conditions
    )

    zero_event_conditions = selection.loc[
        selection["event_n_2yr_all"].le(0), "condition_name"
    ].tolist()
    missing_calibration_conditions = selection.loc[
        ~selection["has_2yr_calibration"], "condition_name"
    ].tolist()
    selection_failures = []
    if zero_event_conditions:
        selection_failures.append(
            "no observed 2-year events: " + ", ".join(zero_event_conditions)
        )
    if missing_calibration_conditions:
        selection_failures.append(
            "no finite 2-year all-participant calibration rows: "
            + ", ".join(missing_calibration_conditions)
        )
    if selection_failures:
        raise RuntimeError(
            "The prespecified Figure 2 condition set is incomplete:\n  - "
            + "\n  - ".join(selection_failures)
        )

    selected_path = CRESCENT_FIG2_DIR / "figure2_revised_selected_endpoints_panelA.csv"
    selection.to_csv(selected_path, index=False)
    selected_ids = selection["condition_id"].astype(int).tolist()


    # =============================================================================
    # Panel A: age- and sex-stratified predicted risk and observed AJ CIF
    # =============================================================================

    finite_age = index_age_years[np.isfinite(index_age_years)]
    if finite_age.size == 0:
        raise ValueError("No finite index ages are available for Panel A.")

    width = float(FIG2_AGE_BIN_WIDTH_YEARS)
    age_start = width * np.floor(float(finite_age.min()) / width)
    age_stop = width * np.ceil(float(finite_age.max()) / width) + width
    age_edges = np.arange(age_start, age_stop + 1e-9, width, dtype=float)
    if age_edges.size < 2:
        age_edges = np.asarray([age_start, age_start + width], dtype=float)
    age_bin_id = np.digitize(index_age_years, age_edges[1:-1], right=False)

    A_rows = []
    risk_2yr = risk_by_horizon[int(FIG2_PRIMARY_HORIZON_MONTHS)]
    for condition_id in selected_ids:
        condition_name = condition_names[condition_id]
        for sex_label in ["Male", "Female"]:
            sex_mask = sex == sex_label
            for bin_number in range(len(age_edges) - 1):
                lower = float(age_edges[bin_number])
                upper = float(age_edges[bin_number + 1])
                participant_mask = (
                    sex_mask
                    & np.isfinite(index_age_years)
                    & (age_bin_id == bin_number)
                )
                eligible = participant_mask & obs_2yr["eligible"][:, condition_id]
                eligible_n = int(eligible.sum())
                if eligible_n < int(FIG2_MIN_AGE_BIN_ELIGIBLE):
                    continue

                known = participant_mask & obs_2yr["known"][:, condition_id]
                cases = participant_mask & obs_2yr["condition_event"][:, condition_id]
                observed = aalen_johansen_cif(
                    obs_2yr["time_rel"][eligible, condition_id],
                    obs_2yr["status"][eligible, condition_id],
                    int(FIG2_PRIMARY_HORIZON_MONTHS),
                )

                A_rows.append(
                    {
                        "Model": FIG2_TARGET_MODEL,
                        "endpoint_type": "condition",
                        "risk_type": "death_adjusted",
                        "condition_id": int(condition_id),
                        "condition_name": condition_name,
                        "horizon_months": int(FIG2_PRIMARY_HORIZON_MONTHS),
                        "sex": sex_label,
                        "age_bin_lower": lower,
                        "age_bin_upper": upper,
                        "age_bin_mid": 0.5 * (lower + upper),
                        "eligible_n": eligible_n,
                        "known_n": int(known.sum()),
                        "event_n": int(cases.sum()),
                        "mean_predicted_risk": float(
                            np.mean(risk_2yr[eligible, condition_id])
                        ),
                        "observed_aj_cif": observed,
                    }
                )

    A = pd.DataFrame(A_rows)
    A_path = CRESCENT_FIG2_DIR / "figure2_revised_panelA_age_sex_rates_2yr.csv"
    A.to_csv(A_path, index=False)


    # =============================================================================
    # Panel F: 2-year ROC curves for the selected CRESCENT endpoints
    # =============================================================================

    F_rows = []
    for condition_id in selected_ids:
        condition_name = condition_names[condition_id]
        known = obs_2yr["known"][:, condition_id]
        y = obs_2yr["y"][:, condition_id][known].astype(int)
        score = risk_2yr[:, condition_id][known].astype(float)

        if y.size < 2 or len(np.unique(y)) < 2:
            continue

        fpr, tpr, threshold = roc_curve(y, score, drop_intermediate=False)
        auroc = float(roc_auc_score(y, score))
        auprc = float(average_precision_score(y, score))

        for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, threshold):
            F_rows.append(
                {
                    "Model": FIG2_TARGET_MODEL,
                    "endpoint_type": "condition",
                    "risk_type": "death_adjusted",
                    "condition_id": int(condition_id),
                    "condition_name": condition_name,
                    "horizon_months": int(FIG2_PRIMARY_HORIZON_MONTHS),
                    "sex": "All",
                    "fpr": float(fpr_value),
                    "tpr": float(tpr_value),
                    "threshold": float(threshold_value),
                    "auroc": auroc,
                    "auprc": auprc,
                    "known_n": int(known.sum()),
                    "event_n": int(y.sum()),
                }
            )

    F = pd.DataFrame(F_rows)
    F_path = CRESCENT_FIG2_DIR / "figure2_revised_panelF_roc_curves_2yr.csv"
    F.to_csv(F_path, index=False)


    # =============================================================================



    # Verify every expected panel input before marking this model export complete.
    fig2_panel_paths = {
        "A": A_path, "B": B_path, "C": C_path, "Cpair": Cpair_path,
        "E": E_path, "F": F_path, "H": DECILES_2YR_PATH,
        "Hsummary": SUMMARY_2YR_PATH, "Hsummary_all": SUMMARY_ALL_FIG2_PATH,
        "colors": color_key_path, "selected": selected_path,
    }
    for label, path in fig2_panel_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Figure 2 {label} export missing: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"Figure 2 {label} export is empty: {path}")

    required_panel_columns = {
        "A": {"condition_id", "condition_name", "sex", "age_bin_mid",
              "mean_predicted_risk", "observed_aj_cif", "eligible_n"},
        "B": {"condition_id", "condition_name", "sex", "event_n",
              "auroc", "auprc", "frequency_for_plot"},
        "C": {"condition_id", "condition_name", "sex", "ICD_Chapter",
              "Chapter_Name", "auroc"},
        "Cpair": {"condition_id", "Female", "Male",
                  "auroc_female_minus_male"},
        "E": {"condition_id", "condition_name", "sex",
              "horizon_months", "auroc"},
        "F": {"condition_id", "condition_name", "fpr", "tpr",
              "threshold", "auroc"},
        "H": required_decile_columns,
        "Hsummary": required_summary_columns,
        "selected": {"selection_rank", "condition_name", "condition_id"},
        "colors": {"condition_name", "ICD_Chapter", "Chapter_Name"},
    }
    for label, required in required_panel_columns.items():
        existing = set(pd.read_csv(fig2_panel_paths[label], nrows=0).columns)
        if required - existing:
            raise ValueError(f"Figure 2 {label} missing {sorted(required-existing)}")

    if len(selection) != len(FIG2_SELECTED_CONDITIONS):
        raise AssertionError("Selected Figure 2 condition list is incomplete")
    if set(E["horizon_months"].astype(int)) != set(FIG2_HORIZONS_MONTHS):
        raise AssertionError("Figure 2 horizon table has the wrong time axis")
    if not set(A["condition_name"]).issubset(set(selection["condition_name"])):
        raise AssertionError("Panel A contains an unselected condition")

    plot_comparator_setting = os.getenv("FIG2_COMPARATOR_METRICS_PATH", "").strip()
    if plot_comparator_setting:
        plot_comparator_path = Path(plot_comparator_setting).expanduser().resolve()
    else:
        plot_comparator_path = (
            project_root / "runs"
            / "Delphi2M_Canonical_A10_SixMonthDelayedEntry_Fresh50k"
            / "test_per_condition_metrics_by_horizon.csv"
        ).resolve()

    panel_h_contract = {
        "status": "requires_matched_comparator_validation",
        "comparator_metrics_path": str(plot_comparator_path),
        "crescent_entry_delay_months": float(entry_delay_months),
        "required_comparator_entry_delay_months": float(entry_delay_months),
        "required_test_eid_alignment": True,
        "required_condition_target_alignment": True,
    }
    comparator_config_path = plot_comparator_path.parent / "config.json"
    if comparator_config_path.is_file():
        comparator_config = json.loads(comparator_config_path.read_text())
        comparator_settings = comparator_config.get("config", comparator_config)
        comparator_entry = comparator_settings.get("ENTRY_DELAY_MONTHS")
        comparator_contract = comparator_config.get("delayed_entry_contract", {})
        if comparator_entry is None:
            comparator_entry = comparator_contract.get("entry_delay_months")
        matched_delay = comparator_entry is not None and np.isclose(float(comparator_entry), entry_delay_months, atol=1e-8)
        matched_likelihood = comparator_contract.get("training_likelihood") == delayed_entry_contract()["training_likelihood"]
        panel_h_contract["comparator_delayed_entry_verified"] = bool(matched_delay and matched_likelihood)
        if not (matched_delay and matched_likelihood):
            panel_h_contract["status"] = "incompatible_or_unverified_delayed_entry"
            warnings.warn("Panel H requires a comparator trained with the same six-month delayed-entry likelihood; a forecast offset alone is insufficient.", RuntimeWarning)
        else:
            panel_h_contract["status"] = "entry_contract_verified_targets_and_eids_still_require_alignment"

    figure2_manifest = {
        "model": FIG2_TARGET_MODEL,
        "comparator_model": "Delphi-2M",
        "run_dir": str(RUN_DIR),
        "metrics_path": str(PER_CONDITION_METRICS_PATH),
        "comparator_metrics_path": str(plot_comparator_path),
        "prediction_path": str(PREDICTION_PATH),
        "canonical_test_samples_path": str(TEST_SAMPLES_PATH),
        "figure2_input_dir": str(FIG2_INPUT_DIR),
        "n_test_participants": int(N),
        "n_conditions": int(K),
        "entry_delay_months": float(entry_delay_months),
        "horizons_months": list(FIG2_HORIZONS_MONTHS),
        "selected_condition_ids": selected_ids,
        "selected_condition_names": selection["condition_name"].tolist(),
        "sex_source": sex_source,
        "color_key_source": str(color_source) if color_source.is_file() else None,
        "panel_h_contract": panel_h_contract,
        "plot_auroc_estimator": "complete_case_known_outcomes_unweighted",
        "primary_auroc_source": "test_per_condition_metrics_by_horizon.csv:auc_ipcw_cr",
        "model_dependent_sources_all_within_crescent_run": True,
        "outputs": {label: str(path) for label, path in fig2_panel_paths.items()},
        "sha256": {label: _sha256_file(path) for label, path in fig2_panel_paths.items()},
    }
    figure2_manifest_path = FIG2_INPUT_DIR / "figure2_crescent_source_manifest.json"
    with figure2_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(figure2_manifest, handle, indent=2, default=str)

    metadata["figure2_crescent_inputs_ready_for_comparison_cell4a"] = True
    metadata["figure2_panels_exported"] = list(fig2_panel_paths)
    metadata["panel_h_contract"] = panel_h_contract
    metadata["outputs"].update({label: str(path)
                               for label, path in fig2_panel_paths.items()})
    metadata["sha256"].update({label: _sha256_file(path)
                              for label, path in fig2_panel_paths.items()})
    metadata["outputs"]["figure2_source_manifest"] = str(figure2_manifest_path)
    metadata["sha256"]["figure2_source_manifest"] = _sha256_file(figure2_manifest_path)
    with METADATA_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=str)

    print("CRESCENT Figure 2 inputs ready:", FIG2_INPUT_DIR)
    for label, path in fig2_panel_paths.items():
        print(f"  {label:12s}: {path}")
    print("Panel H:", panel_h_contract["status"])

# --- Notebook cell 39 ---
if PHASE=='evaluate':
    # V7 changes training/log schemas; audit model-dependent evaluation outputs directly.
    required=['test_prediction_arrays_death_adjusted.npz','test_per_condition_metrics_by_horizon.csv',
              'test_integrated_metrics_by_condition.csv','test_macro_metrics_with_ci.csv']
    missing=[p for p in required if not (RUN_DIR/p).exists()]
    ACCEPTANCE['G8_outputs']={'pass':not missing,'missing':missing}
    ACCEPTANCE['G9_lock_used']={'pass':sha256_file(RUN_DIR/PRIMARY_CHECKPOINT_NAME)==LOCKED_CHECKPOINT_SHA256}
    save_acceptance()
    if missing:raise FileNotFoundError(missing)
    print('V7 evaluation exports complete. Figure 2 and Wolbers cells retain the supplied CRESCENT definitions.')
