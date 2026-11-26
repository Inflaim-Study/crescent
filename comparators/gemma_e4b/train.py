# Gemma 4 E4B (frozen encoder) — Training or resume, validation-only selection lock, and pre-test lock verification
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 18 ---
if PHASE!='select':
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
    if not TEST_ONLY_CPU and (str(DEVICE) != "cuda:0" or torch.cuda.current_device() != 0):
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

    @torch.no_grad()
    def full_validation_report(iteration, log=print):
        """One validation pass supplies NLLs and native CIFs at all report horizons."""
        from sklearn.metrics import average_precision_score
        model.eval()
        horizons=tuple(int(h) for h in cfg.EVAL_HORIZONS_MONTHS)
        risks={h:[] for h in horizons};dnum=dden=mnum=mden=0.
        for batch in val_loader:
            batch=move_batch_to_device(batch,DEVICE);out=model(batch)
            _,parts=total_loss(out,batch)
            npair=parts['disease_n_active_pairs'];ndeath=parts['death_n_eligible_participants']
            dnum+=parts['disease_cr_nll']*npair;dden+=npair
            mnum+=parts['death_nll']*ndeath;mden+=ndeath
            for h in horizons:
                win=conditional_window_cif_from_probs(out['p_condition'],out['p_death_k'],out['p_neither'],
                    window_start_bin(ENTRY_DELAY_MONTHS),month_to_haz_bin(ENTRY_DELAY_MONTHS+h))
                risks[h].append(win['window_cif_condition'].cpu().numpy())
        if dden==0 or mden==0:
            raise ValueError('Validation has no eligible diagnosis pairs or mortality participants.')
        losses={'val_disease_cr_nll':dnum/dden,'val_death_nll':mnum/mden,
                'val_loss':LOSS_ALPHA_NLL*dnum/dden+LOSS_GAMMA_DEATH*mnum/mden}
        risks={h:np.concatenate(v) for h,v in risks.items()}
        rows=[]
        for k,name in enumerate(condition_names):
            eligible=VALIDATION_OUTCOMES['eligible'][:,k]
            t=VALIDATION_OUTCOMES['exit_time'][:,k];status=VALIDATION_OUTCOMES['status'][:,k]
            km=VALIDATION_CONDITION_CENSORING_KMS[k]
            for h in horizons:
                window=make_validation_binary_window(eligible,status,t,h,km)
                known=window['known'];case=window['case'];w=window['weights']
                p=np.clip(risks[h][:,k].astype(float),PROB_EPS,1-PROB_EPS)
                idx=np.searchsorted(km.times,float(h),side='left')-1
                raw_G=float(km.survival_after[idx]) if idx>=0 else 1.
                through=eligible&(t>=h-VALIDATION_TIME_EPS)
                supported=raw_G>=VALIDATION_CENSOR_SURVIVAL_FLOOR and int(through.sum())>0
                valid=known&np.isfinite(p)&np.isfinite(w)&(w>0)
                y=case[valid].astype(int);q=p[valid];ww=w[valid]
                auc=ap=brier=nbll=np.nan
                if supported and eligible.sum() and len(y):
                    if len(np.unique(y))==2:
                        auc=float(roc_auc_score(y,q,sample_weight=ww));ap=float(average_precision_score(y,q,sample_weight=ww))
                    brier=float(np.sum(ww*(y-q)**2)/eligible.sum())
                    nbll=float(-np.sum(ww*(y*np.log(q)+(1-y)*np.log1p(-q)))/eligible.sum())
                rows.append({'condition_id':k,'condition_name':str(name),'horizon_months':h,
                             'eligible_n':int(eligible.sum()),'known_n':int(known.sum()),'event_n':int(case.sum()),
                             'competing_death_n':int((eligible&(status==2)&(t<=h+VALIDATION_TIME_EPS)).sum()),
                             'observed_through_n':int(through.sum()),'censor_survival_raw':raw_G,'supported':supported,
                             'auc_ipcw_cr':auc,'auprc_ipcw_cr':ap,'brier_ipcw_ht':brier,'nbll_ipcw_ht':nbll,
                             'mean_predicted_risk':float(p[eligible].mean()) if eligible.any() else np.nan})
        per=pd.DataFrame(rows)
        macro=[];summary={}
        for h,g in per.groupby('horizon_months',sort=True):
            row={'horizon_months':int(h),'supported_conditions':int(g.supported.sum()),
                 'n_conditions_with_auroc':int(g.auc_ipcw_cr.notna().sum()),
                 'macro_auroc':float(g.auc_ipcw_cr.mean()),'macro_auprc':float(g.auprc_ipcw_cr.mean()),
                 'macro_brier_ipcw_ht':float(g.brier_ipcw_ht.mean()),'macro_nbll_ipcw_ht':float(g.nbll_ipcw_ht.mean())}
            macro.append(row)
            summary[f'val_macro_auc_{int(h)}m']=row['macro_auroc']
            summary[f'val_macro_auprc_{int(h)}m']=row['macro_auprc']
            summary[f'val_macro_brier_{int(h)}m']=row['macro_brier_ipcw_ht']
            summary[f'val_macro_nbll_{int(h)}m']=row['macro_nbll_ipcw_ht']
        macro=pd.DataFrame(macro)
        rank=pd.DataFrame({'condition_id':range(K),'condition_name':condition_names})
        for h in (24,60):
            g=per[per.horizon_months==h].set_index('condition_id')
            rank[f'auc_{h}m']=rank.condition_id.map(g.auc_ipcw_cr)
            rank[f'n_cases_{h}m']=rank.condition_id.map(g.event_n)
        rank['auc_mean_24_60']=rank[['auc_24m','auc_60m']].mean(axis=1,skipna=False)
        top_bottom,n_rankable=top_bottom_conditions(rank)
        tag=f'{iteration:06d}' if isinstance(iteration,int) else str(iteration)
        per.to_csv(RUN_DIR/f'validation_full_per_condition_{tag}.csv',index=False)
        macro.to_csv(RUN_DIR/f'validation_full_macro_{tag}.csv',index=False)
        rank.to_csv(RUN_DIR/f'validation_per_condition_auroc_iteration_{tag}.csv',index=False)
        top_bottom.to_csv(RUN_DIR/f'validation_top5_bottom5_{tag}.csv',index=False)
        log(f'\nFULL VALIDATION — {tag}: joint NLL={losses["val_loss"]:.7f}; diagnosis NLL={losses["val_disease_cr_nll"]:.7f}; death NLL={losses["val_death_nll"]:.7f}')
        log(macro.to_string(index=False))
        log('Per-condition performance (unsupported horizons are NaN):\n'+per.to_string(index=False))
        log(f'Top/bottom five by mean 24/60-month IPCW AUROC ({n_rankable} conditions meet the case threshold):\n'+top_bottom.to_string(index=False))
        atomic_json(RUN_DIR/f'validation_full_summary_{tag}.json',{'iteration':iteration,**losses,**summary,'n_rankable_conditions':n_rankable})
        return losses,summary,rank

    def training_counts(batch):
        """Denominators for the complete effective batch, computed before any forward."""
        types=batch['cr_event_type']; times=batch['cr_event_time_months']; C=batch['C_months']
        finite=torch.isfinite(times)
        event=finite & ((types==CR_TARGET_CONDITION)|(types==CR_TARGET_DEATH))
        cap=float(haz.N_BINS*haz.BIN_WIDTH_MONTHS)
        censor=C[:,None].clamp(0.,cap)
        eligible=(batch['cr_at_risk']>.5) & torch.isfinite(C)[:,None] & (censor>ENTRY_DELAY_MONTHS+ENTRY_TIME_EPS_MONTHS) & ~(event&(times<=ENTRY_DELAY_MONTHS+ENTRY_TIME_EPS_MONTHS))
        death=(batch['death_observed_post_index']>.5)&torch.isfinite(batch['death_time_months'])
        death_eligible=torch.isfinite(C)&(C.clamp(0.,cap)>ENTRY_DELAY_MONTHS+ENTRY_TIME_EPS_MONTHS)&~(death&(batch['death_time_months']<=ENTRY_DELAY_MONTHS+ENTRY_TIME_EPS_MONTHS))
        return int(eligible.sum()),int(death_eligible.sum())


    class StatefulExamples:
        def __init__(self,dataset,seed):
            self.dataset=dataset;self.generator=torch.Generator().manual_seed(seed)
            self.order=torch.randperm(len(dataset),generator=self.generator);self.cursor=0;self.epoch=1

        def next(self,n):
            if self.cursor>=len(self.order):
                self.order=torch.randperm(len(self.dataset),generator=self.generator);self.cursor=0;self.epoch+=1
            indices=self.order[self.cursor:self.cursor+n].tolist();self.cursor+=len(indices)
            return [self.dataset[i] for i in indices]

        def state_dict(self):
            return {'order':self.order,'cursor':self.cursor,'epoch':self.epoch,'generator':self.generator.get_state()}

        def load_state_dict(self,s):
            self.order=s['order'].cpu();self.cursor=s['cursor'];self.epoch=s['epoch'];self.generator.set_state(s['generator'].cpu())


    def capture_rng():
        return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
                'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


    def restore_rng(s):
        random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
        if s['cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


    def run_arm_training():
        if PHASE!='train':
            return
        if cfg.NORMALISE_DISEASE_CR_NLL_BY_CONDITION or TIME_SMOOTHNESS_WEIGHT!=0:
            raise ValueError('V7 uses the global eligible-pair NLL and no auxiliary smoothness term.')
        params=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        groups={}
        for n,p in params:
            lora='lora_' in n
            decay=p.ndim>=2 or not cfg.WEIGHT_DECAY_EXEMPT_1D
            groups.setdefault((lora,decay),[]).append(p)
        optimizer=torch.optim.AdamW([{'params':p,'weight_decay':cfg.WEIGHT_DECAY if decay else 0.,
                                     'initial_lr':ADAPTER_LR if lora else cfg.LR,
                                     'lr':ADAPTER_LR if lora else cfg.LR} for (lora,decay),p in groups.items()])
        trainable=[p for _,p in params]
        ema_decay=float(cfg.EMA_DECAY);ema=[p.detach().clone() for p in trainable] if ema_decay>0 else []
        effective=EFFECTIVE_BATCH if ARM=='C' else cfg.BATCH_SIZE
        micro=cfg.BATCH_SIZE
        dataset=train_ds if ARM=='C' else cached_train_ds
        collate=gemma_collate if ARM=='C' else cached_collate
        stream=StatefulExamples(dataset,SEED)
        best_path=RUN_DIR/'best_model_by_val_loss.pt';last_path=RUN_DIR/'last_model.pt'
        best=math.inf;best_step=-1;no_improve=0;start=1;history=[];iteration_rows=[];presentations=0
        signature=canonical_hash(effective_spec())
        hp=effective_spec(include_seed=False)
        record={'arm':ARM,'seed':SEED,'hpid':HPID,'benchmark':benchmark_binding(),'pipeline_code_sha256':PIPELINE_CODE_SHA256,
                'effective_spec_sha256':signature,'hp_signature':canonical_hash(hp),'hp_spec':hp,
                'encoder_fingerprint_sha256':CACHE_IDENTITY['fingerprint_sha256'],
                'text_serialisation_sha256':CACHE_IDENTITY['text_serialisation_sha256']}

        from contextlib import contextmanager
        @contextmanager
        def validation_weights():
            if not ema:
                yield;return
            raw=[p.detach().clone() for p in trainable]
            with torch.no_grad():
                for p,e in zip(trainable,ema):p.copy_(e)
            try:yield
            finally:
                with torch.no_grad():
                    for p,v in zip(trainable,raw):p.copy_(v)

        if RESUME:
            state=torch.load(last_path,map_location=DEVICE,weights_only=False)
            if state['effective_spec_sha256']!=signature:
                raise ValueError('Resume configuration/data/code changed. Use a fresh run directory.')
            load_checkpoint_state(model,state['model_state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
            stream.load_state_dict(state['stream']);restore_rng(state['rng'])
            if len(ema)!=len(state['ema']):raise ValueError('EMA settings changed on resume.')
            for e,v in zip(ema,state['ema']):e.copy_(v.to(e.device))
            start=state['iteration']+1;best=state['best_val_loss'];best_step=state['best_iteration'];no_improve=state['no_improve']
            history=state['history'];iteration_rows=state['iteration_rows'];presentations=state['participant_presentations']
            if not best_path.exists():raise FileNotFoundError('Resume requires the previously selected checkpoint.')
        atomic_json(RUN_DIR/'run_record.json',record)

        def checkpoint(path,iteration,selected=False):
            payload={'iteration':iteration,'global_step':iteration,'epoch':stream.epoch,
                     'model_state_dict':trainable_state_dict(model),'optimizer_state_dict':optimizer.state_dict(),
                     'ema':[e.detach().clone() for e in ema], 'stream':stream.state_dict(),'rng':capture_rng(),
                     'best_val_loss':best,'best_iteration':best_step,'no_improve':no_improve,'participant_presentations':presentations,
                     'history':history,'iteration_rows':iteration_rows,'effective_spec_sha256':signature,
                     'K':K,'condition_names':list(condition_names),'model_variant':cfg.MODEL_VARIANT,
                     'delayed_entry_contract':delayed_entry_contract(),'model_input_hashes':dict(MODEL_INPUT_HASHES),
                     'hazard_binning':{'BIN_WIDTH_MONTHS':haz.BIN_WIDTH_MONTHS,'MAX_HORIZON_MONTHS':haz.MAX_HORIZON_MONTHS,'N_BINS':haz.N_BINS},
                     'architecture_source':ARCHITECTURE_SOURCE,
                     'training_origin':{'from_scratch':True,'external_checkpoint_loaded':False,'pretrained_Gemma_base':True},
                     'objective':{'exact_piecewise_exponential_nll':True,'alpha_nll':LOSS_ALPHA_NLL,'gamma_death':LOSS_GAMMA_DEATH,
                                  'primary_checkpoint_selection':'joint_nll'},
                     'training_protocol':{'unit':'optimizer_iteration','max_iterations':cfg.MAX_ITERATIONS,
                                          'validation_interval':cfg.VALIDATION_INTERVAL,'validation_participants':len(val_samples)},
                     'extra':{**GEMMA_CHECKPOINT_IDENTITY,'selected':selected,'weights':'ema' if selected and ema else 'raw'}}
            tmp=path.with_name(path.name+'.tmp');torch.save(payload,tmp);tmp.replace(path)

        def update(rows,iteration):
            model.train();optimizer.zero_grad(set_to_none=True)
            warmup=min(cfg.WARMUP_ITERATIONS,max(1,cfg.MAX_ITERATIONS//10))
            ratio=iteration/max(1,warmup) if iteration<=warmup else cfg.SCHEDULER_MIN_LR_RATIO+(1-cfg.SCHEDULER_MIN_LR_RATIO)*.5*(1+math.cos(math.pi*(iteration-warmup)/max(1,cfg.MAX_ITERATIONS-warmup)))
            for g in optimizer.param_groups:g['lr']=g['initial_lr']*ratio
            batches=[collate(rows[a:a+micro],training=True) for a in range(0,len(rows),micro)]
            counts=[training_counts(b) for b in batches];pair_total=sum(c[0] for c in counts);death_total=sum(c[1] for c in counts)
            total_value=disease_value=death_value=0.
            for b,(npair,ndeath) in zip(batches,counts):
                b=move_batch_to_device(b,DEVICE);out=model(b)
                disease,_=disease_competing_risk_nll(out['condition_rate'],b['cr_event_type'],b['cr_event_time_months'],b['cr_at_risk'],b['C_months'])
                death,_=direct_death_nll(out['death_rate'],b['death_observed_post_index'],b['death_time_months'],b['C_months'])
                loss=LOSS_ALPHA_NLL*disease*(npair/max(1,pair_total))+LOSS_GAMMA_DEATH*death*(ndeath/max(1,death_total))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite effective-batch NLL.')
                loss.backward();total_value+=float(loss.detach());disease_value+=float(disease.detach())*npair/max(1,pair_total);death_value+=float(death.detach())*ndeath/max(1,death_total)
            if ARM=='C' and iteration==1 and pair_total+death_total>0:
                lora_norm=sum(float(p.grad.detach().float().square().sum()) for n,p in params if 'lora_' in n and p.grad is not None)
                if not math.isfinite(lora_norm) or lora_norm<=0:raise RuntimeError('LoRA received no finite nonzero gradient on the first update.')
            norm=torch.nn.utils.clip_grad_norm_(trainable,cfg.GRAD_CLIP,error_if_nonfinite=True);optimizer.step()
            if ema:
                decay=min(ema_decay,(1+iteration)/(10+iteration))
                with torch.no_grad():
                    for e,p in zip(ema,trainable):e.lerp_(p.detach(),1-decay)
            return {'iteration':iteration,'epoch':stream.epoch,'loss':total_value,'disease_cr_nll':disease_value,'death_nll':death_value,
                    'grad_norm':float(norm),'lr':float(cfg.LR*ratio),'lr_lora':float(ADAPTER_LR*ratio) if ARM=='C' else 0.,
                    'eligible_pairs':pair_total,'eligible_death_participants':death_total,'participants':len(rows)}

        handle=(RUN_DIR/'cell6_full_training_console.log').open('a' if RESUME else 'w',buffering=1)
        def log(s):print(s,flush=True);print(s,file=handle,flush=True)
        window=[];t0=time.time();last_iteration=start-1
        try:
            log(f'Arm {ARM}: {cfg.MAX_ITERATIONS} updates, effective batch {effective}, micro-batch {micro}; masking disabled.')
            for iteration in range(start,cfg.MAX_ITERATIONS+1):
                examples=stream.next(effective);presentations+=len(examples)
                row=update(examples,iteration);window.append(row);last_iteration=iteration
                if iteration%cfg.LOG_INTERVAL==0 or iteration==cfg.MAX_ITERATIONS:
                    iteration_rows.append({**row,'elapsed_minutes':(time.time()-t0)/60})
                    pd.DataFrame(iteration_rows).to_csv(RUN_DIR/'iteration_log_every_100.csv',index=False)
                    log(f"step {iteration:06d}/{cfg.MAX_ITERATIONS:06d} | epoch {stream.epoch:03d} | NLL={row['loss']:.6f}")
                auroc_due=(iteration%VAL_AUROC_INTERVAL==0) or iteration==cfg.MAX_ITERATIONS
                auc={};full_losses=None
                if auroc_due:
                    with validation_weights():full_losses,auc,_=full_validation_report(iteration,log)
                if iteration%cfg.VALIDATION_INTERVAL and iteration!=cfg.MAX_ITERATIONS and not auroc_due:
                    continue
                if full_losses is not None:val=full_losses
                else:
                    with validation_weights():val=evaluate_validation_loss(val_loader)
                valrow={'iteration':iteration,'epoch':stream.epoch,'train_loss':float(np.mean([r['loss'] for r in window])),**val,**auc}
                history.append(valrow);window=[]
                pd.DataFrame(history).to_csv(RUN_DIR/'training_history.csv',index=False)
                if np.isfinite(val['val_loss']) and val['val_loss']<best:
                    best=float(val['val_loss']);best_step=iteration;no_improve=0
                    with validation_weights():checkpoint(best_path,iteration,True)
                else:no_improve+=1
                checkpoint(last_path,iteration,False)  # raw weights + matching optimizer + separate EMA
                log(f'Validation NLL={val["val_loss"]:.7f}; selected {best:.7f} at {best_step}')
                if cfg.PATIENCE_VALIDATIONS>0 and no_improve>=cfg.PATIENCE_VALIDATIONS:break
            if not best_path.exists():raise RuntimeError('No finite selected checkpoint.')
            selected=torch.load(best_path,map_location=DEVICE,weights_only=False);load_checkpoint_state(model,selected['model_state_dict'])
            _,macro,per=full_validation_report('restored',log)
            per.to_csv(RUN_DIR/'validation_per_condition_auroc_24_60m.csv',index=False)
            ranked,_=top_bottom_conditions(per);ranked.to_csv(RUN_DIR/'validation_top5_bottom5_conditions_24_60m.csv',index=False)
            summary={'training_complete':True,'best_val_loss':best,'best_loss_iteration':best_step,'completed_iterations':last_iteration,
                     'max_iterations_requested':cfg.MAX_ITERATIONS,'restored_checkpoint_path':str(best_path),
                     'primary_checkpoint_selection':'joint_nll','validation_macro_auroc_restored_checkpoint':macro,
                     'training_unit':'optimizer_iteration','effective_batch':effective,'micro_batch':micro,
                     'participant_presentations':presentations,
                     'objective':f'{LOSS_ALPHA_NLL}*disease_nll + {LOSS_GAMMA_DEATH}*death_nll',
                     'weights_used_for_validation_and_checkpoint':'ema' if ema else 'raw','seed':SEED,'arm':ARM,
                     'effective_spec_sha256':signature,'test_parsed_before_lock':False}
            atomic_json(RUN_DIR/'cell6_training_summary_death_adjusted.json',summary)
            atomic_json(RUN_DIR/'run_selection_locked.json',{'schema':'gemma.run_selection.v7','checkpoint_sha256':sha256_file(best_path),
                        'effective_spec_sha256':signature,'best_val_loss':best,'test_parsed_before_lock':False,
                        'locked_at_utc':pd.Timestamp.now(tz='UTC').isoformat()})
            log('Training complete. Locking on validation results, then evaluating on test.' if RUN_ALL else 'Training complete. Use PHASE=select to lock the final seeds, then PHASE=evaluate.')
            return summary
        finally:
            handle.close()

    TRAINING_SUMMARY=run_arm_training()

    if ARM=='C' and PHASE=='train':
        _after=frozen_witness(GEMMA_TEXT)
        if _after!=FROZEN_WITNESS_BEFORE:raise RuntimeError('Frozen Gemma base changed during LoRA training.')
        ACCEPTANCE['G1c_frozen_base']={'pass':True,'before':FROZEN_WITNESS_BEFORE,'after':_after}
        save_acceptance()

# --- Notebook cell 19 ---
# GEMMA_PHASE=all: training is finished (or was already). Lock this run on validation results only, then switch to test evaluation.
if RUN_ALL:
    if not (COMPARISON_DIR / f"selection_{ARM}_locked.json").is_file():
        PHASE = "select"
        select_and_lock_arm()
    PHASE = "evaluate"
    print(f"Arm {ARM} seed {SEED} locked on validation results; evaluating on the test set.")

# --- Notebook cell 21 ---
if PHASE=='select':
    select_and_lock_arm()

# --- Notebook cell 23 ---
if PHASE=='evaluate':
    SOURCE_CRESCENT_RUN_DIR=CRESCENT_RUN_DIR
    SELECTION_LOCK=guard_test_access(CACHE_IDENTITY,check_this_run=True)
    LOCKED_CHECKPOINT_SHA256=sha256_file(RUN_DIR/PRIMARY_CHECKPOINT_NAME)
    EVALUATION_STARTED_UTC=pd.Timestamp.now(tz='UTC').isoformat()
    test_ds=load_locked_test_dataset()
    if ARM=='B':
        CACHE_META['test']=json.loads((CACHE_DIR/'test_cache_meta.json').read_text())
        if CACHE_META['test']['cache_binding_sha256']!=canonical_hash(expected_cache_binding(CACHE_IDENTITY)):
            raise ValueError('Test cache binding differs.')
        cached_test_ds=CachedDataset(test_ds,'test')
        test_loader=DataLoader(cached_test_ds,batch_size=cfg.BATCH_SIZE,shuffle=False,num_workers=0,collate_fn=lambda x:cached_collate(x,False))
    else:
        test_loader=DataLoader(test_ds,batch_size=EVAL_BATCH_C,shuffle=False,num_workers=0,collate_fn=lambda x:gemma_collate(x,False))

    def maybe_load_best_checkpoint(model):
        path=RUN_DIR/PRIMARY_CHECKPOINT_NAME
        if sha256_file(path)!=LOCKED_CHECKPOINT_SHA256:raise ValueError('Locked checkpoint changed.')
        state=torch.load(path,map_location=DEVICE,weights_only=False)
        if state['effective_spec_sha256']!=canonical_hash(effective_spec()) or state['model_input_hashes']!=dict(MODEL_INPUT_HASHES):
            raise ValueError('Checkpoint data/configuration/code differ.')
        if state['condition_names']!=list(condition_names) or state['delayed_entry_contract']!=delayed_entry_contract():
            raise ValueError('Checkpoint target definition differs.')
        load_checkpoint_state(model,state['model_state_dict']);model.eval()
        return str(path)
