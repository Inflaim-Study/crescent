# SurvTRACE-aligned — Post-entry likelihood training (50,000 updates) and EMA selection on validation NLL
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 10 ---
ENTRY_DELAY_MONTHS = cfg.ENTRY_DELAY_MONTHS
NORMALISE_DISEASE_CR_NLL_BY_CONDITION = cfg.NORMALISE_DISEASE_CR_NLL_BY_CONDITION
PROB_EPS = cfg.PROB_EPS
VAL_AUROC_HORIZONS_MONTHS = (24, 60)
MAX_ITERATIONS, WARMUP_ITERATIONS = cfg.MAX_ITERATIONS, cfg.WARMUP_ITERATIONS
EMA_DECAY = cfg.EMA_DECAY
USE_EMA = EMA_DECAY > 0
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

def run_log(message):
    print(message, flush=True)
    with (RUN_DIR / "cell6_full_training_console.log").open("a", encoding="utf-8") as handle:
        print(message, file=handle)

def loss_components(batch):
    out = model(batch)
    disease, ds = disease_competing_risk_nll(
        out["condition_rate"], batch["cr_event_type"], batch["cr_event_time_months"],
        batch["cr_at_risk"], batch["C_months"])
    death, ms = direct_death_nll(
        out["death_rate"], batch["death_observed_post_index"], batch["death_time_months"], batch["C_months"])
    return disease, death, ds["n_active_pairs"], ms["n_eligible_participants"]


@torch.no_grad()
def validation_nll():
    model.eval()
    disease_sum = death_sum = 0.
    pairs = people = 0
    started = time.monotonic()
    for batch_number, batch in enumerate(val_loader, 1):
        if batch_number % 25 == 0 or batch_number == len(val_loader):
            elapsed = time.monotonic() - started
            remaining = elapsed * (len(val_loader) - batch_number) / batch_number
            run_log(f"Validation batch {batch_number}/{len(val_loader)} | elapsed {elapsed:.0f}s | ETA {remaining:.0f}s")
        disease, death, n, m = loss_components(move_batch_to_device(batch, DEVICE))
        disease_sum += disease.item() * n
        death_sum += death.item() * m
        pairs += n
        people += m
    if not pairs or not people:
        raise ValueError("Empty validation risk set.")
    return disease_sum / pairs + cfg.LOSS_GAMMA_DEATH * death_sum / people

def save_atomic(payload, path):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)

def read_checkpoint(path, kind):
    # Only load this run's own checkpoints.
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("contract") != contract or state.get("kind") != kind:
        raise ValueError(f"Incompatible task, architecture, data or checkpoint type: {path}")
    return state

def maybe_load_best_checkpoint(model):
    path = EVAL_PATH if WORKFLOW == "evaluation_only" else RUN_DIR / cfg.PRIMARY_CHECKPOINT
    best = read_checkpoint(path, "minimum_validation_nll")
    model.load_state_dict(best["model_state_dict"], strict=True)
    print(f"Restored update {best['step']:,}, validation NLL {best['val_loss']:.6f}")
    return str(path)


VAL_TOP_BOTTOM_N = 5
VAL_MIN_CASES_FOR_RANKING = 20

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


def save_validation_table(frame, path):
    temporary = path.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)

def report_validation_performance(step):
    scores = validation_per_condition_auroc(val_loader)
    summary = validation_macro_auroc_summary(scores)
    ranking, rankable = top_bottom_conditions(scores)
    summary["val_rankable_conditions"] = rankable
    save_validation_table(scores, RUN_DIR / f"validation_per_condition_auroc_iteration_{step:06d}.csv")
    save_validation_table(ranking, RUN_DIR / f"validation_top5_bottom5_iteration_{step:06d}.csv")
    run_log(f"Full validation at {step:,} updates: {summary}")
    run_log(ranking[["group","rank","condition_name","auc_24m","auc_60m","auc_mean_24_60"]].to_string(index=False))
    return summary

def train_comparator():
    best_loss = np.inf
    start_step = 0
    history = []
    batch_cursor = 0
    epoch_rng = train_generator.get_state()
    last_path = RUN_DIR / "last_checkpoint.pt"
    if RESUME:
        state = read_checkpoint(last_path, "training_state")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        for target, value in zip(ema_params, state["ema"], strict=True):
            target.copy_(value)
        start_step, best_loss, history = state["step"], state["best_loss"], state["history"]
        batch_cursor, epoch_rng = state["batch_cursor"], state["epoch_rng"]
        train_generator.set_state(epoch_rng)
    iterator = iter(train_loader)
    if RESUME:
        # Recreate the shuffled epoch and then restore augmentation/dropout RNG.
        for _ in range(batch_cursor):
            next(iterator)
        torch.set_rng_state(state["torch_rng"])
        if DEVICE.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"], device=DEVICE)
        del state
    for step in range(start_step + 1, MAX_ITERATIONS + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch_rng = train_generator.get_state()
            iterator = iter(train_loader)
            batch_cursor = 0
            batch = next(iterator)
        batch_cursor += 1
        model.train()
        lr = learning_rate_for_iteration(step - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        d, m, _, _ = loss_components(move_batch_to_device(batch, DEVICE))
        loss = d + cfg.LOSS_GAMMA_DEATH * m
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at update {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP, error_if_nonfinite=True)
        optimizer.step()
        ema_update(step)
        if step % cfg.LOG_INTERVAL == 0:
            run_log(f"Update {step:,}/{MAX_ITERATIONS:,} | train NLL {loss.item():.6f}")
        if step % cfg.VALIDATION_INTERVAL != 0 and step != MAX_ITERATIONS:
            continue
        with ema_weights_in_model():
            val_loss = validation_nll()
            if not math.isfinite(val_loss):
                raise FloatingPointError("Non-finite validation NLL.")
            if val_loss < best_loss - cfg.MIN_DELTA:
                best_loss = val_loss
                save_atomic({"kind":"minimum_validation_nll", "contract":contract, "step":step,
                             "val_loss":val_loss, "model_state_dict":model.state_dict()},
                            RUN_DIR / cfg.PRIMARY_CHECKPOINT)
            row = {"step":step, "lr":lr, "val_loss":val_loss}
            if step % cfg.VAL_PERFORMANCE_INTERVAL == 0:
                row.update(report_validation_performance(step))
        history.append(row)
        run_log(row)
        save_atomic({"kind":"training_state", "contract":contract, "step":step, "best_loss":best_loss,
                     "history":history, "model_state_dict":model.state_dict(), "optimizer":optimizer.state_dict(),
                     "ema":ema_params, "epoch_rng":epoch_rng, "batch_cursor":batch_cursor,
                     "torch_rng":torch.get_rng_state(),
                     "cuda_rng":torch.cuda.get_rng_state(DEVICE) if DEVICE.type == "cuda" else None}, last_path)
        pd.DataFrame(history).to_csv(RUN_DIR / "training_history.csv", index=False)
    maybe_load_best_checkpoint(model)
    validation_per_condition_auroc(val_loader).to_csv(RUN_DIR / "validation_per_condition_auroc.csv", index=False)

if WORKFLOW == "train_and_evaluate":
    train_comparator()
