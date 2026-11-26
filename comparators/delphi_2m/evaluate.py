# Delphi-2M-inspired — Defines competing-risk metrics, dense-grid evaluation, tables, and participant bootstrap
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 23 ---
def make_condition_outcomes(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    at_risk = np.stack(frame["cr_at_risk"].to_numpy()).astype(float) > 0.5
    event_type = np.stack(frame["cr_event_type"].to_numpy()).astype(np.int8)
    event_time = np.stack(frame["cr_event_time_months"].to_numpy()).astype(float)
    censor_time_from_index = frame["C_months"].to_numpy(float)

    finite_event = np.isfinite(event_time)
    event_before_or_in_gap = (
        at_risk
        & finite_event
        & np.isin(event_type, [CR_TARGET_CONDITION, CR_TARGET_DEATH])
        & (event_time <= CFG.gap_months + 1e-8)
    )
    eligible = (
        at_risk
        & (censor_time_from_index[:, None] > CFG.gap_months + TIME_EPS_MONTHS)
        & ~event_before_or_in_gap
    )

    status = np.zeros_like(event_type, dtype=np.int8)
    exit_time_untruncated = np.broadcast_to(
        np.maximum(censor_time_from_index - CFG.gap_months, 0.0)[:, None],
        event_type.shape,
    ).copy()
    condition_after_gap = eligible & finite_event & (event_type == CR_TARGET_CONDITION) & (event_time > CFG.gap_months + 1e-8)
    death_after_gap = eligible & finite_event & (event_type == CR_TARGET_DEATH) & (event_time > CFG.gap_months + 1e-8)
    status[condition_after_gap] = 1
    status[death_after_gap] = 2
    exit_time_untruncated[condition_after_gap | death_after_gap] = (
        event_time[condition_after_gap | death_after_gap] - CFG.gap_months
    )
    exit_time = np.clip(
        exit_time_untruncated,
        0.0,
        max(CFG.report_horizons_months),
    )

    invalid = eligible & (~np.isfinite(exit_time) | (exit_time < -1e-6))
    if invalid.any():
        first = np.argwhere(invalid)[0].tolist()
        raise ValueError(f"Invalid condition lead-time outcome at row/condition {first}.")
    return {
        "eligible": eligible,
        "status": status,
        "exit_time": exit_time,
        "exit_time_untruncated": np.maximum(exit_time_untruncated, 0.0),
        "event_type": event_type,
        "event_time_from_index": event_time,
    }


def make_death_outcomes(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    censor_time_from_index = frame["C_months"].to_numpy(float)
    death_observed = frame["death_observed_post_index"].to_numpy(float) > 0.5
    death_time_from_index = frame["death_time_months"].to_numpy(float)
    finite_death = death_observed & np.isfinite(death_time_from_index)
    death_before_or_in_gap = finite_death & (death_time_from_index <= CFG.gap_months + 1e-8)
    eligible = (censor_time_from_index > CFG.gap_months + TIME_EPS_MONTHS) & ~death_before_or_in_gap
    status = np.zeros(len(frame), dtype=np.int8)
    exit_time_untruncated = np.maximum(
        censor_time_from_index - CFG.gap_months,
        0.0,
    )
    death_after_gap = eligible & finite_death & (death_time_from_index > CFG.gap_months + 1e-8)
    status[death_after_gap] = 1
    exit_time_untruncated[death_after_gap] = (
        death_time_from_index[death_after_gap] - CFG.gap_months
    )
    exit_time = np.clip(
        exit_time_untruncated,
        0.0,
        max(CFG.report_horizons_months),
    )
    return {
        "eligible": eligible,
        "status": status,
        "exit_time": exit_time,
        "exit_time_untruncated": np.maximum(exit_time_untruncated, 0.0),
    }

class CensoringKM:
    """O(n log n) Kaplan-Meier estimator of administrative-censoring survival G."""

    def __init__(self, exit_time: np.ndarray, censor_event: np.ndarray):
        time = np.asarray(exit_time, dtype=float)
        event = np.asarray(censor_event, dtype=bool)
        valid = np.isfinite(time) & (time >= 0)
        time, event = time[valid], event[valid]
        if time.size == 0:
            self.times = np.asarray([], dtype=float)
            self.survival_after = np.asarray([], dtype=float)
            return
        order = np.argsort(time, kind="mergesort")
        time = time[order]
        event = event[order].astype(np.int64)
        unique, first, counts = np.unique(time, return_index=True, return_counts=True)
        at_risk = time.size - first
        censor_counts = np.add.reduceat(event, first)
        factors = 1.0 - censor_counts / np.maximum(at_risk, 1)
        self.times = unique
        self.survival_after = np.cumprod(np.clip(factors, 0.0, 1.0))

    def _query(self, query: np.ndarray | float, side: str, clip: bool) -> np.ndarray:
        query_array = np.asarray(query, dtype=float)
        indices = np.searchsorted(self.times, query_array, side=side) - 1
        result = np.ones_like(query_array, dtype=float)
        valid = indices >= 0
        result[valid] = self.survival_after[indices[valid]]
        if clip:
            result = np.maximum(result, CFG.censor_survival_floor)
        return result

    def before_raw(self, query: np.ndarray | float) -> np.ndarray:
        return self._query(query, side="left", clip=False)

    def before(self, query: np.ndarray | float) -> np.ndarray:
        return self._query(query, side="left", clip=True)

    def at(self, query: np.ndarray | float) -> np.ndarray:
        return self._query(query, side="right", clip=True)


def binary_window(eligible: np.ndarray, status: np.ndarray, exit_time: np.ndarray, horizon: float, km: CensoringKM) -> Dict[str, Any]:
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=float)
    case = eligible & (status == 1) & (exit_time <= horizon + 1e-8)
    competing = eligible & (status == 2) & (exit_time <= horizon + 1e-8)
    observed_through = eligible & (exit_time >= horizon - 1e-8)
    known = case | competing | observed_through
    y = case.astype(float)
    weights = np.zeros_like(exit_time, dtype=float)
    event_or_competing = case | competing
    weights[event_or_competing] = 1.0 / km.before(exit_time[event_or_competing])
    weights[observed_through & ~event_or_competing] = 1.0 / float(km.before(horizon))

    raw_g = np.ones_like(exit_time, dtype=float)
    raw_g[event_or_competing] = km.before_raw(exit_time[event_or_competing])
    raw_g[observed_through & ~event_or_competing] = float(km.before_raw(horizon))
    truncated = known & (raw_g < CFG.censor_survival_floor)
    return {
        "eligible": eligible,
        "case": case,
        "competing": competing,
        "known": known,
        "observed_through": observed_through,
        "y": y,
        "weights": weights,
        "truncated": truncated,
        "raw_g": raw_g,
        "km": km,
    }


def aalen_johansen_cif(exit_time: np.ndarray, status: np.ndarray, horizon: float) -> float:
    """O(n log n) Aalen-Johansen CIF for primary status=1 and competing status=2."""
    time = np.asarray(exit_time, dtype=float)
    status = np.asarray(status, dtype=np.int8)
    valid = np.isfinite(time) & (time >= 0)
    time, status = time[valid], status[valid]
    if time.size == 0:
        return np.nan
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    status = status[order]
    unique, first, counts = np.unique(time, return_index=True, return_counts=True)
    at_risk = time.size - first
    primary = np.add.reduceat((status == 1).astype(np.int64), first)
    all_events = np.add.reduceat((status != 0).astype(np.int64), first)
    event_mask = (unique <= horizon + 1e-8) & (all_events > 0)
    if not np.any(event_mask):
        return 0.0
    hazards_all = all_events / np.maximum(at_risk, 1)
    survival_after = np.cumprod(np.clip(1.0 - hazards_all, 0.0, 1.0))
    survival_before = np.concatenate(([1.0], survival_after[:-1]))
    increments = survival_before * primary / np.maximum(at_risk, 1)
    return float(np.sum(increments[event_mask]))

# --- Notebook cell 25 ---
def ipcw_weight_diagnostics(window: Mapping[str, np.ndarray]) -> Dict[str, float]:
    known = np.asarray(window["known"], dtype=bool)
    weights = np.asarray(window["weights"], dtype=float)
    valid_weights = weights[known & np.isfinite(weights) & (weights > 0)]
    if valid_weights.size == 0:
        return {
            "ipcw_weight_sum": 0.0,
            "ipcw_weight_mean": np.nan,
            "ipcw_weight_max": np.nan,
            "ipcw_effective_sample_size": 0.0,
            "ipcw_truncated_n": int(np.asarray(window["truncated"], dtype=bool).sum()),
            "ipcw_truncated_fraction": np.nan,
        }
    weight_sum = float(valid_weights.sum())
    ess = weight_sum ** 2 / float(np.sum(valid_weights ** 2))
    truncated_n = int(np.asarray(window["truncated"], dtype=bool).sum())
    return {
        "ipcw_weight_sum": weight_sum,
        "ipcw_weight_mean": float(valid_weights.mean()),
        "ipcw_weight_max": float(valid_weights.max()),
        "ipcw_effective_sample_size": float(ess),
        "ipcw_truncated_n": truncated_n,
        "ipcw_truncated_fraction": float(truncated_n / max(int(known.sum()), 1)),
    }


def ipcw_ht_scores(
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    eligible_n: int,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-7, 1.0 - 1e-7)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction) & np.isfinite(weights) & (weights > 0)
    if eligible_n <= 0 or not valid.any():
        return {"brier_ipcw_ht": np.nan, "nbll_ipcw_ht": np.nan}
    brier = np.sum(weights[valid] * (y[valid] - prediction[valid]) ** 2) / eligible_n
    nbll = -np.sum(
        weights[valid]
        * (
            y[valid] * np.log(prediction[valid])
            + (1.0 - y[valid]) * np.log1p(-prediction[valid])
        )
    ) / eligible_n
    return {"brier_ipcw_ht": float(brier), "nbll_ipcw_ht": float(nbll)}


def competing_risk_auc(
    window: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> Dict[str, float]:
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = known & np.isfinite(prediction) & (weights > 0)
    result = {"auc_ipcw_cr": np.nan, "auprc_ipcw_cr": np.nan}
    if valid.any() and np.unique(y[valid].astype(int)).size == 2:
        result["auc_ipcw_cr"] = float(
            roc_auc_score(y[valid], prediction[valid], sample_weight=weights[valid])
        )
        result["auprc_ipcw_cr"] = float(
            average_precision_score(y[valid], prediction[valid], sample_weight=weights[valid])
        )
    return result


@njit(cache=False, nogil=True)
def _delphi_fenwick_add(tree, index, value):
    size = tree.shape[0]
    while index < size:
        tree[index] += value
        index += index & -index


@njit(cache=False, nogil=True)
def _delphi_fenwick_sum(tree, index):
    total = 0.0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=False, nogil=True)
def _delphi_wolbers_cindex_core(
    exit_time,
    status,
    risk_rank,
    g_before,
    g_after,
    tau,
    n_risk_ranks,
):
    """Exact O(n log n) Wolbers-style truncated competing-risk concordance."""
    n_rows = exit_time.shape[0]
    order = np.argsort(exit_time)
    weighted_concordant = 0.0
    weighted_pairs = 0.0
    raw_pairs = 0

    # Pair type A: a later observed time, including censoring tied at case time.
    tree_a = np.zeros(n_risk_ranks + 1, dtype=np.float64)
    total_a = 0.0
    end = n_rows
    while end > 0:
        start = end - 1
        current_time = exit_time[order[start]]
        while start > 0 and exit_time[order[start - 1]] == current_time:
            start -= 1
        for position in range(start, end):
            row = order[position]
            if status[row] == 0:
                _delphi_fenwick_add(tree_a, risk_rank[row], 1.0)
                total_a += 1.0
        if current_time <= tau:
            for position in range(start, end):
                row = order[position]
                if status[row] == 1:
                    rank = risk_rank[row]
                    lower = _delphi_fenwick_sum(tree_a, rank - 1)
                    through = _delphi_fenwick_sum(tree_a, rank)
                    tied = through - lower
                    pair_weight = 1.0 / (g_before[row] * g_after[row])
                    weighted_pairs += pair_weight * total_a
                    weighted_concordant += pair_weight * (lower + 0.5 * tied)
                    raw_pairs += int(total_a)
        for position in range(start, end):
            row = order[position]
            if status[row] != 0:
                _delphi_fenwick_add(tree_a, risk_rank[row], 1.0)
                total_a += 1.0
        end = start

    # Pair type B: competing event at or before the case time.
    tree_b = np.zeros(n_risk_ranks + 1, dtype=np.float64)
    total_b_weight = 0.0
    total_b_count = 0
    start = 0
    while start < n_rows:
        end = start + 1
        current_time = exit_time[order[start]]
        while end < n_rows and exit_time[order[end]] == current_time:
            end += 1
        for position in range(start, end):
            row = order[position]
            if status[row] == 2:
                comparator_weight = 1.0 / g_before[row]
                _delphi_fenwick_add(
                    tree_b,
                    risk_rank[row],
                    comparator_weight,
                )
                total_b_weight += comparator_weight
                total_b_count += 1
        if current_time <= tau:
            for position in range(start, end):
                row = order[position]
                if status[row] == 1:
                    rank = risk_rank[row]
                    lower = _delphi_fenwick_sum(tree_b, rank - 1)
                    through = _delphi_fenwick_sum(tree_b, rank)
                    tied = through - lower
                    case_weight = 1.0 / g_before[row]
                    weighted_pairs += case_weight * total_b_weight
                    weighted_concordant += case_weight * (lower + 0.5 * tied)
                    raw_pairs += total_b_count
        start = end

    estimate = (
        weighted_concordant / weighted_pairs
        if weighted_pairs > 0.0
        else np.nan
    )
    return estimate, weighted_concordant, weighted_pairs, raw_pairs


def competing_risk_cindex(
    eligible: np.ndarray,
    status: np.ndarray,
    exit_time: np.ndarray,
    prediction: np.ndarray,
    horizon: float,
    km: CensoringKM,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Exact Wolbers-style truncated IPCW competing-risk C-index.

    The implementation matches the attached main model's Cell 7B definition.
    No cases or comparators are sampled.
    """
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = (
        eligible
        & np.isfinite(exit_time)
        & np.isfinite(prediction)
        & np.isin(status, [0, 1, 2])
    )
    selected_time = exit_time[valid]
    selected_status = status[valid]
    selected_prediction = prediction[valid]
    if selected_time.size < 2:
        return {
            "cindex_ipcw_cr": np.nan,
            "cindex_pair_count": 0,
            "cindex_pair_weight_sum": 0.0,
            "cindex_cases_sampled": False,
            "cindex_comparators_sampled": False,
        }
    _, inverse_rank = np.unique(
        selected_prediction,
        return_inverse=True,
    )
    risk_rank = inverse_rank.astype(np.int64) + 1
    g_before = np.asarray(km.before(selected_time), dtype=np.float64)
    g_after = np.asarray(km.at(selected_time), dtype=np.float64)
    estimate, weighted_concordant, weighted_pairs, raw_pairs = (
        _delphi_wolbers_cindex_core(
            selected_time,
            selected_status,
            risk_rank,
            g_before,
            g_after,
            float(horizon),
            int(risk_rank.max()),
        )
    )
    return {
        "cindex_ipcw_cr": float(estimate),
        "cindex_pair_count": int(raw_pairs),
        "cindex_pair_weight_sum": float(weighted_pairs),
        "cindex_cases_sampled": False,
        "cindex_comparators_sampled": False,
    }


def validate_competing_risk_cindex_implementation() -> None:
    """Cross-check the optimized Wolbers core against a direct pair loop."""
    exit_time = np.asarray(
        [2.0, 4.0, 4.0, 5.0, 1.0, 3.0, 6.0, 2.0],
        dtype=np.float64,
    )
    status = np.asarray([1, 0, 2, 1, 2, 0, 1, 1], dtype=np.int8)
    risk = np.asarray(
        [0.8, 0.2, 0.2, 0.6, 0.1, 0.4, 0.6, 0.8],
        dtype=np.float64,
    )
    g_before = np.asarray(
        [0.95, 0.90, 0.90, 0.85, 1.00, 0.95, 0.80, 0.95],
        dtype=np.float64,
    )
    g_after = np.asarray(
        [0.95, 0.85, 0.85, 0.85, 1.00, 0.90, 0.80, 0.95],
        dtype=np.float64,
    )
    tau = 5.0
    _, inverse_rank = np.unique(risk, return_inverse=True)
    fast = _delphi_wolbers_cindex_core(
        exit_time,
        status,
        inverse_rank.astype(np.int64) + 1,
        g_before,
        g_after,
        tau,
        int(inverse_rank.max()) + 1,
    )
    concordant = 0.0
    weighted_pairs = 0.0
    raw_pairs = 0
    for case in range(len(exit_time)):
        if status[case] != 1 or exit_time[case] > tau:
            continue
        for comparator in range(len(exit_time)):
            if case == comparator:
                continue
            if (
                exit_time[case] < exit_time[comparator]
                or (
                    exit_time[case] == exit_time[comparator]
                    and status[comparator] == 0
                )
            ):
                weight = 1.0 / (
                    g_before[case] * g_after[case]
                )
            elif (
                status[comparator] == 2
                and exit_time[comparator] <= exit_time[case]
            ):
                weight = 1.0 / (
                    g_before[case] * g_before[comparator]
                )
            else:
                continue
            weighted_pairs += weight
            raw_pairs += 1
            if risk[case] > risk[comparator]:
                concordant += weight
            elif risk[case] == risk[comparator]:
                concordant += 0.5 * weight
    slow = (
        concordant / weighted_pairs,
        concordant,
        weighted_pairs,
        raw_pairs,
    )
    if not np.allclose(
        np.asarray(fast[:3], dtype=float),
        np.asarray(slow[:3], dtype=float),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ) or int(fast[3]) != int(slow[3]):
        raise AssertionError(
            "Optimized Wolbers C-index failed its direct-reference test: "
            f"fast={fast}, slow={slow}"
        )


validate_competing_risk_cindex_implementation()

def calibration_intercept_slope_cloglog(
    window: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> Dict[str, float]:
    empty = {"calibration_intercept_cloglog": np.nan, "calibration_slope_cloglog": np.nan}
    if not SCIPY_AVAILABLE:
        return empty
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.clip(np.asarray(prediction, dtype=float), 1e-7, 1.0 - 1e-7)
    valid = known & (weights > 0) & np.isfinite(prediction)
    if valid.sum() < 20 or np.unique(y[valid].astype(int)).size < 2:
        return empty
    x = np.log(-np.log1p(-prediction[valid]))
    target = y[valid]
    weight = weights[valid]

    def objective(parameters: np.ndarray) -> float:
        eta = np.clip(parameters[0] + parameters[1] * x, -20.0, 20.0)
        probability = np.clip(-np.expm1(-np.exp(eta)), 1e-10, 1.0 - 1e-10)
        return float(-np.sum(weight * (
            target * np.log(probability)
            + (1.0 - target) * np.log1p(-probability)
        )))

    fit = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not fit.success or not np.all(np.isfinite(fit.x)):
        return empty
    return {
        "calibration_intercept_cloglog": float(fit.x[0]),
        "calibration_slope_cloglog": float(fit.x[1]),
    }


def smooth_isotonic_calibration(
    window: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    grid_size: int = 101,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    empty = {"ici_ipcw_isotonic": np.nan, "e50_ipcw_isotonic": np.nan, "e90_ipcw_isotonic": np.nan}
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = known & (weights > 0) & np.isfinite(prediction)
    if valid.sum() < 20 or np.unique(prediction[valid]).size < 2:
        return empty, pd.DataFrame(columns=["predicted_risk", "observed_risk_smooth"])
    fit = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    fit.fit(prediction[valid], y[valid], sample_weight=weights[valid])
    fitted = fit.predict(prediction[valid])
    error = np.abs(fitted - prediction[valid])
    scalar = {
        "ici_ipcw_isotonic": float(np.average(error, weights=weights[valid])),
        "e50_ipcw_isotonic": float(np.quantile(error, 0.50)),
        "e90_ipcw_isotonic": float(np.quantile(error, 0.90)),
    }
    grid = np.linspace(float(prediction[valid].min()), float(prediction[valid].max()), grid_size)
    curve = pd.DataFrame({
        "predicted_risk": grid,
        "observed_risk_smooth": fit.predict(grid),
    })
    return scalar, curve


def grouped_aj_calibration(
    prediction: np.ndarray,
    eligible: np.ndarray,
    exit_time: np.ndarray,
    status: np.ndarray,
    horizon: int,
) -> List[Dict[str, Any]]:
    prediction = np.asarray(prediction, dtype=float)
    eligible = np.asarray(eligible, dtype=bool)
    values = prediction[eligible]
    times = np.asarray(exit_time, dtype=float)[eligible]
    states = np.asarray(status, dtype=np.int8)[eligible]
    if values.size < max(
        CFG.min_calibration_group_n,
        CFG.calibration_bins,
        CFG.min_calibration_bin_n,
    ):
        return []
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, CFG.calibration_bins + 1)))
    if edges.size <= 2:
        bin_id = np.zeros(values.size, dtype=int)
    else:
        bin_id = np.searchsorted(edges[1:-1], values, side="right")
    rows = []
    for current_bin in np.unique(bin_id):
        selected = bin_id == current_bin
        if int(selected.sum()) < CFG.min_calibration_bin_n:
            continue
        predicted_mean = float(values[selected].mean())
        observed = aalen_johansen_cif(times[selected], states[selected], float(horizon))
        rows.append({
            "calibration_bin": int(current_bin),
            "n": int(selected.sum()),
            "predicted_mean": predicted_mean,
            "predicted_median": float(np.median(values[selected])),
            "predicted_min": float(values[selected].min()),
            "predicted_max": float(values[selected].max()),
            "observed_aj_cif": observed,
            "predicted_minus_observed": predicted_mean - observed if np.isfinite(observed) else np.nan,
        })
    return rows


def integrate_metric(values: Sequence[float], horizons: Sequence[int]) -> float:
    values_array = np.asarray(values, dtype=float)
    horizon_array = np.asarray(horizons, dtype=float)
    valid = np.isfinite(values_array) & np.isfinite(horizon_array)
    if valid.sum() < 2:
        return np.nan
    x = horizon_array[valid]
    y = values_array[valid]
    order = np.argsort(x)
    x, y = x[order], y[order]
    span = float(x[-1] - x[0])
    if span <= 0:
        return np.nan
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / span)

# --- Notebook cell 27 ---
def evaluate_all_metrics(
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    train_outcomes = make_condition_outcomes(TRAIN_SAMPLES)
    test_outcomes = make_condition_outcomes(test_frame)
    condition_kms = [
        CensoringKM(
            train_outcomes["exit_time"][train_outcomes["eligible"][:, k], k],
            train_outcomes["status"][train_outcomes["eligible"][:, k], k] == 0,
        )
        for k in range(K)
    ]
    condition_cindex_kms = [
        CensoringKM(
            train_outcomes["exit_time_untruncated"][
                train_outcomes["eligible"][:, k], k
            ],
            train_outcomes["status"][train_outcomes["eligible"][:, k], k] == 0,
        )
        for k in range(K)
    ]

    train_null = np.full((K, len(DENSE_HORIZONS_MONTHS)), np.nan, dtype=float)
    for condition in range(K):
        selected = train_outcomes["eligible"][:, condition]
        for grid_index, horizon in enumerate(DENSE_HORIZONS_MONTHS):
            train_null[condition, grid_index] = aalen_johansen_cif(
                train_outcomes["exit_time"][selected, condition],
                train_outcomes["status"][selected, condition],
                float(horizon),
            )

    risk_sources = [("death_adjusted", "adjusted")]
    if predictions.get("unadjusted") is not None:
        risk_sources.append(("unadjusted", "unadjusted"))

    per_condition_rows: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    smooth_rows: List[Dict[str, Any]] = []
    integrated_rows: List[Dict[str, Any]] = []
    dense_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(
        np.random.SeedSequence([int(CFG.metric_seed), 2])
    )
    dense_index = {int(h): j for j, h in enumerate(DENSE_HORIZONS_MONTHS)}

    for risk_type, source in risk_sources:
        risk_cube = predictions[source]
        for horizon in CFG.report_horizons_months:
            grid_index = dense_index[int(horizon)]
            for condition in range(K):
                eligible = test_outcomes["eligible"][:, condition]
                status = test_outcomes["status"][:, condition]
                exit_time = test_outcomes["exit_time"][:, condition]
                km = condition_kms[condition]
                window = binary_window(eligible, status, exit_time, float(horizon), km)
                prediction = np.asarray(risk_cube[:, condition, grid_index], dtype=float)
                eligible_n = int(eligible.sum())

                auc = competing_risk_auc(window, prediction)
                scores = ipcw_ht_scores(
                    window["y"], prediction, window["weights"], eligible_n
                )
                null_prediction = np.full_like(prediction, train_null[condition, grid_index])
                null_scores = ipcw_ht_scores(
                    window["y"], null_prediction, window["weights"], eligible_n
                )
                null_brier = null_scores["brier_ipcw_ht"]
                scaled_brier = (
                    1.0 - scores["brier_ipcw_ht"] / null_brier
                    if np.isfinite(null_brier) and null_brier > 0
                    else np.nan
                )
                cindex = competing_risk_cindex(
                    eligible,
                    status,
                    test_outcomes["exit_time_untruncated"][:, condition],
                    prediction,
                    float(horizon),
                    condition_cindex_kms[condition],
                    rng,
                )
                observed = aalen_johansen_cif(
                    exit_time[eligible], status[eligible], float(horizon)
                )
                mean_prediction = float(prediction[eligible].mean()) if eligible_n else np.nan
                calibration_in_large = (
                    mean_prediction - observed
                    if np.isfinite(mean_prediction) and np.isfinite(observed)
                    else np.nan
                )
                oe_ratio = (
                    observed / mean_prediction
                    if np.isfinite(observed) and np.isfinite(mean_prediction) and mean_prediction > 0
                    else np.nan
                )
                calibration_parameters = calibration_intercept_slope_cloglog(
                    window, prediction
                )
                smooth_scalar, curve = smooth_isotonic_calibration(window, prediction)
                weight_diagnostics = ipcw_weight_diagnostics(window)
                valid_weight = (
                    window["known"]
                    & np.isfinite(window["weights"])
                    & (window["weights"] > 0)
                )
                weighted_event_rate = (
                    float(np.average(
                        window["y"][valid_weight],
                        weights=window["weights"][valid_weight],
                    ))
                    if np.any(valid_weight)
                    else np.nan
                )

                per_condition_rows.append({
                    "model_name": "Delphi2M_Canonical_AR_RateGrid",
                    "horizon_months": int(horizon),
                    "horizon_years": float(horizon / 12.0),
                    "risk_type": risk_type,
                    "condition_id": int(condition),
                    "condition_name": CONDITION_NAMES[condition],
                    "eligible_n": eligible_n,
                    "known_n": int(window["known"].sum()),
                    "event_n": int(window["case"].sum()),
                    "death_competing_n": int(window["competing"].sum()),
                    "observed_event_free_n": int((window["known"] & (window["y"] == 0)).sum()),
                    "admin_censored_before_horizon_n": int((eligible & ~window["known"]).sum()),
                    "mean_predicted_cif": mean_prediction,
                    "event_rate": weighted_event_rate,
                    "observed_aj_cif": observed,
                    "calibration_in_large_pred_minus_obs": calibration_in_large,
                    "observed_to_expected_ratio": oe_ratio,
                    **auc,
                    **scores,
                    "brier_null_ipcw_ht": null_brier,
                    "scaled_brier_ipcw_ht": scaled_brier,
                    **cindex,
                    **calibration_parameters,
                    **smooth_scalar,
                    **weight_diagnostics,
                })

                for calibration_row in grouped_aj_calibration(
                    prediction, eligible, exit_time, status, int(horizon)
                ):
                    calibration_rows.append({
                        "model_name": "Delphi2M_Canonical_AR_RateGrid",
                        "horizon_months": int(horizon),
                        "horizon_years": float(horizon / 12.0),
                        "risk_type": risk_type,
                        "condition_id": int(condition),
                        "condition_name": CONDITION_NAMES[condition],
                        **calibration_row,
                    })
                if not curve.empty:
                    curve.insert(0, "condition_name", CONDITION_NAMES[condition])
                    curve.insert(0, "condition_id", int(condition))
                    curve.insert(0, "risk_type", risk_type)
                    curve.insert(0, "horizon_months", int(horizon))
                    smooth_rows.extend(curve.to_dict("records"))

        # Dense-grid integrated metrics for each condition.
        for condition in range(K):
            eligible = test_outcomes["eligible"][:, condition]
            status = test_outcomes["status"][:, condition]
            exit_time = test_outcomes["exit_time"][:, condition]
            km = condition_kms[condition]
            eligible_n = int(eligible.sum())
            brier_values, null_brier_values = [], []
            nbll_values, auc_values, calibration_values = [], [], []
            for grid_index, horizon in enumerate(DENSE_HORIZONS_MONTHS):
                window = binary_window(eligible, status, exit_time, float(horizon), km)
                prediction = np.asarray(risk_cube[:, condition, grid_index], dtype=float)
                scores = ipcw_ht_scores(
                    window["y"], prediction, window["weights"], eligible_n
                )
                null_prediction = np.full_like(prediction, train_null[condition, grid_index])
                null_scores = ipcw_ht_scores(
                    window["y"], null_prediction, window["weights"], eligible_n
                )
                discrimination = competing_risk_auc(window, prediction)
                auc = discrimination["auc_ipcw_cr"]
                observed = aalen_johansen_cif(
                    exit_time[eligible], status[eligible], float(horizon)
                )
                mean_prediction = float(prediction[eligible].mean()) if eligible_n else np.nan
                brier_values.append(scores["brier_ipcw_ht"])
                null_brier_values.append(null_scores["brier_ipcw_ht"])
                nbll_values.append(scores["nbll_ipcw_ht"])
                auc_values.append(auc)
                calibration_values.append(
                    abs(mean_prediction - observed)
                    if np.isfinite(mean_prediction) and np.isfinite(observed)
                    else np.nan
                )
                if risk_type == "death_adjusted":
                    dense_rows.append({
                        "model_name": "Delphi2M_Canonical_AR_RateGrid",
                        "risk_type": "death_adjusted",
                        "condition_id": int(condition),
                        "condition_name": CONDITION_NAMES[condition],
                        "horizon_months": int(horizon),
                        "eligible_n": int(eligible_n),
                        "known_n": int(window["known"].sum()),
                        "event_n": int(window["case"].sum()),
                        "death_competing_n": int(window["competing"].sum()),
                        **discrimination,
                        **scores,
                        "brier_null_ipcw_ht": null_scores["brier_ipcw_ht"],
                        "mean_predicted_cif": mean_prediction,
                        "observed_aj_cif": observed,
                        "absolute_calibration_error": (
                            abs(mean_prediction - observed)
                            if np.isfinite(mean_prediction) and np.isfinite(observed)
                            else np.nan
                        ),
                    })

            ibs = integrate_metric(brier_values, DENSE_HORIZONS_MONTHS)
            ibs_null = integrate_metric(null_brier_values, DENSE_HORIZONS_MONTHS)
            integrated_rows.append({
                "model_name": "Delphi2M_Canonical_AR_RateGrid",
                "risk_type": risk_type,
                "condition_id": int(condition),
                "condition_name": CONDITION_NAMES[condition],
                "integration_start_months": int(DENSE_HORIZONS_MONTHS[0]),
                "integration_end_months": int(DENSE_HORIZONS_MONTHS[-1]),
                "grid_step_months": BIN_WIDTH_MONTHS,
                "n_grid_points": len(DENSE_HORIZONS_MONTHS),
                "ibs_ipcw_ht": ibs,
                "ibs_null_ipcw_ht": ibs_null,
                "scaled_ibs_ipcw_ht": (
                    1.0 - ibs / ibs_null
                    if np.isfinite(ibs) and np.isfinite(ibs_null) and ibs_null > 0
                    else np.nan
                ),
                "inbll_ipcw_ht": integrate_metric(nbll_values, DENSE_HORIZONS_MONTHS),
                "iauc_ipcw_cr": integrate_metric(auc_values, DENSE_HORIZONS_MONTHS),
                "integrated_abs_calibration_error": integrate_metric(
                    calibration_values, DENSE_HORIZONS_MONTHS
                ),
            })

    per_condition = pd.DataFrame(per_condition_rows)
    calibration = pd.DataFrame(calibration_rows)
    smooth = pd.DataFrame(smooth_rows)
    integrated = pd.DataFrame(integrated_rows)
    dense = pd.DataFrame(dense_rows)

    # Main-model compatibility aliases are additive: the richer canonical
    # competing-risk fields remain available for Delphi-specific analyses.
    per_condition["admin_censored_n"] = (
        per_condition["known_n"]
        - per_condition["event_n"]
        - per_condition["death_competing_n"]
    ).astype(int)
    per_condition["mean_predicted_risk"] = per_condition["mean_predicted_cif"]
    per_condition["auroc"] = per_condition["auc_ipcw_cr"]
    per_condition["auprc"] = per_condition["auprc_ipcw_cr"]
    per_condition["brier_ipcw"] = per_condition["brier_ipcw_ht"]
    per_condition["nbll_ipcw"] = per_condition["nbll_ipcw_ht"]
    per_condition["weight_sum"] = per_condition["ipcw_weight_sum"]
    per_condition["normalization_n"] = per_condition["eligible_n"]
    # Preserve the richer Delphi point estimates in the extended result. The
    # strict main-model view below disables CTD and blanks canonical-only
    # competing-risk fields for its unadjusted rows.
    per_condition["ctd_ipcw"] = per_condition["cindex_ipcw_cr"]
    per_condition["ctd_status"] = "exact_wolbers_competing_risk"

    # Macro summaries preserve equal condition weighting.
    macro_rows: List[Dict[str, Any]] = []
    summary_metrics = [
        "auc_ipcw_cr", "auprc_ipcw_cr", "cindex_ipcw_cr",
        "brier_ipcw_ht", "scaled_brier_ipcw_ht", "nbll_ipcw_ht",
        "calibration_in_large_pred_minus_obs", "calibration_slope_cloglog",
        "ici_ipcw_isotonic", "observed_aj_cif", "mean_predicted_cif",
    ]
    for (risk_type, horizon), group in per_condition.groupby(["risk_type", "horizon_months"], sort=True):
        row: Dict[str, Any] = {
            "model_name": "Delphi2M_Canonical_AR_RateGrid",
            "risk_type": risk_type,
            "horizon_months": int(horizon),
            "horizon_years": float(horizon / 12.0),
            "n_conditions": K,
            "conditions_with_auc": int(group["auc_ipcw_cr"].notna().sum()),
        }
        for metric in summary_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"macro_mean_{metric}"] = float(values.mean(skipna=True))
            row[f"macro_median_{metric}"] = float(values.median(skipna=True))
            row[f"macro_q25_{metric}"] = float(values.quantile(0.25))
            row[f"macro_q75_{metric}"] = float(values.quantile(0.75))
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    # Death-head metrics use the same lead-time gap and training-split censoring KM.
    train_death = make_death_outcomes(TRAIN_SAMPLES)
    test_death = make_death_outcomes(test_frame)
    death_km = CensoringKM(
        train_death["exit_time"][train_death["eligible"]],
        train_death["status"][train_death["eligible"]] == 0,
    )
    death_cindex_km = CensoringKM(
        train_death["exit_time_untruncated"][train_death["eligible"]],
        train_death["status"][train_death["eligible"]] == 0,
    )
    death_rows: List[Dict[str, Any]] = []
    death_brier, death_null_brier, death_nbll, death_auc = [], [], [], []
    for grid_index, horizon in enumerate(DENSE_HORIZONS_MONTHS):
        window = binary_window(
            test_death["eligible"], test_death["status"], test_death["exit_time"],
            float(horizon), death_km
        )
        prediction = np.asarray(predictions["death"][:, grid_index], dtype=float)
        eligible_n = int(test_death["eligible"].sum())
        scores = ipcw_ht_scores(window["y"], prediction, window["weights"], eligible_n)
        train_null_death = aalen_johansen_cif(
            train_death["exit_time"][train_death["eligible"]],
            train_death["status"][train_death["eligible"]],
            float(horizon),
        )
        null_prediction = np.full_like(prediction, train_null_death)
        null_scores = ipcw_ht_scores(window["y"], null_prediction, window["weights"], eligible_n)
        auc = competing_risk_auc(window, prediction)
        observed = aalen_johansen_cif(
            test_death["exit_time"][test_death["eligible"]],
            test_death["status"][test_death["eligible"]],
            float(horizon),
        )
        mean_prediction = float(prediction[test_death["eligible"]].mean()) if eligible_n else np.nan
        death_brier.append(scores["brier_ipcw_ht"])
        death_null_brier.append(null_scores["brier_ipcw_ht"])
        death_nbll.append(scores["nbll_ipcw_ht"])
        death_auc.append(auc["auc_ipcw_cr"])

        if int(horizon) in CFG.report_horizons_months:
            cindex = competing_risk_cindex(
                test_death["eligible"],
                test_death["status"],
                test_death["exit_time_untruncated"],
                prediction,
                float(horizon),
                death_cindex_km,
                rng,
            )
            death_rows.append({
                "model_name": "Delphi2M_Canonical_AR_RateGrid",
                "horizon_months": int(horizon),
                "horizon_years": float(horizon / 12.0),
                "eligible_n": eligible_n,
                "known_n": int(window["known"].sum()),
                "death_event_n": int(window["case"].sum()),
                "admin_censored_before_horizon_n": int((test_death["eligible"] & ~window["known"]).sum()),
                "mean_predicted_death_risk": mean_prediction,
                "observed_death_risk": observed,
                "calibration_in_large_pred_minus_obs": (
                    mean_prediction - observed
                    if np.isfinite(mean_prediction) and np.isfinite(observed)
                    else np.nan
                ),
                "observed_to_expected_ratio": (
                    observed / mean_prediction
                    if np.isfinite(observed) and np.isfinite(mean_prediction) and mean_prediction > 0
                    else np.nan
                ),
                **auc,
                **scores,
                "brier_null_ipcw_ht": null_scores["brier_ipcw_ht"],
                "scaled_brier_ipcw_ht": (
                    1.0 - scores["brier_ipcw_ht"] / null_scores["brier_ipcw_ht"]
                    if np.isfinite(null_scores["brier_ipcw_ht"]) and null_scores["brier_ipcw_ht"] > 0
                    else np.nan
                ),
                **cindex,
                **calibration_intercept_slope_cloglog(window, prediction),
                **ipcw_weight_diagnostics(window),
            })

    death = pd.DataFrame(death_rows)
    death_ibs = integrate_metric(death_brier, DENSE_HORIZONS_MONTHS)
    death_ibs_null = integrate_metric(death_null_brier, DENSE_HORIZONS_MONTHS)
    death_integrated = pd.DataFrame([{
        "model_name": "Delphi2M_Canonical_AR_RateGrid",
        "integration_start_months": int(DENSE_HORIZONS_MONTHS[0]),
        "integration_end_months": int(DENSE_HORIZONS_MONTHS[-1]),
        "ibs_ipcw_ht": death_ibs,
        "ibs_null_ipcw_ht": death_ibs_null,
        "scaled_ibs_ipcw_ht": (
            1.0 - death_ibs / death_ibs_null
            if np.isfinite(death_ibs) and np.isfinite(death_ibs_null) and death_ibs_null > 0
            else np.nan
        ),
        "inbll_ipcw_ht": integrate_metric(death_nbll, DENSE_HORIZONS_MONTHS),
        "iauc_ipcw": integrate_metric(death_auc, DENSE_HORIZONS_MONTHS),
    }])

    return {
        "per_condition": per_condition,
        "integrated": integrated,
        "calibration": calibration,
        "smooth_calibration": smooth,
        "macro": macro,
        "death": death,
        "death_integrated": death_integrated,
        "dense": dense,
    }

# --- Notebook cell 29 ---
def save_compatibility_prediction_npz(
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> None:
    dense_index = {int(h): j for j, h in enumerate(DENSE_HORIZONS_MONTHS)}
    payload: Dict[str, Any] = {
        "eid": test_frame["eid"].astype(str).to_numpy(),
        "condition_names": np.asarray(CONDITION_NAMES, dtype=object),
        "horizons_months": np.asarray(CFG.report_horizons_months, dtype=np.int32),
        "entry_delay_months": np.asarray(CFG.gap_months, dtype=np.float32),
        "gap_months": np.asarray(CFG.gap_months, dtype=np.float32),
        "index_age_years": test_frame["index_age_years"].to_numpy(np.float32),
        "at_risk": np.stack(test_frame["at_risk"].to_numpy()).astype(np.float32),
        "event_obs": np.stack(test_frame["event_obs"].to_numpy()).astype(np.float32),
        "event_bin": np.stack(test_frame["event_bin"].to_numpy()).astype(np.int16),
        "t_tilde": np.stack(test_frame["t_tilde"].to_numpy()).astype(np.float32),
        "C_months": test_frame["C_months"].to_numpy(np.float32),
        "C_bin": test_frame["C_bin"].to_numpy(np.int16),
        "death_observed_post_index": test_frame["death_observed_post_index"].to_numpy(np.float32),
        "death_time_months": test_frame["death_time_months"].to_numpy(np.float32),
        "death_bin": test_frame["death_bin"].to_numpy(np.int16),
    }
    for horizon in CFG.report_horizons_months:
        j = dense_index[int(horizon)]
        payload[f"risk_death_adjusted_h{horizon}"] = np.asarray(
            predictions["adjusted"][:, :, j], dtype=np.float32
        )
        payload[f"risk_unadjusted_h{horizon}"] = np.asarray(
            predictions["unadjusted"][:, :, j], dtype=np.float32
        )
        payload[f"risk_death_h{horizon}"] = np.asarray(
            predictions["death"][:, j], dtype=np.float32
        )
    np.savez_compressed(RUN_DIR / "test_prediction_arrays_death_adjusted.npz", **payload)


def weighted_quantile_compatibility(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    if not np.any(valid):
        return np.nan
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    threshold = float(quantile) * float(cumulative[-1])
    return float(values[min(np.searchsorted(cumulative, threshold), len(values) - 1)])


def build_main_compatibility_tables(
    result: Mapping[str, pd.DataFrame],
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    """Create the attached main model's public test-table views."""
    train_outcomes = make_condition_outcomes(TRAIN_SAMPLES)
    test_outcomes = make_condition_outcomes(test_frame)
    condition_kms = [
        CensoringKM(
            train_outcomes["exit_time"][train_outcomes["eligible"][:, k], k],
            train_outcomes["status"][train_outcomes["eligible"][:, k], k] == 0,
        )
        for k in range(K)
    ]
    dense_index = {int(value): i for i, value in enumerate(DENSE_HORIZONS_MONTHS)}
    micro_rows: List[Dict[str, Any]] = []
    for horizon in CFG.report_horizons_months:
        grid_index = dense_index[int(horizon)]
        for risk_type, source in (
            ("death_adjusted", "adjusted"),
            ("unadjusted", "unadjusted"),
        ):
            y_parts, prediction_parts = [], []
            known_parts, weight_parts = [], []
            eligible_pairs = 0
            condition_events = 0
            death_events = 0
            for condition in range(K):
                eligible = test_outcomes["eligible"][:, condition]
                status = test_outcomes["status"][:, condition]
                exit_time = test_outcomes["exit_time"][:, condition]
                window = binary_window(
                    eligible,
                    status,
                    exit_time,
                    float(horizon),
                    condition_kms[condition],
                )
                y_parts.append(window["y"])
                known_parts.append(window["known"])
                weight_parts.append(window["weights"])
                prediction_parts.append(np.asarray(
                    predictions[source][:, condition, grid_index], dtype=float
                ))
                eligible_pairs += int(eligible.sum())
                condition_events += int(window["case"].sum())
                death_events += int(window["competing"].sum())
            y = np.concatenate(y_parts)
            prediction = np.concatenate(prediction_parts)
            known = np.concatenate(known_parts)
            weights = np.concatenate(weight_parts)
            valid = (
                known
                & np.isfinite(prediction)
                & np.isfinite(weights)
                & (weights > 0)
            )
            window_flat = {"known": known, "y": y, "weights": weights}
            discrimination = competing_risk_auc(window_flat, prediction)
            scores = ipcw_ht_scores(y, prediction, weights, eligible_pairs)
            event_rate = (
                float(np.average(y[valid], weights=weights[valid]))
                if np.any(valid)
                else np.nan
            )
            micro_rows.append({
                "horizon_months": int(horizon),
                "risk_type": risk_type,
                "auroc": discrimination["auc_ipcw_cr"],
                "auprc": discrimination["auprc_ipcw_cr"],
                "event_rate": event_rate,
                "brier_ipcw": scores["brier_ipcw_ht"],
                "nbll_ipcw": scores["nbll_ipcw_ht"],
                "weight_sum": float(weights[valid].sum()) if np.any(valid) else 0.0,
                "normalization_n": int(eligible_pairs),
                "eligible_pairs": int(eligible_pairs),
                "known_pairs": int(known.sum()),
                "condition_events": int(condition_events),
                "death_competing_events": int(death_events),
            })
    micro = pd.DataFrame(micro_rows)

    per_condition = result["per_condition"].copy()
    per_condition["absolute_calibration_error"] = np.abs(
        per_condition["calibration_in_large_pred_minus_obs"]
    )
    # Match the attached main point-table contract exactly. CTD is deliberately
    # disabled in that public table; Delphi's exact Wolbers C-index remains in
    # test_per_condition_metrics_delphi_extended.csv and the dedicated CI files.
    per_condition["ctd_ipcw"] = np.nan
    per_condition["ctd_status"] = "disabled_not_validly_implemented"
    canonical_only_columns = [
        "auc_ipcw_cr", "auprc_ipcw_cr", "brier_ipcw_ht",
        "brier_null_ipcw_ht", "nbll_ipcw_ht",
    ]
    per_condition.loc[
        per_condition["risk_type"].eq("unadjusted"),
        canonical_only_columns,
    ] = np.nan

    calibration = result["calibration"].copy()
    if calibration.empty and "predicted_minus_observed" not in calibration.columns:
        calibration = pd.DataFrame(columns=[
            "horizon_months", "risk_type", "condition_id", "condition_name",
            "calibration_bin", "n", "predicted_mean", "predicted_min",
            "predicted_max", "observed_aj_cif", "predicted_minus_observed",
        ])
    calibration["observed_minus_predicted"] = -pd.to_numeric(
        calibration["predicted_minus_observed"], errors="coerce"
    )
    calibration_columns = [
        "horizon_months", "risk_type", "condition_id", "condition_name",
        "calibration_bin", "n", "predicted_mean", "predicted_min",
        "predicted_max", "observed_aj_cif", "observed_minus_predicted",
    ]
    calibration_compatibility = calibration[calibration_columns].copy()
    calibration_error_rows: List[Dict[str, Any]] = []
    for horizon in CFG.report_horizons_months:
        for risk_type in ("death_adjusted", "unadjusted"):
            group = calibration_compatibility[
                calibration_compatibility["horizon_months"].eq(int(horizon))
                & calibration_compatibility["risk_type"].eq(risk_type)
            ]
            absolute_error = np.abs(
                pd.to_numeric(group["observed_minus_predicted"], errors="coerce")
                .to_numpy(float)
            )
            weights = pd.to_numeric(group["n"], errors="coerce").to_numpy(float)
            valid = (
                np.isfinite(absolute_error)
                & np.isfinite(weights)
                & (weights > 0)
            )
            calibration_error_rows.append({
                "horizon_months": int(horizon),
                "risk_type": str(risk_type),
                "ici": (
                    float(np.average(
                        absolute_error[valid], weights=weights[valid]
                    ))
                    if np.any(valid)
                    else np.nan
                ),
                "e50": weighted_quantile_compatibility(
                    absolute_error[valid], weights[valid], 0.50
                ),
                "e90": weighted_quantile_compatibility(
                    absolute_error[valid], weights[valid], 0.90
                ),
                "n_calibration_bins": int(valid.sum()),
                "calibration_weight_n": (
                    int(weights[valid].sum()) if np.any(valid) else 0
                ),
            })
    calibration_error = pd.DataFrame(calibration_error_rows)

    death = result["death"].copy()
    train_death = make_death_outcomes(TRAIN_SAMPLES)
    test_death = make_death_outcomes(test_frame)
    death_km = CensoringKM(
        train_death["exit_time"][train_death["eligible"]],
        train_death["status"][train_death["eligible"]] == 0,
    )
    death_event_rates: Dict[int, float] = {}
    for horizon in CFG.report_horizons_months:
        window = binary_window(
            test_death["eligible"],
            test_death["status"],
            test_death["exit_time"],
            float(horizon),
            death_km,
        )
        valid = window["known"] & (window["weights"] > 0)
        death_event_rates[int(horizon)] = (
            float(np.average(
                window["y"][valid], weights=window["weights"][valid]
            ))
            if np.any(valid)
            else np.nan
        )
    death["event_rate"] = death["horizon_months"].map(death_event_rates)
    death["auroc"] = death["auc_ipcw_cr"]
    death["auprc"] = death["auprc_ipcw_cr"]
    death["brier_ipcw"] = death["brier_ipcw_ht"]
    death["nbll_ipcw"] = death["nbll_ipcw_ht"]
    death["weight_sum"] = death["ipcw_weight_sum"]
    death["normalization_n"] = death["eligible_n"]
    death["ctd_ipcw"] = np.nan
    death["ctd_status"] = "disabled_not_validly_implemented"

    summary_rows: List[Dict[str, Any]] = []
    for horizon in CFG.report_horizons_months:
        for risk_type in ("death_adjusted", "unadjusted"):
            micro_selected = micro[
                micro["horizon_months"].eq(int(horizon))
                & micro["risk_type"].eq(risk_type)
            ].iloc[0]
            condition_selected = per_condition[
                per_condition["horizon_months"].eq(int(horizon))
                & per_condition["risk_type"].eq(risk_type)
            ]
            calibration_selected = calibration_error[
                calibration_error["horizon_months"].eq(int(horizon))
                & calibration_error["risk_type"].eq(risk_type)
            ]
            summary_rows.append({
                "horizon_months": int(horizon),
                "risk_type": risk_type,
                "micro_auroc": float(micro_selected["auroc"]),
                "micro_auprc": float(micro_selected["auprc"]),
                "micro_brier_ipcw": float(micro_selected["brier_ipcw"]),
                "micro_nbll_ipcw": float(micro_selected["nbll_ipcw"]),
                "macro_auroc": float(condition_selected["auroc"].mean(skipna=True)),
                "macro_auprc": float(condition_selected["auprc"].mean(skipna=True)),
                "macro_brier_ipcw": float(
                    condition_selected["brier_ipcw"].mean(skipna=True)
                ),
                "macro_nbll_ipcw": float(
                    condition_selected["nbll_ipcw"].mean(skipna=True)
                ),
                "macro_ctd_ipcw": float(
                    condition_selected["ctd_ipcw"].mean(skipna=True)
                ),
                "ctd_status": "disabled_not_validly_implemented",
                "mean_predicted_risk": float(
                    condition_selected["mean_predicted_risk"].mean(skipna=True)
                ),
                "n_conditions_with_auroc": int(
                    condition_selected["auroc"].notna().sum()
                ),
                "n_conditions": int(K),
                "ici": (
                    float(calibration_selected["ici"].iloc[0])
                    if len(calibration_selected)
                    else np.nan
                ),
                "e50": (
                    float(calibration_selected["e50"].iloc[0])
                    if len(calibration_selected)
                    else np.nan
                ),
                "e90": (
                    float(calibration_selected["e90"].iloc[0])
                    if len(calibration_selected)
                    else np.nan
                ),
            })
    summary = pd.DataFrame(summary_rows)
    integrated_summary_rows: List[Dict[str, Any]] = []
    numeric_summary_columns = [
        "micro_auroc", "micro_auprc", "micro_brier_ipcw",
        "micro_nbll_ipcw", "macro_auroc", "macro_auprc",
        "macro_brier_ipcw", "macro_nbll_ipcw", "macro_ctd_ipcw",
        "mean_predicted_risk", "ici", "e50", "e90",
    ]
    for risk_type in ("death_adjusted", "unadjusted"):
        selected = summary[summary["risk_type"].eq(risk_type)]
        row: Dict[str, Any] = {
            "horizon_months": "integrated_mean_over_reported_horizons",
            "risk_type": risk_type,
            "ctd_status": "disabled_not_validly_implemented",
            "n_conditions_with_auroc": int(
                selected["n_conditions_with_auroc"].mean()
            ),
            "n_conditions": int(K),
        }
        for column in numeric_summary_columns:
            row[column] = float(selected[column].mean(skipna=True))
        integrated_summary_rows.append(row)
    summary = pd.concat(
        [summary, pd.DataFrame(integrated_summary_rows)], ignore_index=True
    )
    per_condition_columns = [
        "horizon_months", "risk_type", "condition_id", "condition_name",
        "eligible_n", "known_n", "event_n", "death_competing_n",
        "admin_censored_n", "mean_predicted_risk", "auroc", "auprc",
        "event_rate", "brier_ipcw", "nbll_ipcw", "weight_sum",
        "normalization_n", "ctd_ipcw", "ctd_status", "auc_ipcw_cr",
        "auprc_ipcw_cr", "brier_ipcw_ht", "brier_null_ipcw_ht",
        "nbll_ipcw_ht",
    ]
    death_columns = [
        "horizon_months", "eligible_n", "known_n", "death_event_n",
        "mean_predicted_death_risk", "auroc", "auprc", "event_rate",
        "brier_ipcw", "nbll_ipcw", "weight_sum", "normalization_n",
        "ctd_ipcw", "ctd_status",
    ]
    summary_columns = [
        "horizon_months", "risk_type", "micro_auroc", "micro_auprc",
        "micro_brier_ipcw", "micro_nbll_ipcw", "macro_auroc",
        "macro_auprc", "macro_brier_ipcw", "macro_nbll_ipcw",
        "macro_ctd_ipcw", "ctd_status", "mean_predicted_risk",
        "n_conditions_with_auroc", "n_conditions", "ici", "e50", "e90",
    ]
    return {
        "micro": micro,
        "per_condition": per_condition[per_condition_columns].copy(),
        "calibration": calibration_compatibility,
        "calibration_error": calibration_error,
        "death": death[death_columns].copy(),
        "summary": summary[summary_columns].copy(),
    }


def export_figure2_calibration_tables(
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Write the A10 condition × horizon × sex calibration schemas.

    This mirrors the main notebook's Figure-2 export: Aalen-Johansen observed
    condition incidence with death competing, ten quantile bins, minimum
    stratum size 100, and minimum retained-bin size 50.
    """
    model_name = "Delphi2M_Canonical_AR_RateGrid"
    if "gender_idx" not in test_frame.columns:
        raise KeyError(
            "canonical_test_samples.pkl contains no gender_idx for "
            "sex-stratified calibration."
        )
    gender = pd.to_numeric(test_frame["gender_idx"], errors="coerce")
    # Locked A10 mapping: 0=Female, 1=Male, 2=Unknown.
    sex = (
        gender.map({0: "Female", 1: "Male", 2: "Unknown"})
        .fillna("Unknown")
        .to_numpy(dtype=object)
    )
    if not np.any(sex == "Male") or not np.any(sex == "Female"):
        raise ValueError(
            "A10 sex-stratified calibration requires both Male and Female "
            "test participants."
        )
    strata = {
        "All": np.ones(len(test_frame), dtype=bool),
        "Male": sex == "Male",
        "Female": sex == "Female",
    }
    outcomes = make_condition_outcomes(test_frame)
    empty_km = CensoringKM(
        np.asarray([], dtype=float), np.asarray([], dtype=bool)
    )
    dense_index = {
        int(horizon): index
        for index, horizon in enumerate(DENSE_HORIZONS_MONTHS)
    }
    missing_horizons = sorted(
        set(map(int, CFG.report_horizons_months)).difference(dense_index)
    )
    if missing_horizons:
        raise RuntimeError(
            "Dense predictions lack Figure-2 horizons: "
            f"{missing_horizons}."
        )

    decile_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    adjusted = predictions["adjusted"]
    for horizon in map(int, CFG.report_horizons_months):
        grid_index = dense_index[horizon]
        for condition_id, condition_name in enumerate(CONDITION_NAMES):
            window = binary_window(
                outcomes["eligible"][:, condition_id],
                outcomes["status"][:, condition_id],
                outcomes["exit_time"][:, condition_id],
                float(horizon),
                empty_km,
            )
            for stratum_name, participant_mask in strata.items():
                selected = participant_mask & outcomes["eligible"][:, condition_id]
                eligible_n = int(selected.sum())
                if eligible_n < int(CFG.min_calibration_group_n):
                    continue
                scores = np.asarray(
                    adjusted[selected, condition_id, grid_index], dtype=float
                )
                if not np.all(np.isfinite(scores)):
                    raise ValueError(
                        "Non-finite predictions in sex-stratified calibration."
                    )
                times = outcomes["exit_time"][selected, condition_id].astype(float)
                statuses = outcomes["status"][selected, condition_id].astype(np.int16)
                known = window["known"][selected]
                cases = window["case"][selected]
                competing = window["competing"][selected]
                edges = np.unique(
                    np.quantile(
                        scores,
                        np.linspace(0.0, 1.0, int(CFG.calibration_bins) + 1),
                    )
                )
                if edges.size <= 2:
                    bin_id = np.zeros(eligible_n, dtype=np.int16)
                    n_bins_actual = 1
                else:
                    bin_id = np.searchsorted(
                        edges[1:-1], scores, side="right"
                    ).astype(np.int16)
                    n_bins_actual = int(bin_id.max()) + 1

                condition_rows: List[Dict[str, Any]] = []
                for calibration_bin in range(n_bins_actual):
                    in_bin = bin_id == calibration_bin
                    bin_n = int(in_bin.sum())
                    if bin_n < int(CFG.min_calibration_bin_n):
                        continue
                    predicted_mean = float(scores[in_bin].mean())
                    observed = aalen_johansen_cif(
                        times[in_bin], statuses[in_bin], float(horizon)
                    )
                    error = (
                        float(observed - predicted_mean)
                        if np.isfinite(observed)
                        else np.nan
                    )
                    row = {
                        "Model": model_name,
                        "model": model_name,
                        "endpoint_type": "condition",
                        "risk_type": "death_adjusted",
                        "condition_id": int(condition_id),
                        "condition_name": str(condition_name),
                        "horizon_months": int(horizon),
                        "calibration_stratum": str(stratum_name),
                        "sex": str(stratum_name),
                        "calibration_bin": int(calibration_bin),
                        "n": int(bin_n),
                        "eligible_n": int(bin_n),
                        "known_n": int(known[in_bin].sum()),
                        "event_n": int(cases[in_bin].sum()),
                        "death_competing_n": int(competing[in_bin].sum()),
                        "predicted_mean": predicted_mean,
                        "predicted_min": float(scores[in_bin].min()),
                        "predicted_max": float(scores[in_bin].max()),
                        "observed_aj_cif": observed,
                        "observed_minus_predicted": error,
                        "absolute_error": (
                            float(abs(error)) if np.isfinite(error) else np.nan
                        ),
                    }
                    decile_rows.append(row)
                    condition_rows.append(row)

                finite_rows = [
                    row
                    for row in condition_rows
                    if np.isfinite(row["absolute_error"])
                ]
                absolute_errors = np.asarray(
                    [row["absolute_error"] for row in finite_rows], dtype=float
                )
                bin_weights = np.asarray(
                    [row["n"] for row in finite_rows], dtype=float
                )
                if absolute_errors.size:
                    ici = float(
                        np.average(absolute_errors, weights=bin_weights)
                    )
                    e50 = weighted_quantile_compatibility(
                        absolute_errors, bin_weights, 0.50
                    )
                    e90 = weighted_quantile_compatibility(
                        absolute_errors, bin_weights, 0.90
                    )
                    ici_unweighted = float(absolute_errors.mean())
                    e50_unweighted = float(np.quantile(absolute_errors, 0.50))
                    e90_unweighted = float(np.quantile(absolute_errors, 0.90))
                else:
                    ici = e50 = e90 = np.nan
                    ici_unweighted = e50_unweighted = e90_unweighted = np.nan
                summary_rows.append({
                    "Model": model_name,
                    "model": model_name,
                    "endpoint_type": "condition",
                    "risk_type": "death_adjusted",
                    "condition_id": int(condition_id),
                    "condition_name": str(condition_name),
                    "horizon_months": int(horizon),
                    "calibration_stratum": str(stratum_name),
                    "sex": str(stratum_name),
                    "ici": ici,
                    "e50": e50,
                    "e90": e90,
                    "ici_unweighted_bins": ici_unweighted,
                    "e50_unweighted_bins": e50_unweighted,
                    "e90_unweighted_bins": e90_unweighted,
                    "n_calibration_bins": int(len(finite_rows)),
                    "total_eligible_n": int(eligible_n),
                    "total_known_n": int(known.sum()),
                    "total_event_n": int(cases.sum()),
                    "total_death_competing_n": int(competing.sum()),
                    "mean_predicted_risk": float(scores.mean()),
                })

    decile_columns = [
        "Model", "model", "endpoint_type", "risk_type", "condition_id",
        "condition_name", "horizon_months", "calibration_stratum", "sex",
        "calibration_bin", "n", "eligible_n", "known_n", "event_n",
        "death_competing_n", "predicted_mean", "predicted_min",
        "predicted_max", "observed_aj_cif", "observed_minus_predicted",
        "absolute_error",
    ]
    summary_columns = [
        "Model", "model", "endpoint_type", "risk_type", "condition_id",
        "condition_name", "horizon_months", "calibration_stratum", "sex",
        "ici", "e50", "e90", "ici_unweighted_bins",
        "e50_unweighted_bins", "e90_unweighted_bins",
        "n_calibration_bins", "total_eligible_n", "total_known_n",
        "total_event_n", "total_death_competing_n", "mean_predicted_risk",
    ]
    deciles = pd.DataFrame(decile_rows).reindex(columns=decile_columns)
    summary = pd.DataFrame(summary_rows).reindex(columns=summary_columns)
    if deciles.empty or summary.empty:
        raise RuntimeError(
            "No A10 sex-stratified calibration rows met the locked group/bin "
            "size requirements."
        )
    deciles = deciles.sort_values([
        "horizon_months", "condition_id", "calibration_stratum",
        "calibration_bin",
    ]).reset_index(drop=True)
    summary = summary.sort_values([
        "horizon_months", "condition_id", "calibration_stratum",
    ]).reset_index(drop=True)
    observed_horizons = sorted(
        pd.to_numeric(summary["horizon_months"], errors="coerce")
        .dropna().astype(int).unique().tolist()
    )
    if observed_horizons != sorted(map(int, CFG.report_horizons_months)):
        raise RuntimeError(
            "Sex-stratified calibration does not cover every A10 reporting "
            f"horizon: {observed_horizons}."
        )
    observed_strata = set(summary["calibration_stratum"].astype(str))
    if not {"All", "Male", "Female"}.issubset(observed_strata):
        raise RuntimeError(
            "Sex-stratified calibration lacks All, Male, or Female rows."
        )

    figure_dir = RUN_DIR / "figure2_revised_inputs_2yr"
    figure_dir.mkdir(parents=True, exist_ok=True)
    primary_horizon = 24
    deciles_2yr = deciles[deciles["horizon_months"].eq(primary_horizon)].copy()
    summary_2yr = summary[summary["horizon_months"].eq(primary_horizon)].copy()
    if deciles_2yr.empty or summary_2yr.empty:
        raise RuntimeError("Required 24-month Figure-2 calibration rows are empty.")

    output_paths = {
        "calibration_deciles_2yr": (
            figure_dir / "figure2_revised_panelH_calibration_deciles_2yr.csv"
        ),
        "calibration_summary_2yr": (
            figure_dir / "figure2_revised_panelH_calibration_summary_2yr.csv"
        ),
        "calibration_deciles_all_horizons": (
            figure_dir
            / "figure2_revised_panelH_calibration_deciles_all_horizons.csv"
        ),
        "calibration_summary_all_horizons_root": (
            RUN_DIR
            / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
        ),
        "calibration_summary_all_horizons_figure2": (
            figure_dir
            / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
        ),
        "calibration_summary_alias_root": (
            RUN_DIR / "test_calibration_error_by_horizon_condition_sex.csv"
        ),
        "calibration_summary_alias_figure2": (
            figure_dir / "test_calibration_error_by_horizon_condition_sex.csv"
        ),
    }
    deciles_2yr.to_csv(output_paths["calibration_deciles_2yr"], index=False)
    summary_2yr.to_csv(output_paths["calibration_summary_2yr"], index=False)
    deciles.to_csv(
        output_paths["calibration_deciles_all_horizons"], index=False
    )
    summary.to_csv(
        output_paths["calibration_summary_all_horizons_root"], index=False
    )
    summary.to_csv(
        output_paths["calibration_summary_all_horizons_figure2"], index=False
    )
    summary.to_csv(output_paths["calibration_summary_alias_root"], index=False)
    summary.to_csv(
        output_paths["calibration_summary_alias_figure2"], index=False
    )
    metadata_path = RUN_DIR / "figure2_calibration_export_metadata.json"
    atomic_json_dump({
        "model": model_name,
        "risk_type": "death_adjusted",
        "run_dir": str(RUN_DIR),
        "n_test_participants": int(len(test_frame)),
        "n_conditions": int(K),
        "horizons_months": list(map(int, CFG.report_horizons_months)),
        "primary_horizon_months": int(primary_horizon),
        "gap_months": int(CFG.gap_months),
        "n_calibration_bins_requested": int(CFG.calibration_bins),
        "minimum_calibration_group_n": int(CFG.min_calibration_group_n),
        "minimum_calibration_bin_n": int(CFG.min_calibration_bin_n),
        "sex_source": "gender_idx_mapping_0_female_1_male_2_unknown",
        "gender_index_to_sex": {
            "0": "Female", "1": "Male", "2": "Unknown",
        },
        "sex_counts": {
            str(key): int(value)
            for key, value in pd.Series(sex).value_counts(dropna=False).items()
        },
        "observed_calibration": (
            "Aalen-Johansen condition CIF with death competing, using the "
            "canonical A10 lead-time outcome contract"
        ),
        "calibration_error_definition": (
            "ICI/E50/E90 are participant-count-weighted absolute errors "
            "across retained quantile bins within condition, horizon, and stratum"
        ),
        "outputs": {
            key: str(value) for key, value in output_paths.items()
        },
        "sha256": {
            key: sha256_file(value) for key, value in output_paths.items()
        },
        "figure2_inputs_ready": True,
    }, metadata_path)
    return deciles, summary


def artifact_hash_table() -> pd.DataFrame:
    rows = []
    for path in sorted(RUN_DIR.iterdir()):
        if path.is_file() and path.name != "artifact_sha256.csv":
            rows.append({
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return pd.DataFrame(rows)


def save_evaluation_outputs(
    result: Mapping[str, pd.DataFrame],
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
    evaluation_mode: str,
) -> None:
    compatibility = build_main_compatibility_tables(
        result, test_frame, predictions
    )

    # Match the A10 publication guardrail explicitly: both calibration output
    # families must contain the five-year (60-month) result before anything is
    # persisted as a completed comparison.
    for calibration_name, calibration_frame in (
        ("test_aj_calibration_by_horizon", compatibility["calibration"]),
        ("test_calibration_error_summary", compatibility["calibration_error"]),
    ):
        if "horizon_months" not in calibration_frame.columns:
            raise RuntimeError(
                f"{calibration_name} is missing its horizon_months column."
            )
        calibration_horizons = set(
            pd.to_numeric(
                calibration_frame["horizon_months"], errors="coerce"
            ).dropna().astype(int)
        )
        if 60 not in calibration_horizons:
            raise RuntimeError(
                f"{calibration_name} does not contain the required 60-month calibration."
            )

    # Exact main-model public filenames. Delphi-specific extensions are kept
    # in explicitly named sidecars so downstream loaders need no special case.
    compatibility["micro"].to_csv(
        RUN_DIR / "test_micro_metrics_by_horizon.csv", index=False
    )
    compatibility["per_condition"].to_csv(
        RUN_DIR / "test_per_condition_metrics_by_horizon.csv", index=False
    )
    integrated_compatibility_columns = [
        "model_name", "risk_type", "condition_id", "condition_name",
        "integration_start_months", "integration_end_months",
        "grid_step_months", "n_grid_points", "ibs_ipcw_ht",
        "ibs_null_ipcw_ht", "scaled_ibs_ipcw_ht", "inbll_ipcw_ht",
        "iauc_ipcw_cr", "integrated_abs_calibration_error",
    ]
    primary_integrated = result["integrated"].query(
        "risk_type == 'death_adjusted'"
    )[integrated_compatibility_columns].copy()
    primary_integrated.to_csv(
        RUN_DIR / "test_integrated_metrics_by_condition.csv", index=False
    )
    result["integrated"].to_csv(
        RUN_DIR / "test_integrated_metrics_all_risk_types_delphi.csv", index=False
    )
    dense_compatibility_columns = [
        "model_name", "risk_type", "condition_id", "condition_name",
        "horizon_months", "eligible_n", "known_n", "event_n",
        "death_competing_n", "mean_predicted_cif", "observed_aj_cif",
        "absolute_calibration_error", "auc_ipcw_cr", "auprc_ipcw_cr",
        "brier_ipcw_ht", "brier_null_ipcw_ht", "nbll_ipcw_ht",
    ]
    primary_dense = result["dense"].query(
        "risk_type == 'death_adjusted'"
    )[dense_compatibility_columns].copy()
    primary_dense.to_csv(
        RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv", index=False
    )
    result["dense"].to_csv(
        RUN_DIR / "test_dense_metrics_all_risk_types_delphi.csv", index=False
    )
    compatibility["calibration"].to_csv(
        RUN_DIR / "test_aj_calibration_by_horizon.csv", index=False
    )
    compatibility["calibration_error"].to_csv(
        RUN_DIR / "test_calibration_error_summary.csv", index=False
    )
    compatibility["death"].to_csv(
        RUN_DIR / "test_death_head_metrics_by_horizon.csv", index=False
    )
    compatibility["summary"].to_csv(
        RUN_DIR / "test_summary_publication.csv", index=False
    )
    compatibility["micro"].to_csv(
        RUN_DIR / "within_test_withleadtime_death_adjusted.csv", index=False
    )
    legacy_condition_columns = [
        "horizon_months", "risk_type", "condition_id", "condition_name",
        "eligible_n", "known_n", "event_n", "death_competing_n",
        "admin_censored_n", "mean_predicted_risk", "auroc", "auprc",
        "event_rate", "brier_ipcw", "nbll_ipcw", "weight_sum",
        "normalization_n", "ctd_ipcw", "ctd_status",
    ]
    compatibility["per_condition"][legacy_condition_columns].to_csv(
        RUN_DIR / "condition_test_withleadtime_death_adjusted.csv", index=False
    )

    # Extended Delphi audit tables.
    result["per_condition"].to_csv(
        RUN_DIR / "test_per_condition_metrics_delphi_extended.csv", index=False
    )
    result["calibration"].to_csv(
        RUN_DIR / "test_aj_calibration_delphi_extended.csv", index=False
    )
    result["smooth_calibration"].to_csv(
        RUN_DIR / "test_smooth_calibration_curves.csv", index=False
    )
    result["macro"].to_csv(
        RUN_DIR / "test_macro_summary_by_horizon.csv", index=False
    )
    result["death"].to_csv(
        RUN_DIR / "test_death_metrics_by_horizon.csv", index=False
    )
    result["death_integrated"].to_csv(
        RUN_DIR / "test_death_integrated_metrics.csv", index=False
    )

    save_compatibility_prediction_npz(test_frame, predictions)
    export_figure2_calibration_tables(test_frame, predictions)
    np.savez_compressed(
        RUN_DIR / "test_dense_prediction_arrays_for_integrated_metrics.npz",
        eid=test_frame["eid"].astype(str).to_numpy(dtype=object),
        condition_names=np.asarray(CONDITION_NAMES, dtype=object),
        dense_horizons_months=np.asarray(DENSE_HORIZONS_MONTHS, dtype=np.int32),
        entry_delay_months=np.asarray(CFG.gap_months, dtype=np.float32),
        gap_months=np.asarray(CFG.gap_months, dtype=np.float32),
        risk_death_adjusted=np.asarray(predictions["adjusted"], dtype=np.float32),
    )
    with PREDICTION_PATHS["manifest"].open(encoding="utf-8") as handle:
        prediction_manifest_for_metadata = json.load(handle)
    atomic_json_dump({
        "schema_version": (
            "ehr-competing-risk-canonical-integrated-evaluation.v1"
        ),
        "model_name": "Delphi2M_Canonical_AR_RateGrid",
        "risk_type": "death_adjusted",
        "checkpoint": str(
            prediction_manifest_for_metadata.get("checkpoint_path", "")
        ),
        "gap_months": int(CFG.gap_months),
        "dense_horizons_months": list(DENSE_HORIZONS_MONTHS),
        "integration_method": "trapezoidal mean over the canonical six-month grid",
        "censoring_distribution": (
            "condition-specific administrative-censoring Kaplan-Meier "
            "estimated from canonical train_samples"
        ),
        "censor_survival_floor": float(CFG.censor_survival_floor),
        "estimand": (
            f"Condition incidence during ({CFG.gap_months} months, "
            f"{CFG.gap_months} months + horizon] "
            "before death, conditional on being alive and condition-free "
            f"through the {CFG.gap_months}-month lead-time gap."
        ),
        "n_test_participants": int(len(test_frame)),
        "n_conditions": int(K),
        "output_files": {
            "integrated_metrics": str(
                RUN_DIR / "test_integrated_metrics_by_condition.csv"
            ),
            "dense_point_metrics": str(
                RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv"
            ),
            "standard_point_metrics": str(
                RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
            ),
            "dense_predictions": str(
                RUN_DIR / "test_dense_prediction_arrays_for_integrated_metrics.npz"
            ),
            "information_diagnostics_by_condition": str(
                PREDICTION_PATHS["diagnostics"]
            ),
            "information_diagnostics_summary": str(
                PREDICTION_PATHS["summary"]
            ),
        },
    }, RUN_DIR / "test_integrated_metrics_metadata.json")

    # These no-CI views are valid immediately and are overwritten by the
    # participant bootstrap later in the complete workflow.
    summary_with_ci = compatibility["summary"].copy()
    summary_with_ci["ci_computed"] = False
    summary_with_ci["ci_lower"] = np.nan
    summary_with_ci["ci_upper"] = np.nan
    summary_with_ci.to_csv(
        RUN_DIR / "test_summary_publication_with_ci.csv", index=False
    )
    death_with_ci = compatibility["death"].copy()
    death_with_ci["ci_computed"] = False
    death_with_ci["ci_lower"] = np.nan
    death_with_ci["ci_upper"] = np.nan
    death_with_ci.to_csv(
        RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv", index=False
    )
    pd.DataFrame().to_csv(
        RUN_DIR / "test_bootstrap_replicates_summary.csv", index=False
    )
    pd.DataFrame().to_csv(
        RUN_DIR / "test_bootstrap_summary_cis.csv", index=False
    )

    primary_macro = compatibility["summary"][
        compatibility["summary"]["risk_type"].eq("death_adjusted")
        & pd.to_numeric(
            compatibility["summary"]["horizon_months"], errors="coerce"
        ).notna()
    ].copy()

    metric_definitions = {
        "schema_version": "ehr-competing-risk-canonical-evaluation.v3.0",
        "model": "Delphi2M_Canonical_AR_RateGrid",
        "estimand": (
            f"Incident condition during ({CFG.gap_months} months, "
            f"{CFG.gap_months} months + horizon] after the "
            "locked main-model index, before death, conditional on being alive and "
            f"condition-free through the {CFG.gap_months}-month lead-time gap."
        ),
        "canonical_index_contract": str(CANONICAL_PATHS["target_config"]),
        "time_origin": CANONICAL_INDEX_DESIGN,
        "gap_months": CFG.gap_months,
        "report_horizons_months": list(CFG.report_horizons_months),
        "dense_integration_horizons_months": list(DENSE_HORIZONS_MONTHS),
        "censoring_distribution": (
            "Condition-specific administrative-censoring Kaplan-Meier estimated from the "
            "canonical training split and applied unchanged to test metrics."
        ),
        "auc_ipcw_cr": (
            "IPCW cumulative/dynamic competing-risk AUC; cases have condition by t, controls "
            "are condition-free through t or have competing death."
        ),
        "cindex_ipcw_cr": (
            "Truncated competing-risk concordance through tau with pairwise inverse-censoring weights."
        ),
        "brier_ipcw_ht": (
            "Horvitz-Thompson IPCW Brier score divided by the number eligible at the lead-time landmark."
        ),
        "nbll_ipcw_ht": "Horvitz-Thompson IPCW Bernoulli negative log-likelihood.",
        "observed_aj_cif": "Aalen-Johansen observed cumulative incidence with death competing.",
        "calibration_in_large_pred_minus_obs": "Mean predicted CIF minus observed AJ CIF; ideal is zero.",
        "observed_to_expected_ratio": "Observed AJ CIF divided by mean predicted CIF; ideal is one.",
        "calibration_intercept_slope_cloglog": (
            "IPCW complementary-log-log calibration intercept and slope; ideals are zero and one."
        ),
        "ibs_ipcw_ht": (
            "Trapezoidal mean IPCW Brier score over the 12-to-120-month dense six-month grid."
        ),
        "scaled_ibs_ipcw_ht": (
            "1 - IBS(model)/IBS(null), where the null is the canonical-training Aalen-Johansen CIF."
        ),
        "inbll_ipcw_ht": "Integrated IPCW negative log-likelihood over 12 to 120 months.",
        "iauc_ipcw_cr": "Integrated IPCW competing-risk AUC over available dense-grid horizons.",
        "confidence_intervals": (
            "Participant-level percentile bootstrap with the fitted model, "
            "predictions, condition axis, and train-derived censoring "
            "distributions frozen. The exact competing-risk C-index CI is "
            "reported at 60 months after the lead-time gap."
        ),
    }
    atomic_json_dump(metric_definitions, RUN_DIR / "test_metric_definitions.json")

    evaluation_manifest = {
        "schema_version": CFG.schema_version,
        "evaluation_mode": evaluation_mode,
        "model": "Delphi2M_Canonical_AR_RateGrid",
        "main_model_run_dir": str(MAIN_RUN_DIR),
        "main_source_contract": MAIN_SOURCE_CONTRACT,
        "test_rows": len(test_frame),
        "test_eid_axis_sha256": canonical_json_hash(test_frame["eid"].astype(str).tolist()),
        "canonical_test_samples_sha256": sha256_file(CANONICAL_PATHS["test"]),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "preprocessing_contract_sha256": PREPROCESSING_CONTRACT_HASH,
        "prediction_manifest_sha256": sha256_file(PREDICTION_PATHS["manifest"]),
        "same_index_as_main_model": True,
        "same_split_as_main_model": True,
        "same_condition_axis_as_main_model": True,
        "same_targets_as_main_model": True,
        "test_used_for_model_selection": False,
        "report_horizons_months": list(CFG.report_horizons_months),
        "gap_months": CFG.gap_months,
        "primary_metric": "macro mean integrated IPCW Brier score",
        "primary_prediction": "death-adjusted condition CIF",
    }
    atomic_json_dump(evaluation_manifest, RUN_DIR / "test_evaluation_manifest.json")

    # Main Cell-7 metadata filename and key schema, populated for Delphi. The
    # no-CI compatibility tables remain unchanged by the additive Cell-7B-style
    # bootstrap outputs, matching the main notebook's behavior.
    cell7_compatibility_metadata = {
        "model_variant": "Delphi2M_Canonical_AR_RateGrid",
        "checkpoint_used": str(
            prediction_manifest_for_metadata.get("checkpoint_path", "")
        ),
        "n_test_rows": int(len(test_frame)),
        "K": int(K),
        "condition_names_n": int(len(CONDITION_NAMES)),
        "gap_months": float(CFG.gap_months),
        "horizons_months": list(map(int, CFG.report_horizons_months)),
        "death_tie_policy": str(
            TARGET_CONFIG.get(
                "DEATH_TIE_POLICY",
                TARGET_METADATA.get("death_adjusted_design", {}).get(
                    "death_tie_policy", ""
                ),
            )
        ),
        "bootstrap": {
            "enabled": False,
            "n": 0,
            "alpha": float(CFG.ci_alpha),
            "cluster": "eid",
            "include_ctd": False,
            "ctd_status": "disabled_not_validly_implemented",
            "censoring_km_source": "canonical_train_samples",
            "seed": int(CFG.metric_seed),
            "ctd_max_cases_point": int(CFG.cindex_max_cases),
            "ctd_max_comparators_point": int(CFG.cindex_max_comparators),
            "ctd_max_cases_bootstrap": 0,
            "ctd_max_comparators_bootstrap": 0,
        },
        "output_files": {
            "pred_arrays": str(
                RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
            ),
            "micro": str(RUN_DIR / "test_micro_metrics_by_horizon.csv"),
            "per_condition": str(
                RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
            ),
            "death": str(
                RUN_DIR / "test_death_head_metrics_by_horizon.csv"
            ),
            "death_with_ci": str(
                RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv"
            ),
            "calibration": str(
                RUN_DIR / "test_aj_calibration_by_horizon.csv"
            ),
            "calibration_error": str(
                RUN_DIR / "test_calibration_error_summary.csv"
            ),
            "summary": str(RUN_DIR / "test_summary_publication.csv"),
            "summary_with_ci": str(
                RUN_DIR / "test_summary_publication_with_ci.csv"
            ),
            "bootstrap_replicates": str(
                RUN_DIR / "test_bootstrap_replicates_summary.csv"
            ),
            "bootstrap_cis": str(
                RUN_DIR / "test_bootstrap_summary_cis.csv"
            ),
        },
        "cell8_compatible_schema": True,
    }
    atomic_json_dump(
        cell7_compatibility_metadata,
        RUN_DIR / "cell7_test_evaluation_metadata.json",
    )
    artifact_hash_table().to_csv(RUN_DIR / "artifact_sha256.csv", index=False)

    print("\nPrimary death-adjusted macro results:")
    print(primary_macro.to_string(index=False))
    print("\nPrimary integrated condition results:")
    print(primary_integrated[[
        "condition_id", "condition_name", "ibs_ipcw_ht", "scaled_ibs_ipcw_ht",
        "inbll_ipcw_ht", "iauc_ipcw_cr", "integrated_abs_calibration_error",
    ]].head(20).to_string(index=False))

# -----------------------------------------------------------------------------
# Required test-performance artifacts and post-write verification
# -----------------------------------------------------------------------------
TEST_PERFORMANCE_FILES = {
    "micro": RUN_DIR / "test_micro_metrics_by_horizon.csv",
    "per_condition": RUN_DIR / "test_per_condition_metrics_by_horizon.csv",
    "integrated": RUN_DIR / "test_integrated_metrics_by_condition.csv",
    "dense_metrics": RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv",
    "dense_predictions": RUN_DIR / "test_dense_prediction_arrays_for_integrated_metrics.npz",
    "integrated_metadata": RUN_DIR / "test_integrated_metrics_metadata.json",
    "calibration": RUN_DIR / "test_aj_calibration_by_horizon.csv",
    "calibration_error": RUN_DIR / "test_calibration_error_summary.csv",
    "calibration_error_by_sex": (
        RUN_DIR / "test_calibration_error_by_horizon_condition_sex.csv"
    ),
    "figure2_calibration_summary": (
        RUN_DIR / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
    ),
    "figure2_calibration_metadata": (
        RUN_DIR / "figure2_calibration_export_metadata.json"
    ),
    "smooth_calibration": RUN_DIR / "test_smooth_calibration_curves.csv",
    "macro": RUN_DIR / "test_macro_summary_by_horizon.csv",
    "death": RUN_DIR / "test_death_head_metrics_by_horizon.csv",
    "death_integrated": RUN_DIR / "test_death_integrated_metrics.csv",
    "publication": RUN_DIR / "test_summary_publication.csv",
    "publication_with_ci": RUN_DIR / "test_summary_publication_with_ci.csv",
    "death_with_ci": RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv",
    "bootstrap_replicates": RUN_DIR / "test_bootstrap_replicates_summary.csv",
    "bootstrap_summary": RUN_DIR / "test_bootstrap_summary_cis.csv",
    "prediction_npz": RUN_DIR / "test_prediction_arrays_death_adjusted.npz",
    "prediction_manifest": PREDICTION_PATHS["manifest"],
    "prediction_row_key": PREDICTION_PATHS["row_key"],
    "rate_grid_diagnostics": PREDICTION_PATHS["diagnostics"],
    "rate_grid_summary": PREDICTION_PATHS["summary"],
    "evaluation_manifest": RUN_DIR / "test_evaluation_manifest.json",
    "metric_definitions": RUN_DIR / "test_metric_definitions.json",
    "cell7_metadata": RUN_DIR / "cell7_test_evaluation_metadata.json",
}

# The source A10 test metrics are zero-lead. Their outcome counts cannot be
# compared with six-month-lead outcome counts. The canonical test sample hash,
# participant order, output schemas, and prediction manifest remain verified.


def _check_main_model_count_parity(saved_per_condition: pd.DataFrame) -> Dict[str, Any]:
    """Verify exact outcome-count parity when main run shares the 6-month delayed-entry contract."""
    main_metrics_path = CANONICAL_PATHS.get("main_per_condition_metrics")
    if CANONICAL_ENTRY_DELAY_MONTHS != CFG.gap_months:
        return {
            "passed": False,
            "status": "not_applicable_different_entry_delay",
            "main_entry_delay_months": int(CANONICAL_ENTRY_DELAY_MONTHS),
            "comparator_gap_months": int(CFG.gap_months),
        }
    if main_metrics_path is None or not Path(main_metrics_path).is_file():
        return {
            "passed": False,
            "status": "main_metrics_file_missing",
            "path": str(main_metrics_path),
        }
    main_df = pd.read_csv(main_metrics_path)
    keys = ["horizon_months", "risk_type", "condition_id", "condition_name"]
    count_cols = ["eligible_n", "known_n", "event_n", "death_competing_n"]
    if not set(keys + count_cols).issubset(main_df.columns):
        return {
            "passed": False,
            "status": "main_metrics_missing_count_columns",
        }
    left = saved_per_condition.loc[
        saved_per_condition["risk_type"].astype(str).eq("death_adjusted"),
        keys + count_cols,
    ].copy()
    right = main_df.loc[
        main_df["risk_type"].astype(str).eq("death_adjusted"),
        keys + count_cols,
    ].copy()
    merged = left.merge(
        right, on=keys, how="outer", suffixes=("_delphi", "_crescent"),
        indicator=True, validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(
            "Count-parity failure: Delphi and CRESCENT death-adjusted test tables "
            "have different (horizon, condition) keys."
        )
    mismatches = {}
    for col in count_cols:
        diff = int((
            pd.to_numeric(merged[f"{col}_delphi"])
            != pd.to_numeric(merged[f"{col}_crescent"])
        ).sum())
        if diff:
            mismatches[col] = diff
    if mismatches:
        raise RuntimeError(
            f"Count-parity failure against CRESCENT_Final: {mismatches}"
        )
    return {
        "passed": True,
        "status": "exact_match",
        "rows_verified": int(len(merged)),
        "main_metrics_path": str(main_metrics_path),
    }


def verify_test_performance_outputs(
    result: Mapping[str, pd.DataFrame],
    test_frame: pd.DataFrame,
    evaluation_mode: str,
) -> Path:
    """Fail loudly unless all expected test-performance outputs were saved correctly."""
    required_result_keys = {
        "per_condition", "integrated", "calibration", "smooth_calibration",
        "macro", "death", "death_integrated", "dense",
    }
    missing_result_keys = sorted(required_result_keys.difference(result))
    if missing_result_keys:
        raise RuntimeError(f"Evaluation result lacks tables: {missing_result_keys}")

    if result["per_condition"].empty:
        raise RuntimeError("Per-condition test performance table is empty.")
    if result["macro"].empty:
        raise RuntimeError("Macro test performance table is empty.")
    if result["integrated"].empty:
        raise RuntimeError("Integrated test performance table is empty.")
    if result["death"].empty:
        raise RuntimeError("Death test performance table is empty.")

    required_per_condition_columns = {
        "horizon_months", "risk_type", "condition_id", "condition_name",
        "eligible_n", "event_n", "death_competing_n", "auc_ipcw_cr",
        "cindex_ipcw_cr", "brier_ipcw_ht", "nbll_ipcw_ht",
        "mean_predicted_cif", "observed_aj_cif",
        "calibration_in_large_pred_minus_obs", "observed_to_expected_ratio",
        "calibration_intercept_cloglog", "calibration_slope_cloglog",
    }
    missing_columns = sorted(
        required_per_condition_columns.difference(result["per_condition"].columns)
    )
    if missing_columns:
        raise RuntimeError(
            "Per-condition test performance table lacks required columns: "
            + ", ".join(missing_columns)
        )

    primary = result["per_condition"].query("risk_type == 'death_adjusted'")
    expected_primary_rows = K * len(CFG.report_horizons_months)
    if len(primary) != expected_primary_rows:
        raise RuntimeError(
            f"Expected {expected_primary_rows} death-adjusted condition×horizon rows, "
            f"but generated {len(primary)}."
        )
    if set(primary["horizon_months"].astype(int)) != set(CFG.report_horizons_months):
        raise RuntimeError("Saved test performance does not cover all requested horizons.")
    if primary["condition_id"].nunique() != K:
        raise RuntimeError("Saved test performance does not cover every condition.")

    saved_integrated = pd.read_csv(
        RUN_DIR / "test_integrated_metrics_by_condition.csv"
    )
    if len(saved_integrated) != K or not saved_integrated[
        "risk_type"
    ].astype(str).eq("death_adjusted").all():
        raise RuntimeError(
            "Integrated compatibility table must contain exactly one "
            "death-adjusted row per canonical condition."
        )
    saved_dense = pd.read_csv(
        RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv"
    )
    if len(saved_dense) != K * len(DENSE_HORIZONS_MONTHS):
        raise RuntimeError("Dense condition×horizon metric table has the wrong size.")
    if not saved_dense["risk_type"].astype(str).eq("death_adjusted").all():
        raise RuntimeError(
            "Dense compatibility table must contain death-adjusted rows only."
        )
    expected_integrated_columns = [
        "model_name", "risk_type", "condition_id", "condition_name",
        "integration_start_months", "integration_end_months",
        "grid_step_months", "n_grid_points", "ibs_ipcw_ht",
        "ibs_null_ipcw_ht", "scaled_ibs_ipcw_ht", "inbll_ipcw_ht",
        "iauc_ipcw_cr", "integrated_abs_calibration_error",
    ]
    expected_dense_columns = [
        "model_name", "risk_type", "condition_id", "condition_name",
        "horizon_months", "eligible_n", "known_n", "event_n",
        "death_competing_n", "mean_predicted_cif", "observed_aj_cif",
        "absolute_calibration_error", "auc_ipcw_cr", "auprc_ipcw_cr",
        "brier_ipcw_ht", "brier_null_ipcw_ht", "nbll_ipcw_ht",
    ]
    if saved_integrated.columns.tolist() != expected_integrated_columns:
        raise RuntimeError(
            "Integrated CSV columns/order differ from the locked main schema."
        )
    if saved_dense.columns.tolist() != expected_dense_columns:
        raise RuntimeError(
            "Dense CSV columns/order differ from the locked main schema."
        )
    with np.load(
        RUN_DIR / "test_prediction_arrays_death_adjusted.npz",
        allow_pickle=True,
    ) as compatibility_npz:
        expected_npz_keys = {
            "eid", "condition_names", "horizons_months", "entry_delay_months",
            "gap_months", "index_age_years",
            "at_risk", "event_obs", "event_bin", "t_tilde", "C_months",
            "C_bin", "death_observed_post_index", "death_time_months",
            "death_bin",
        }
        for horizon in CFG.report_horizons_months:
            expected_npz_keys.update({
                f"risk_death_adjusted_h{int(horizon)}",
                f"risk_unadjusted_h{int(horizon)}",
                f"risk_death_h{int(horizon)}",
            })
        if set(compatibility_npz.files) != expected_npz_keys:
            raise RuntimeError(
                "Compatibility prediction NPZ keys differ from the locked "
                f"main-model schema: {sorted(compatibility_npz.files)}."
            )

    missing_files = [
        str(path) for path in TEST_PERFORMANCE_FILES.values()
        if not Path(path).is_file()
    ]
    if missing_files:
        raise RuntimeError(
            "Required test-performance files were not written:\n" + "\n".join(missing_files)
        )
    empty_files = [
        str(path) for path in TEST_PERFORMANCE_FILES.values()
        if Path(path).stat().st_size == 0
    ]
    if empty_files:
        raise RuntimeError(
            "Required test-performance files are empty:\n" + "\n".join(empty_files)
        )

    # Re-read the two principal CSVs from disk: verification must test persistence,
    # not merely the in-memory DataFrames.
    saved_per_condition = pd.read_csv(TEST_PERFORMANCE_FILES["per_condition"])
    saved_macro = pd.read_csv(TEST_PERFORMANCE_FILES["macro"])
    # Cross-model count parity requires a separately evaluated six-month-lead
    # main-model result set. The zero-lead source table is not comparable.
    if len(saved_per_condition) != len(result["per_condition"]):
        raise RuntimeError("Saved per-condition row count differs from the evaluated table.")
    if len(saved_macro) != len(result["macro"]):
        raise RuntimeError("Saved macro row count differs from the evaluated table.")
    schema_requirements = {
        "micro": {
            "horizon_months", "risk_type", "auroc", "auprc",
            "event_rate", "brier_ipcw", "nbll_ipcw", "weight_sum",
            "normalization_n", "eligible_pairs", "known_pairs",
            "condition_events", "death_competing_events",
        },
        "per_condition": {
            "horizon_months", "risk_type", "condition_id", "condition_name",
            "eligible_n", "known_n", "event_n", "death_competing_n",
            "admin_censored_n", "mean_predicted_risk", "event_rate",
            "auroc", "auprc", "ctd_ipcw", "brier_ipcw", "nbll_ipcw",
            "auc_ipcw_cr", "auprc_ipcw_cr", "brier_ipcw_ht",
            "brier_null_ipcw_ht", "nbll_ipcw_ht",
        },
        "death": {
            "horizon_months", "eligible_n", "known_n", "death_event_n",
            "mean_predicted_death_risk", "auroc", "auprc", "ctd_ipcw",
            "brier_ipcw", "nbll_ipcw",
        },
        "calibration": {
            "horizon_months", "risk_type", "condition_id", "condition_name",
            "calibration_bin", "n", "predicted_mean", "predicted_min",
            "predicted_max", "observed_aj_cif", "observed_minus_predicted",
        },
        "summary": {
            "horizon_months", "risk_type", "micro_auroc", "micro_auprc",
            "macro_auroc", "macro_auprc", "macro_ctd_ipcw",
            "macro_brier_ipcw", "macro_nbll_ipcw", "mean_predicted_risk",
            "n_conditions_with_auroc", "n_conditions",
        },
        "calibration_error": {
            "horizon_months", "risk_type", "ici", "e50", "e90",
            "n_calibration_bins", "calibration_weight_n",
        },
    }
    schema_frames = {
        "micro": pd.read_csv(TEST_PERFORMANCE_FILES["micro"]),
        "per_condition": saved_per_condition,
        "death": pd.read_csv(TEST_PERFORMANCE_FILES["death"]),
        "calibration": pd.read_csv(TEST_PERFORMANCE_FILES["calibration"]),
        "summary": pd.read_csv(TEST_PERFORMANCE_FILES["publication"]),
        "calibration_error": pd.read_csv(
            TEST_PERFORMANCE_FILES["calibration_error"]
        ),
    }
    for name, required_columns in schema_requirements.items():
        missing_schema_columns = sorted(
            required_columns.difference(schema_frames[name].columns)
        )
        if missing_schema_columns:
            raise RuntimeError(
                f"{name} compatibility schema is missing columns: "
                f"{missing_schema_columns}."
            )
    exact_main_columns = {
        "micro": [
            "horizon_months", "risk_type", "auroc", "auprc", "event_rate",
            "brier_ipcw", "nbll_ipcw", "weight_sum", "normalization_n",
            "eligible_pairs", "known_pairs", "condition_events",
            "death_competing_events",
        ],
        "per_condition": [
            "horizon_months", "risk_type", "condition_id", "condition_name",
            "eligible_n", "known_n", "event_n", "death_competing_n",
            "admin_censored_n", "mean_predicted_risk", "auroc", "auprc",
            "event_rate", "brier_ipcw", "nbll_ipcw", "weight_sum",
            "normalization_n", "ctd_ipcw", "ctd_status", "auc_ipcw_cr",
            "auprc_ipcw_cr", "brier_ipcw_ht", "brier_null_ipcw_ht",
            "nbll_ipcw_ht",
        ],
        "death": [
            "horizon_months", "eligible_n", "known_n", "death_event_n",
            "mean_predicted_death_risk", "auroc", "auprc", "event_rate",
            "brier_ipcw", "nbll_ipcw", "weight_sum", "normalization_n",
            "ctd_ipcw", "ctd_status",
        ],
        "calibration": [
            "horizon_months", "risk_type", "condition_id", "condition_name",
            "calibration_bin", "n", "predicted_mean", "predicted_min",
            "predicted_max", "observed_aj_cif", "observed_minus_predicted",
        ],
        "calibration_error": [
            "horizon_months", "risk_type", "ici", "e50", "e90",
            "n_calibration_bins", "calibration_weight_n",
        ],
        "summary": [
            "horizon_months", "risk_type", "micro_auroc", "micro_auprc",
            "micro_brier_ipcw", "micro_nbll_ipcw", "macro_auroc",
            "macro_auprc", "macro_brier_ipcw", "macro_nbll_ipcw",
            "macro_ctd_ipcw", "ctd_status", "mean_predicted_risk",
            "n_conditions_with_auroc", "n_conditions", "ici", "e50", "e90",
        ],
    }
    for name, expected_columns in exact_main_columns.items():
        if schema_frames[name].columns.tolist() != expected_columns:
            raise RuntimeError(
                f"{name} CSV columns/order differ from the locked main schema: "
                f"{schema_frames[name].columns.tolist()}"
            )
    legacy_condition_columns = exact_main_columns["per_condition"][:19]
    saved_legacy_condition = pd.read_csv(
        RUN_DIR / "condition_test_withleadtime_death_adjusted.csv"
    )
    if saved_legacy_condition.columns.tolist() != legacy_condition_columns:
        raise RuntimeError(
            "Legacy condition CSV columns/order differ from the locked main "
            "schema."
        )
    expected_publication_with_ci_columns = (
        exact_main_columns["summary"]
        + ["ci_computed", "ci_lower", "ci_upper"]
    )
    expected_death_with_ci_columns = (
        exact_main_columns["death"]
        + ["ci_computed", "ci_lower", "ci_upper"]
    )
    if pd.read_csv(
        TEST_PERFORMANCE_FILES["publication_with_ci"]
    ).columns.tolist() != expected_publication_with_ci_columns:
        raise RuntimeError(
            "Publication-with-CI placeholder schema differs from the locked "
            "main schema."
        )
    if pd.read_csv(
        TEST_PERFORMANCE_FILES["death_with_ci"]
    ).columns.tolist() != expected_death_with_ci_columns:
        raise RuntimeError(
            "Death-with-CI placeholder schema differs from the locked main "
            "schema."
        )
    for legacy_bootstrap_key in ("bootstrap_replicates", "bootstrap_summary"):
        if TEST_PERFORMANCE_FILES[legacy_bootstrap_key].read_text().strip():
            raise RuntimeError(
                "Legacy bootstrap placeholder differs from the locked main "
                f"schema: {TEST_PERFORMANCE_FILES[legacy_bootstrap_key]}"
            )
    with TEST_PERFORMANCE_FILES["integrated_metadata"].open(
        encoding="utf-8"
    ) as handle:
        saved_integrated_metadata = json.load(handle)
    expected_integrated_metadata_keys = {
        "schema_version", "model_name", "risk_type", "checkpoint",
        "gap_months", "dense_horizons_months", "integration_method",
        "censoring_distribution", "censor_survival_floor", "estimand",
        "n_test_participants", "n_conditions", "output_files",
    }
    if set(saved_integrated_metadata) != expected_integrated_metadata_keys:
        raise RuntimeError(
            "Integrated metadata keys differ from the locked main schema."
        )
    expected_integrated_output_keys = {
        "integrated_metrics", "dense_point_metrics",
        "standard_point_metrics", "dense_predictions",
        "information_diagnostics_by_condition",
        "information_diagnostics_summary",
    }
    if set(saved_integrated_metadata["output_files"]) != (
        expected_integrated_output_keys
    ):
        raise RuntimeError(
            "Integrated metadata output-file keys differ from the locked main "
            "schema."
        )
    if len(schema_frames["micro"]) != 2 * len(CFG.report_horizons_months):
        raise RuntimeError("Micro compatibility table has the wrong row count.")
    if len(schema_frames["death"]) != len(CFG.report_horizons_months):
        raise RuntimeError("Death compatibility table has the wrong row count.")
    if len(schema_frames["summary"]) != 2 * (
        len(CFG.report_horizons_months) + 1
    ):
        raise RuntimeError("Publication summary has the wrong row count.")
    if len(saved_per_condition) != 2 * K * len(CFG.report_horizons_months):
        raise RuntimeError("Per-condition compatibility table has the wrong row count.")

    completion_path = RUN_DIR / "TEST_EVALUATION_COMPLETE.json"
    completion_payload = {
        "schema_version": CFG.schema_version,
        "evaluation_mode": evaluation_mode,
        "completed_unix_time": time.time(),
        "output_directory": str(RUN_DIR),
        "test_rows": int(len(test_frame)),
        "conditions": int(K),
        "report_horizons_months": list(CFG.report_horizons_months),
        "death_adjusted_condition_horizon_rows": int(len(primary)),
        "test_eid_axis_sha256": canonical_json_hash(test_frame["eid"].astype(str).tolist()),
        "main_model_count_parity": _check_main_model_count_parity(
            saved_per_condition
        ),
        "files": {
            name: {
                "path": str(path),
                "size_bytes": int(Path(path).stat().st_size),
                "sha256": sha256_file(path),
            }
            for name, path in TEST_PERFORMANCE_FILES.items()
        },
    }
    atomic_json_dump(completion_payload, completion_path)
    return completion_path


def run_model_test_evaluation(checkpoint_path: Path, evaluation_mode: str) -> Dict[str, pd.DataFrame]:
    """Load canonical test data, infer from a frozen model, save and verify metrics."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    # Validate the checkpoint and persist the locked unit-scaling contract
    # (no post-hoc validation calibration) before opening the test pickle.
    model = load_frozen_model(checkpoint_path)
    fit_validation_rate_scaling(model, checkpoint_path)
    test_samples = load_canonical_test_samples()
    predictions = generate_predictions(model, test_samples, checkpoint_path)
    result = evaluate_all_metrics(test_samples, predictions)
    save_evaluation_outputs(result, test_samples, predictions, evaluation_mode)
    completion = verify_test_performance_outputs(result, test_samples, evaluation_mode)
    print(f"Verified test-performance completion: {completion}", flush=True)
    return result


def run_saved_prediction_metrics(evaluation_mode: str = "metrics_only") -> Dict[str, pd.DataFrame]:
    """Regenerate, save and verify metrics without rerunning neural-network inference."""
    test_samples = load_canonical_test_samples()
    predictions = load_saved_predictions(test_samples)
    result = evaluate_all_metrics(test_samples, predictions)
    save_evaluation_outputs(result, test_samples, predictions, evaluation_mode)
    completion = verify_test_performance_outputs(result, test_samples, evaluation_mode)
    print(f"Verified test-performance completion: {completion}", flush=True)
    return result


CI_MODEL_NAME = "Delphi2M_Canonical_AR_RateGrid"

# --- Notebook cell 31 ---
import gzip
import warnings

try:
    import sklearn
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    sklearn = None
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False


# Match the attached main-model CI scope exactly: fixed-horizon AUROC/AUPRC at
# 24 and 60 months, four integrated metrics, and the 60-month C-index.
CI_POINT_HORIZONS_MONTHS = (24, 60)
CI_POINT_METRICS = (
    "auc_ipcw_cr",
    "auprc_ipcw_cr",
)
CI_INTEGRATED_METRICS = (
    "ibs_ipcw_ht",
    "ibs_null_ipcw_ht",
    "scaled_ibs_ipcw_ht",
    "integrated_abs_calibration_error",
)


def _ci_group_keys(frame: pd.DataFrame) -> pd.DataFrame:
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


def _ci_metric_row(
    bootstrap_id: int,
    condition_id: int,
    metric: str,
    estimate: float,
    horizon_months: float = np.nan,
    integration_start_months: float = np.nan,
    integration_end_months: float = np.nan,
    eligible_n: int = 0,
    event_n: int = 0,
    comparable_pair_weight: float = np.nan,
) -> Dict[str, Any]:
    return {
        "bootstrap_id": int(bootstrap_id),
        "model_name": CI_MODEL_NAME,
        "risk_type": "death_adjusted",
        "condition_id": int(condition_id),
        "condition_name": CONDITION_NAMES[int(condition_id)],
        "metric": str(metric),
        "estimate": float(estimate) if np.isfinite(estimate) else np.nan,
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


def _atomic_csv(
    frame: pd.DataFrame,
    path: Path,
    compression: Optional[str] = None,
) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression=compression,
    )
    temporary.replace(path)


def run_participant_bootstrap_ci() -> Dict[str, pd.DataFrame]:
    """
    Generate the attached main model's canonical percentile-CI result set.

    Test participants are resampled as clusters. The fitted Delphi model,
    predictions, condition axis, and train-derived censoring KMs remain frozen.
    """
    dense_prediction_path = PREDICTION_PATHS["adjusted"]
    point_condition_path = (
        RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
    )
    cindex_reference_path = (
        RUN_DIR / "test_per_condition_metrics_delphi_extended.csv"
    )
    dense_point_path = (
        RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv"
    )
    integrated_point_path = (
        RUN_DIR / "test_integrated_metrics_by_condition.csv"
    )
    require_files([
        CANONICAL_PATHS["train"],
        CANONICAL_PATHS["test"],
        dense_prediction_path,
        PREDICTION_PATHS["manifest"],
        integrated_point_path,
        point_condition_path,
        cindex_reference_path,
        dense_point_path,
    ])

    if not NUMBA_AVAILABLE and CFG.ci_bootstrap_n >= 100:
        raise ImportError(
            "The publication bootstrap requires numba for the exact Wolbers "
            "C-index. Install numba, restart the kernel, and rerun this cell. "
            "A smoke test below 100 replicates can use the exact Python fallback."
        )
    if not NUMBA_AVAILABLE:
        warnings.warn(
            "numba is unavailable; the exact C-index smoke test may be slow.",
            RuntimeWarning,
        )

    test_samples_ci = pd.read_pickle(CANONICAL_PATHS["test"]).copy()
    train_samples_ci = pd.read_pickle(CANONICAL_PATHS["train"]).copy()
    with PREDICTION_PATHS["manifest"].open(encoding="utf-8") as handle:
        prediction_manifest = json.load(handle)
    expected_eid_hash = canonical_json_hash(
        test_samples_ci["eid"].astype(str).tolist()
    )
    if prediction_manifest.get("test_eid_axis_sha256") != expected_eid_hash:
        raise AssertionError(
            "Dense prediction rows do not match canonical test eid order."
        )
    if (
        prediction_manifest.get("condition_axis_sha256")
        != canonical_json_hash(CONDITION_NAMES)
    ):
        raise AssertionError(
            "Dense prediction condition order differs from the canonical axis."
        )
    dense_eids = test_samples_ci["eid"].astype(str).to_numpy()
    dense_condition_names = list(CONDITION_NAMES)
    dense_horizons = tuple(int(value) for value in DENSE_HORIZONS_MONTHS)
    dense_gap = float(CFG.gap_months)
    dense_risk = np.load(dense_prediction_path, mmap_mode="r")

    test_eids = test_samples_ci["eid"].astype(str).to_numpy()
    if pd.Series(test_eids).duplicated().any():
        raise AssertionError(
            "Canonical test samples are not one row per participant."
        )
    if dense_eids.tolist() != test_eids.tolist():
        raise AssertionError(
            "Dense prediction rows do not match canonical test eid order."
        )
    if dense_condition_names != CONDITION_NAMES:
        raise AssertionError(
            "Dense prediction condition order differs from the canonical axis."
        )
    if not np.isclose(dense_gap, float(CFG.gap_months)):
        raise AssertionError(
            "Dense prediction lead-time gap differs from the current contract."
        )
    expected_shape = (
        len(test_samples_ci),
        K,
        len(dense_horizons),
    )
    if dense_risk.shape != expected_shape:
        raise ValueError(
            f"Dense risk shape {dense_risk.shape} != {expected_shape}."
        )
    for row_start in range(0, len(test_samples_ci), 1024):
        risk_block = np.asarray(
            dense_risk[row_start:row_start + 1024],
            dtype=np.float32,
        )
        if not np.isfinite(risk_block).all():
            raise ValueError("Dense risk cube contains non-finite values.")
        if float(risk_block.min()) < -1e-6 or float(risk_block.max()) > 1.0 + 1e-6:
            raise ValueError("Dense risk cube contains values outside [0, 1].")

    report_horizons = tuple(int(x) for x in CI_POINT_HORIZONS_MONTHS)
    missing_report_horizons = sorted(
        set(report_horizons).difference(dense_horizons)
    )
    if missing_report_horizons:
        raise ValueError(
            "Report horizons are absent from the dense prediction grid: "
            f"{missing_report_horizons}"
        )
    if int(CFG.ci_cindex_tau_months) not in dense_horizons:
        raise ValueError(
            "The C-index tau must be present in the dense horizon grid."
        )
    horizon_to_index = {
        int(horizon): int(index)
        for index, horizon in enumerate(dense_horizons)
    }

    selected_raw = os.getenv("EHR_DELPHI_CI_CONDITION_IDS", "").strip()
    selected_condition_ids = (
        tuple(range(K))
        if not selected_raw
        else tuple(
            int(value.strip())
            for value in selected_raw.split(",")
            if value.strip()
        )
    )
    if len(set(selected_condition_ids)) != len(selected_condition_ids):
        raise ValueError("EHR_DELPHI_CI_CONDITION_IDS contains duplicates.")
    invalid_conditions = [
        value
        for value in selected_condition_ids
        if value < 0 or value >= K
    ]
    if invalid_conditions:
        raise ValueError(
            "EHR_DELPHI_CI_CONDITION_IDS contains out-of-range values: "
            f"{invalid_conditions}"
        )

    train_outcomes = make_condition_outcomes(train_samples_ci)
    test_outcomes = make_condition_outcomes(test_samples_ci)
    condition_kms = []
    condition_cindex_kms = []
    for condition in range(K):
        selected_train = train_outcomes["eligible"][:, condition]
        censor_event = (
            train_outcomes["status"][selected_train, condition] == 0
        )
        condition_kms.append(
            CensoringKM(
                train_outcomes["exit_time"][selected_train, condition],
                censor_event,
            )
        )
        condition_cindex_kms.append(
            CensoringKM(
                train_outcomes["exit_time_untruncated"][
                    selected_train,
                    condition,
                ],
                censor_event,
            )
        )

    train_null = np.full(
        (K, len(dense_horizons)),
        np.nan,
        dtype=np.float64,
    )
    for condition in selected_condition_ids:
        selected_train = train_outcomes["eligible"][:, condition]
        for grid_index, horizon in enumerate(dense_horizons):
            train_null[condition, grid_index] = aalen_johansen_cif(
                train_outcomes["exit_time"][selected_train, condition],
                train_outcomes["status"][selected_train, condition],
                float(horizon),
            )

    def evaluate_sample(
        index: np.ndarray,
        bootstrap_id: int,
    ) -> pd.DataFrame:
        index = np.asarray(index, dtype=np.int64)
        rows: List[Dict[str, Any]] = []
        for condition in selected_condition_ids:
            eligible = test_outcomes["eligible"][index, condition]
            status = test_outcomes["status"][index, condition]
            exit_time = test_outcomes["exit_time"][index, condition]
            exit_time_untruncated = test_outcomes[
                "exit_time_untruncated"
            ][index, condition]
            prediction_by_horizon = dense_risk[index, condition, :]
            eligible_n = int(eligible.sum())
            brier_values: List[float] = []
            null_brier_values: List[float] = []
            nbll_values: List[float] = []
            auc_values: List[float] = []
            calibration_values: List[float] = []

            for grid_index, horizon in enumerate(dense_horizons):
                prediction = prediction_by_horizon[:, grid_index]
                window = binary_window(
                    eligible,
                    status,
                    exit_time,
                    float(horizon),
                    condition_kms[condition],
                )
                discrimination = competing_risk_auc(window, prediction)
                scores = ipcw_ht_scores(
                    window["y"],
                    prediction,
                    window["weights"],
                    eligible_n,
                )
                null_prediction = np.full_like(
                    prediction,
                    train_null[condition, grid_index],
                    dtype=np.float64,
                )
                null_scores = ipcw_ht_scores(
                    window["y"],
                    null_prediction,
                    window["weights"],
                    eligible_n,
                )
                observed = aalen_johansen_cif(
                    exit_time[eligible],
                    status[eligible],
                    float(horizon),
                )
                mean_prediction = (
                    float(prediction[eligible].mean())
                    if eligible_n
                    else np.nan
                )
                calibration_in_large = (
                    mean_prediction - observed
                    if np.isfinite(mean_prediction) and np.isfinite(observed)
                    else np.nan
                )
                absolute_calibration_error = (
                    abs(calibration_in_large)
                    if np.isfinite(calibration_in_large)
                    else np.nan
                )
                observed_to_expected = (
                    observed / mean_prediction
                    if (
                        np.isfinite(observed)
                        and np.isfinite(mean_prediction)
                        and mean_prediction > 0
                    )
                    else np.nan
                )
                null_brier = null_scores["brier_ipcw_ht"]
                scaled_brier = (
                    1.0 - scores["brier_ipcw_ht"] / null_brier
                    if np.isfinite(null_brier) and null_brier > 0
                    else np.nan
                )
                brier_values.append(scores["brier_ipcw_ht"])
                null_brier_values.append(null_brier)
                nbll_values.append(scores["nbll_ipcw_ht"])
                auc_values.append(discrimination["auc_ipcw_cr"])
                calibration_values.append(absolute_calibration_error)

                if int(horizon) in report_horizons:
                    values = {
                        "auc_ipcw_cr": discrimination["auc_ipcw_cr"],
                        "auprc_ipcw_cr": discrimination["auprc_ipcw_cr"],
                        "brier_ipcw_ht": scores["brier_ipcw_ht"],
                        "brier_null_ipcw_ht": null_brier,
                        "scaled_brier_ipcw_ht": scaled_brier,
                        "nbll_ipcw_ht": scores["nbll_ipcw_ht"],
                        "mean_predicted_cif": mean_prediction,
                        "observed_aj_cif": observed,
                        "calibration_in_large_pred_minus_obs": calibration_in_large,
                        "observed_to_expected_ratio": observed_to_expected,
                        "absolute_calibration_error": absolute_calibration_error,
                    }
                    event_n = int(window["case"].sum())
                    for metric in CI_POINT_METRICS:
                        rows.append(
                            _ci_metric_row(
                                bootstrap_id,
                                condition,
                                metric,
                                values[metric],
                                horizon_months=int(horizon),
                                eligible_n=eligible_n,
                                event_n=event_n,
                            )
                        )

            ibs = integrate_metric(brier_values, dense_horizons)
            ibs_null = integrate_metric(
                null_brier_values,
                dense_horizons,
            )
            integrated_values = {
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
                "inbll_ipcw_ht": integrate_metric(
                    nbll_values,
                    dense_horizons,
                ),
                "iauc_ipcw_cr": integrate_metric(
                    auc_values,
                    dense_horizons,
                ),
                "integrated_abs_calibration_error": integrate_metric(
                    calibration_values,
                    dense_horizons,
                ),
            }
            end_horizon = int(dense_horizons[-1])
            end_event_n = int(
                (
                    eligible
                    & (status == 1)
                    & (exit_time <= end_horizon + 1e-8)
                ).sum()
            )
            for metric in CI_INTEGRATED_METRICS:
                rows.append(
                    _ci_metric_row(
                        bootstrap_id,
                        condition,
                        metric,
                        integrated_values[metric],
                        integration_start_months=int(dense_horizons[0]),
                        integration_end_months=end_horizon,
                        eligible_n=eligible_n,
                        event_n=end_event_n,
                    )
                )

            tau = int(CFG.ci_cindex_tau_months)
            tau_prediction = prediction_by_horizon[
                :,
                horizon_to_index[tau],
            ]
            cindex = competing_risk_cindex(
                eligible,
                status,
                exit_time_untruncated,
                tau_prediction,
                float(tau),
                condition_cindex_kms[condition],
            )
            cindex_event_n = int(
                (
                    eligible
                    & (status == 1)
                    & (exit_time_untruncated <= tau + 1e-8)
                ).sum()
            )
            rows.append(
                _ci_metric_row(
                    bootstrap_id,
                    condition,
                    "cindex_ipcw_cr",
                    cindex["cindex_ipcw_cr"],
                    horizon_months=tau,
                    eligible_n=eligible_n,
                    event_n=cindex_event_n,
                    comparable_pair_weight=(
                        cindex["cindex_pair_weight_sum"]
                    ),
                )
            )
        return pd.DataFrame(rows)

    print("=" * 88)
    print("DELPHI PARTICIPANT-BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 88)
    print({
        "test_participants": len(test_samples_ci),
        "conditions_selected": len(selected_condition_ids),
        "bootstrap_replicates": CFG.ci_bootstrap_n,
        "report_horizons_months": report_horizons,
        "dense_horizons_months": dense_horizons,
        "cindex_tau_months": CFG.ci_cindex_tau_months,
        "point_metrics": CI_POINT_METRICS,
        "integrated_metrics": CI_INTEGRATED_METRICS,
        "n_jobs": max(1, CFG.ci_n_jobs),
        "numba_available": NUMBA_AVAILABLE,
    })
    started = time.perf_counter()
    point = evaluate_sample(
        np.arange(len(test_samples_ci), dtype=np.int64),
        -1,
    )

    dense_reference = pd.read_csv(dense_point_path).query(
        "risk_type == 'death_adjusted'"
    )
    integrated_reference = pd.read_csv(integrated_point_path).query(
        "risk_type == 'death_adjusted'"
    )
    cindex_reference = pd.read_csv(cindex_reference_path).query(
        "risk_type == 'death_adjusted'"
    )
    maximum_point_difference = 0.0
    for row in point.itertuples(index=False):
        if row.metric == "cindex_ipcw_cr":
            reference = cindex_reference[
                cindex_reference["condition_id"].eq(row.condition_id)
                & cindex_reference["horizon_months"].eq(row.horizon_months)
            ]
            reference_value = (
                float(reference["cindex_ipcw_cr"].iloc[0])
                if len(reference) == 1
                else np.nan
            )
        elif np.isfinite(row.horizon_months):
            reference = dense_reference[
                dense_reference["condition_id"].eq(row.condition_id)
                & dense_reference["horizon_months"].eq(row.horizon_months)
            ]
            if len(reference) != 1:
                raise AssertionError(
                    "Dense point-reference lookup was not one-to-one."
                )
            reference_row = reference.iloc[0]
            if row.metric in reference_row.index:
                reference_value = float(reference_row[row.metric])
            elif row.metric == "scaled_brier_ipcw_ht":
                brier = float(reference_row["brier_ipcw_ht"])
                null_brier = float(reference_row["brier_null_ipcw_ht"])
                reference_value = (
                    1.0 - brier / null_brier
                    if np.isfinite(null_brier) and null_brier > 0
                    else np.nan
                )
            elif row.metric == "calibration_in_large_pred_minus_obs":
                reference_value = float(
                    reference_row["mean_predicted_cif"]
                    - reference_row["observed_aj_cif"]
                )
            elif row.metric == "observed_to_expected_ratio":
                predicted = float(reference_row["mean_predicted_cif"])
                observed = float(reference_row["observed_aj_cif"])
                reference_value = (
                    observed / predicted
                    if np.isfinite(predicted) and predicted > 0
                    else np.nan
                )
            elif row.metric == "absolute_calibration_error":
                reference_value = abs(
                    float(reference_row["mean_predicted_cif"])
                    - float(reference_row["observed_aj_cif"])
                )
            else:
                raise KeyError(
                    f"No point-reference mapping for {row.metric}."
                )
        else:
            reference = integrated_reference[
                integrated_reference["condition_id"].eq(row.condition_id)
            ]
            if len(reference) != 1:
                raise AssertionError(
                    "Integrated point-reference lookup was not one-to-one."
                )
            reference_value = float(reference[row.metric].iloc[0])

        calculated_value = float(row.estimate)
        if np.isfinite(calculated_value) and np.isfinite(reference_value):
            difference = abs(calculated_value - reference_value)
            maximum_point_difference = max(
                maximum_point_difference,
                difference,
            )
            if not np.isclose(
                calculated_value,
                reference_value,
                rtol=5e-5,
                atol=1e-7,
            ):
                raise AssertionError(
                    "Bootstrap point estimate differs from its canonical "
                    f"reference: condition={row.condition_id}, "
                    f"metric={row.metric}, calculated={calculated_value}, "
                    f"reference={reference_value}."
                )
        elif not (
            np.isnan(calculated_value) and np.isnan(reference_value)
        ):
            raise AssertionError(
                "Bootstrap and canonical point estimates disagree on "
                "finite/non-finite status."
            )

    configuration = {
        "schema_version": "delphi-participant-bootstrap-ci.v1",
        "implementation_version": "2026-07-29-five-year-cindex-all-canonical-metrics",
        "model_name": CI_MODEL_NAME,
        "bootstrap_n": int(CFG.ci_bootstrap_n),
        "bootstrap_seed": int(CFG.ci_bootstrap_seed),
        "alpha": float(CFG.ci_alpha),
        "ci_method": "percentile",
        "resampling_unit": "eid",
        "point_horizons_months": list(report_horizons),
        "point_metrics": list(CI_POINT_METRICS),
        "integrated_metrics": list(CI_INTEGRATED_METRICS),
        "cindex_tau_months": int(CFG.ci_cindex_tau_months),
        "dense_horizons_months": list(dense_horizons),
        "selected_condition_ids": list(selected_condition_ids),
        "gap_months": float(CFG.gap_months),
        "test_n": int(len(test_samples_ci)),
        "condition_n": int(K),
        "eid_axis_sha256": canonical_json_hash(test_eids.tolist()),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "censoring_source": str(CANONICAL_PATHS["train"]),
        "censoring_frozen_across_bootstraps": True,
        "model_frozen_across_bootstraps": True,
        "integrated_auroc_included": False,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": (
            sklearn.__version__ if sklearn is not None else "unavailable"
        ),
        "python_version": platform.python_version(),
    }
    signature = hashlib.sha256(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    chunk_directory = RUN_DIR / f"test_bootstrap_ci_chunks_{signature}"
    chunk_directory.mkdir(parents=True, exist_ok=True)
    configuration_path = chunk_directory / "configuration.json"
    if configuration_path.is_file():
        with configuration_path.open(encoding="utf-8") as handle:
            existing_configuration = json.load(handle)
        if existing_configuration != configuration:
            raise RuntimeError(
                "Existing bootstrap chunks use a different configuration."
            )
    else:
        atomic_json_dump(configuration, configuration_path)

    keyed_point = _ci_group_keys(point)
    key_columns = [
        "condition_id", "condition_name", "metric", "horizon_key",
        "integration_start_key", "integration_end_key",
    ]
    point_key_order = [
        tuple(row)
        for row in keyed_point[key_columns].itertuples(
            index=False,
            name=None,
        )
    ]
    n_metric_keys = len(point_key_order)
    expected_rows_per_replicate = n_metric_keys
    matrix_path = chunk_directory / "bootstrap_estimates.float64.mmap"
    estimate_matrix = np.memmap(
        matrix_path,
        mode="w+",
        dtype=np.float64,
        shape=(int(CFG.ci_bootstrap_n), n_metric_keys),
    )
    estimate_matrix[:] = np.nan
    chunk_paths: List[Path] = []

    def validate_and_store_chunk(
        chunk_frame: pd.DataFrame,
        chunk_ids: Sequence[int],
    ) -> None:
        expected_rows = len(chunk_ids) * expected_rows_per_replicate
        if len(chunk_frame) != expected_rows:
            raise RuntimeError(
                f"Bootstrap chunk has {len(chunk_frame)} rows; "
                f"expected {expected_rows}."
            )
        keyed_chunk = _ci_group_keys(chunk_frame)
        for bootstrap_id in chunk_ids:
            group = keyed_chunk[
                keyed_chunk["bootstrap_id"].eq(int(bootstrap_id))
            ]
            observed_keys = [
                tuple(row)
                for row in group[key_columns].itertuples(
                    index=False,
                    name=None,
                )
            ]
            if observed_keys != point_key_order:
                raise RuntimeError(
                    f"Bootstrap replicate {bootstrap_id} changed metric order."
                )
            estimate_matrix[int(bootstrap_id), :] = pd.to_numeric(
                group["estimate"],
                errors="coerce",
            ).to_numpy(float)

    for chunk_start in range(
        0,
        int(CFG.ci_bootstrap_n),
        int(CFG.ci_chunk_size),
    ):
        chunk_stop = min(
            chunk_start + int(CFG.ci_chunk_size),
            int(CFG.ci_bootstrap_n),
        )
        chunk_ids = list(range(chunk_start, chunk_stop))
        chunk_path = chunk_directory / (
            f"bootstrap_{chunk_start:05d}_{chunk_stop - 1:05d}.csv.gz"
        )
        if chunk_path.is_file() and not CFG.ci_force_recompute:
            chunk_frame = pd.read_csv(chunk_path)
            validate_and_store_chunk(chunk_frame, chunk_ids)
            chunk_paths.append(chunk_path)
            print(
                f"Reused bootstrap replicates {chunk_start + 1}–"
                f"{chunk_stop} of {CFG.ci_bootstrap_n}."
            )
            continue

        def evaluate_replicate(bootstrap_id: int) -> pd.DataFrame:
            seed_sequence = np.random.SeedSequence([
                int(CFG.ci_bootstrap_seed),
                int(bootstrap_id),
            ])
            generator = np.random.default_rng(seed_sequence)
            bootstrap_index = generator.integers(
                0,
                len(test_samples_ci),
                size=len(test_samples_ci),
                dtype=np.int64,
            )
            return evaluate_sample(bootstrap_index, bootstrap_id)

        if JOBLIB_AVAILABLE and max(1, CFG.ci_n_jobs) > 1:
            results = Parallel(
                n_jobs=max(1, CFG.ci_n_jobs),
                prefer="threads",
            )(
                delayed(evaluate_replicate)(bootstrap_id)
                for bootstrap_id in chunk_ids
            )
        else:
            results = [
                evaluate_replicate(bootstrap_id)
                for bootstrap_id in chunk_ids
            ]
        chunk_frame = pd.concat(results, ignore_index=True)
        validate_and_store_chunk(chunk_frame, chunk_ids)
        _atomic_csv(chunk_frame, chunk_path, compression="gzip")
        chunk_paths.append(chunk_path)
        print(
            f"Completed bootstrap replicates {chunk_start + 1}–"
            f"{chunk_stop} of {CFG.ci_bootstrap_n}."
        )
    estimate_matrix.flush()

    valid_count = np.sum(np.isfinite(estimate_matrix), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ci_lower = np.nanquantile(
            estimate_matrix,
            float(CFG.ci_alpha) / 2.0,
            axis=0,
        )
        ci_upper = np.nanquantile(
            estimate_matrix,
            1.0 - float(CFG.ci_alpha) / 2.0,
            axis=0,
        )
    valid_fraction = valid_count / float(CFG.ci_bootstrap_n)
    ci_status = np.where(
        valid_fraction >= float(CFG.ci_min_valid_fraction),
        "ok",
        "insufficient_valid_replicates",
    )
    ci_lower[ci_status != "ok"] = np.nan
    ci_upper[ci_status != "ok"] = np.nan

    per_condition_ci = point.drop(columns=["bootstrap_id"]).copy()
    per_condition_ci["n_bootstrap_total"] = int(CFG.ci_bootstrap_n)
    per_condition_ci["n_valid_bootstrap"] = valid_count.astype(int)
    per_condition_ci["ci_lower"] = ci_lower
    per_condition_ci["ci_upper"] = ci_upper
    per_condition_ci["valid_bootstrap_fraction"] = valid_fraction
    per_condition_ci["ci_level"] = 1.0 - float(CFG.ci_alpha)
    per_condition_ci["ci_method"] = "participant_bootstrap_percentile"
    per_condition_ci["ci_status"] = ci_status
    per_condition_ci = per_condition_ci[[
        "condition_id",
        "condition_name",
        "metric",
        "model_name",
        "risk_type",
        "estimate",
        "horizon_months",
        "integration_start_months",
        "integration_end_months",
        "eligible_n",
        "event_n",
        "comparable_pair_weight",
        "n_bootstrap_total",
        "n_valid_bootstrap",
        "ci_lower",
        "ci_upper",
        "valid_bootstrap_fraction",
        "ci_level",
        "ci_method",
        "ci_status",
    ]]

    macro_key_columns = [
        "metric", "horizon_key",
        "integration_start_key", "integration_end_key",
    ]
    macro_rows: List[Dict[str, Any]] = []
    macro_replicate_rows: List[Dict[str, Any]] = []
    for macro_key, group in keyed_point.groupby(
        macro_key_columns,
        sort=False,
    ):
        fixed_group = group[np.isfinite(group["estimate"])].copy()
        column_indices = fixed_group.index.to_numpy(dtype=int)
        fixed_condition_ids = fixed_group["condition_id"].astype(int).tolist()
        required_finite = (
            int(math.ceil(
                CFG.ci_macro_min_condition_fraction
                * len(column_indices)
            ))
            if len(column_indices)
            else 1
        )
        replicate_values = np.full(
            int(CFG.ci_bootstrap_n),
            np.nan,
            dtype=float,
        )
        if len(column_indices):
            selected_matrix = np.asarray(
                estimate_matrix[:, column_indices],
                dtype=float,
            )
            finite_counts = np.sum(
                np.isfinite(selected_matrix),
                axis=1,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                means = np.nanmean(selected_matrix, axis=1)
            replicate_values[finite_counts >= required_finite] = means[
                finite_counts >= required_finite
            ]
        else:
            finite_counts = np.zeros(
                int(CFG.ci_bootstrap_n),
                dtype=int,
            )
        valid_macro = replicate_values[np.isfinite(replicate_values)]
        valid_macro_fraction = (
            len(valid_macro) / float(CFG.ci_bootstrap_n)
        )
        macro_status = (
            "ok"
            if valid_macro_fraction >= CFG.ci_min_valid_fraction
            else "insufficient_valid_replicates"
        )
        metric, horizon_key, start_key, end_key = macro_key
        macro_rows.append({
            "model_name": CI_MODEL_NAME,
            "risk_type": "death_adjusted",
            "metric": str(metric),
            "horizon_months": (
                int(horizon_key) if int(horizon_key) >= 0 else np.nan
            ),
            "integration_start_months": (
                int(start_key) if int(start_key) >= 0 else np.nan
            ),
            "integration_end_months": (
                int(end_key) if int(end_key) >= 0 else np.nan
            ),
            "estimate": (
                float(fixed_group["estimate"].mean())
                if len(fixed_group)
                else np.nan
            ),
            "fixed_condition_n": int(len(fixed_condition_ids)),
            "fixed_condition_ids": "|".join(
                str(value) for value in fixed_condition_ids
            ),
            "ci_lower": (
                float(np.quantile(valid_macro, CFG.ci_alpha / 2.0))
                if macro_status == "ok"
                else np.nan
            ),
            "ci_upper": (
                float(np.quantile(
                    valid_macro,
                    1.0 - CFG.ci_alpha / 2.0,
                ))
                if macro_status == "ok"
                else np.nan
            ),
            "n_bootstrap_total": int(CFG.ci_bootstrap_n),
            "n_valid_bootstrap": int(len(valid_macro)),
            "valid_bootstrap_fraction": float(valid_macro_fraction),
            "ci_level": 1.0 - float(CFG.ci_alpha),
            "ci_method": "participant_bootstrap_percentile",
            "ci_status": macro_status,
            "minimum_condition_fraction_per_replicate": float(
                CFG.ci_macro_min_condition_fraction
            ),
        })
        for bootstrap_id, estimate in enumerate(replicate_values):
            macro_replicate_rows.append({
                "bootstrap_id": int(bootstrap_id),
                "model_name": CI_MODEL_NAME,
                "risk_type": "death_adjusted",
                "aggregation_level": "macro",
                "condition_id": -1,
                "condition_name": "MACRO",
                "metric": str(metric),
                "estimate": (
                    float(estimate) if np.isfinite(estimate) else np.nan
                ),
                "horizon_months": (
                    int(horizon_key) if int(horizon_key) >= 0 else np.nan
                ),
                "integration_start_months": (
                    int(start_key) if int(start_key) >= 0 else np.nan
                ),
                "integration_end_months": (
                    int(end_key) if int(end_key) >= 0 else np.nan
                ),
                "fixed_condition_n": int(len(fixed_condition_ids)),
                "finite_condition_n": int(finite_counts[bootstrap_id]),
            })
    macro_ci = pd.DataFrame(macro_rows)
    macro_replicates = pd.DataFrame(macro_replicate_rows)

    per_condition_path = (
        RUN_DIR / "test_per_condition_metrics_with_ci.csv"
    )
    point_horizon_path = (
        RUN_DIR / "test_point_horizon_metrics_with_ci.csv"
    )
    cindex_path = (
        RUN_DIR / "test_cindex_by_condition_with_ci.csv"
    )
    integrated_path = (
        RUN_DIR / "test_integrated_metrics_with_ci.csv"
    )
    macro_path = RUN_DIR / "test_macro_metrics_with_ci.csv"
    replicate_path = (
        RUN_DIR / "test_bootstrap_replicates_metrics.csv.gz"
    )
    metadata_path = RUN_DIR / "test_bootstrap_ci_metadata.json"

    point_horizon_ci = per_condition_ci[
        per_condition_ci["horizon_months"].notna()
        & per_condition_ci["metric"].ne("cindex_ipcw_cr")
    ].copy()
    cindex_ci = per_condition_ci[
        per_condition_ci["metric"].eq("cindex_ipcw_cr")
    ].copy()
    integrated_ci = per_condition_ci[
        per_condition_ci["integration_start_months"].notna()
    ].copy()

    _atomic_csv(per_condition_ci, per_condition_path)
    _atomic_csv(point_horizon_ci, point_horizon_path)
    _atomic_csv(cindex_ci, cindex_path)
    _atomic_csv(integrated_ci, integrated_path)
    _atomic_csv(macro_ci, macro_path)

    # Stream condition chunks plus macro rows into the main-model replicate file.
    temporary_replicate_path = replicate_path.with_name(
        replicate_path.name + ".tmp"
    )
    condition_columns = [
        "bootstrap_id", "model_name", "risk_type", "aggregation_level",
        "condition_id", "condition_name", "metric", "estimate",
        "horizon_months", "integration_start_months",
        "integration_end_months", "eligible_n", "event_n",
        "comparable_pair_weight", "fixed_condition_n",
        "finite_condition_n",
    ]
    wrote_header = False
    with gzip.open(
        temporary_replicate_path,
        mode="wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        for chunk_path in chunk_paths:
            chunk = pd.read_csv(chunk_path)
            chunk.insert(3, "aggregation_level", "condition")
            for column in condition_columns:
                if column not in chunk.columns:
                    chunk[column] = np.nan
            chunk[condition_columns].to_csv(
                handle,
                index=False,
                header=not wrote_header,
            )
            wrote_header = True
        for column in condition_columns:
            if column not in macro_replicates.columns:
                macro_replicates[column] = np.nan
        macro_replicates[condition_columns].to_csv(
            handle,
            index=False,
            header=not wrote_header,
        )
    temporary_replicate_path.replace(replicate_path)

    # The attached main notebook leaves its legacy *_summary_* bootstrap files
    # and publication_with_ci table as no-CI compatibility placeholders. Keep
    # those public files unchanged and write this convenient enriched view only
    # as an explicitly Delphi-named sidecar.
    publication_with_ci = pd.read_csv(
        RUN_DIR / "test_summary_publication.csv"
    )
    publication_with_ci["ci_computed"] = True
    publication_with_ci["ci_lower"] = np.nan
    publication_with_ci["ci_upper"] = np.nan
    publication_with_ci["ci_detail_file"] = str(macro_path)
    for output_column, metric in {
        "macro_auroc": "auc_ipcw_cr",
        "macro_auprc": "auprc_ipcw_cr",
        "macro_brier_ipcw": "brier_ipcw_ht",
        "macro_nbll_ipcw": "nbll_ipcw_ht",
        "macro_cindex_ipcw_cr": "cindex_ipcw_cr",
    }.items():
        publication_with_ci[f"{output_column}_ci_lower"] = np.nan
        publication_with_ci[f"{output_column}_ci_upper"] = np.nan
        for row_index, row in publication_with_ci.iterrows():
            horizon = pd.to_numeric(
                pd.Series([row["horizon_months"]]),
                errors="coerce",
            ).iloc[0]
            if not np.isfinite(horizon):
                continue
            selected = macro_ci[
                macro_ci["metric"].eq(metric)
                & macro_ci["horizon_months"].eq(int(horizon))
            ]
            if len(selected) == 1:
                publication_with_ci.loc[
                    row_index,
                    f"{output_column}_ci_lower",
                ] = selected["ci_lower"].iloc[0]
                publication_with_ci.loc[
                    row_index,
                    f"{output_column}_ci_upper",
                ] = selected["ci_upper"].iloc[0]
    publication_with_ci.to_csv(
        RUN_DIR / "test_summary_publication_with_ci_delphi_extended.csv",
        index=False,
    )

    elapsed = float(time.perf_counter() - started)
    metadata = {
        **configuration,
        "status": "completed",
        "configuration_signature": signature,
        "bootstrap_seed_rule": (
            "default_rng(SeedSequence([bootstrap_seed, bootstrap_id]))"
            ".integers(0, test_n, size=test_n)"
        ),
        "point_estimate_validation": {
            "passed": True,
            "maximum_absolute_difference": maximum_point_difference,
            "reference_point_file": str(dense_point_path),
            "reference_integrated_point_file": str(integrated_point_path),
            "reference_cindex_file": str(cindex_reference_path),
        },
        "cindex_definition": {
            "name": (
                "Wolbers-style truncated IPCW competing-risk C-index"
            ),
            "tau_months_after_gap": int(CFG.ci_cindex_tau_months),
            "prediction": "death-adjusted condition CIF",
            "censoring_model": (
                "condition-specific marginal KM from canonical train split "
                "using untruncated observed follow-up times"
            ),
            "prediction_ties": "counted_as_one_half",
        },
        "confidence_interval_scope": {
            "fixed_horizon_metrics": list(CI_POINT_METRICS),
            "integrated_metrics": list(CI_INTEGRATED_METRICS),
            "cindex": "cindex_ipcw_cr",
            "aggregation_levels": ["condition", "macro"],
        },
        "validity_thresholds": {
            "minimum_valid_bootstrap_fraction": float(
                CFG.ci_min_valid_fraction
            ),
            "macro_minimum_condition_fraction_per_replicate": float(
                CFG.ci_macro_min_condition_fraction
            ),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "n_jobs": max(1, int(CFG.ci_n_jobs)),
            "joblib_available": JOBLIB_AVAILABLE,
            "numba_available": NUMBA_AVAILABLE,
            "chunk_size": int(CFG.ci_chunk_size),
            "chunk_directory": str(chunk_directory),
            "estimate_memmap": str(matrix_path),
        },
        "output_files": {
            "per_condition_ci": str(per_condition_path),
            "point_horizon_ci": str(point_horizon_path),
            "cindex_ci": str(cindex_path),
            "integrated_ci": str(integrated_path),
            "macro_ci": str(macro_path),
            "bootstrap_replicates": str(replicate_path),
            "metadata": str(metadata_path),
        },
    }
    atomic_json_dump(metadata, metadata_path)
    artifact_hash_table().to_csv(
        RUN_DIR / "artifact_sha256.csv",
        index=False,
    )

    expected_per_condition_rows = (
        len(selected_condition_ids)
        * (
            len(report_horizons) * len(CI_POINT_METRICS)
            + len(CI_INTEGRATED_METRICS)
            + 1
        )
    )
    if len(per_condition_ci) != expected_per_condition_rows:
        raise RuntimeError(
            "Final per-condition CI table has an unexpected row count."
        )
    if len(cindex_ci) != len(selected_condition_ids):
        raise RuntimeError(
            "C-index CI table must contain one row per selected condition."
        )
    if per_condition_ci["n_bootstrap_total"].ne(
        int(CFG.ci_bootstrap_n)
    ).any():
        raise RuntimeError(
            "At least one CI row lacks the requested replicate count."
        )
    expected_ci_columns = [
        "condition_id", "condition_name", "metric", "model_name",
        "risk_type", "estimate", "horizon_months",
        "integration_start_months", "integration_end_months",
        "eligible_n", "event_n", "comparable_pair_weight",
        "n_bootstrap_total", "n_valid_bootstrap", "ci_lower", "ci_upper",
        "valid_bootstrap_fraction", "ci_level", "ci_method", "ci_status",
    ]
    expected_macro_ci_columns = [
        "model_name", "risk_type", "metric", "horizon_months",
        "integration_start_months", "integration_end_months", "estimate",
        "fixed_condition_n", "fixed_condition_ids", "ci_lower", "ci_upper",
        "n_bootstrap_total", "n_valid_bootstrap",
        "valid_bootstrap_fraction", "ci_level", "ci_method", "ci_status",
        "minimum_condition_fraction_per_replicate",
    ]
    if per_condition_ci.columns.tolist() != expected_ci_columns:
        raise RuntimeError(
            "Per-condition CI columns/order differ from the locked main schema."
        )
    if macro_ci.columns.tolist() != expected_macro_ci_columns:
        raise RuntimeError(
            "Macro CI columns/order differ from the locked main schema."
        )
    if len(point_horizon_ci) != (
        len(selected_condition_ids)
        * len(CI_POINT_HORIZONS_MONTHS)
        * len(CI_POINT_METRICS)
    ):
        raise RuntimeError("Point-horizon CI table has the wrong row count.")
    if len(integrated_ci) != (
        len(selected_condition_ids) * len(CI_INTEGRATED_METRICS)
    ):
        raise RuntimeError("Integrated CI table has the wrong row count.")
    saved_replicate_columns = pd.read_csv(
        replicate_path, compression="gzip", nrows=0
    ).columns.tolist()
    if saved_replicate_columns != condition_columns:
        raise RuntimeError(
            "Bootstrap replicate columns/order differ from the locked main "
            "schema."
        )
    for legacy_bootstrap_path in (
        RUN_DIR / "test_bootstrap_replicates_summary.csv",
        RUN_DIR / "test_bootstrap_summary_cis.csv",
    ):
        if legacy_bootstrap_path.read_text().strip():
            raise RuntimeError(
                "A locked legacy bootstrap placeholder was unexpectedly "
                f"overwritten: {legacy_bootstrap_path}"
            )

    print("=" * 88)
    print("DELPHI PARTICIPANT-BOOTSTRAP CIs COMPLETE")
    print("=" * 88)
    print(
        f"Elapsed: {elapsed / 60.0:.1f} minutes | "
        f"replicates: {CFG.ci_bootstrap_n:,} | "
        f"conditions: {len(selected_condition_ids):,}"
    )
    print(macro_ci[[
        "metric", "horizon_months", "estimate", "ci_lower",
        "ci_upper", "n_valid_bootstrap", "ci_status",
    ]])
    return {
        "per_condition_ci": per_condition_ci,
        "point_horizon_ci": point_horizon_ci,
        "cindex_ci": cindex_ci,
        "integrated_ci": integrated_ci,
        "macro_ci": macro_ci,
    }


# =============================================================================
# Complete evaluation wrapper shared by both supported workflows.
# =============================================================================

def run_full_results(
    checkpoint_path: Path,
    evaluation_mode: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generate all existing point outputs plus the complete additive CI set."""
    point_results = run_model_test_evaluation(
        Path(checkpoint_path),
        evaluation_mode=evaluation_mode,
    )
    ci_results = run_participant_bootstrap_ci()
    full_results = {
        "point_and_integrated_metrics": point_results,
        "confidence_intervals": ci_results,
    }
    atomic_json_dump(
        {
            "schema_version": CFG.schema_version,
            "workflow": CFG.workflow,
            "evaluation_mode": evaluation_mode,
            "checkpoint_path": str(
                Path(checkpoint_path).expanduser().resolve()
            ),
            "cindex_tau_months_after_gap": int(
                CFG.ci_cindex_tau_months
            ),
            "completed_unix_time": time.time(),
            "point_evaluation_complete": str(
                RUN_DIR / "TEST_EVALUATION_COMPLETE.json"
            ),
            "bootstrap_ci_metadata": str(
                RUN_DIR / "test_bootstrap_ci_metadata.json"
            ),
        },
        RUN_DIR / "FULL_RESULTS_COMPLETE.json",
    )
    return full_results
