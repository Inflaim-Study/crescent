# Penalised Cox — Ridge-penalised linear Cox model and numerical self-tests
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 17 ---
def signed_cox_labels(
    exit_time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """XGBoost Cox labels: positive event time, negative censored time."""
    time_array = np.asarray(exit_time, dtype=np.float64)
    event_array = np.asarray(event, dtype=bool)
    if time_array.shape != event_array.shape:
        raise ValueError("Survival time and event axes are inconsistent.")
    if not np.all(np.isfinite(time_array)) or np.any(time_array < 0):
        raise ValueError("Survival times must be finite and non-negative.")
    safe_time = np.maximum(time_array, 1e-6)
    return np.where(event_array, safe_time, -safe_time).astype(np.float32)


def make_dmatrix(
    matrix: sparse.csr_matrix,
    exit_time: Optional[np.ndarray] = None,
    event: Optional[np.ndarray] = None,
) -> xgb.DMatrix:
    label = None
    if exit_time is not None or event is not None:
        if exit_time is None or event is None:
            raise ValueError("Both time and event are required for Cox labels.")
        label = signed_cox_labels(exit_time, event)
    return xgb.DMatrix(
        matrix,
        label=label,
        missing=np.nan,
        nthread=CFG.nthread,
    )


def harrell_c_index(
    exit_time: np.ndarray,
    event: np.ndarray,
    risk_score: np.ndarray,
) -> float:
    """Exact Harrell C with strict time comparability and 0.5 risk ties."""
    times = np.asarray(exit_time, dtype=np.float64)
    events = np.asarray(event, dtype=bool)
    scores = np.asarray(risk_score, dtype=np.float64)
    valid = np.isfinite(times) & np.isfinite(scores) & (times >= 0.0)
    times, events, scores = times[valid], events[valid], scores[valid]
    if not np.any(events) or len(times) < 2:
        return np.nan

    unique_scores = np.unique(scores)
    ranks = np.searchsorted(unique_scores, scores).astype(np.int64) + 1
    tree = np.zeros(len(unique_scores) + 1, dtype=np.int64)

    def update(index: int) -> None:
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def prefix(index: int) -> int:
        total = 0
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    order = np.argsort(-times, kind="mergesort")
    times, events, ranks = times[order], events[order], ranks[order]
    group_start = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    group_end = np.r_[group_start[1:], len(times)]
    concordant = 0.0
    comparable = 0
    later_n = 0
    for start, stop in zip(group_start, group_end):
        for position in range(int(start), int(stop)):
            if not events[position]:
                continue
            rank = int(ranks[position])
            lower = prefix(rank - 1)
            equal = prefix(rank) - lower
            concordant += float(lower) + 0.5 * float(equal)
            comparable += int(later_n)
        for position in range(int(start), int(stop)):
            update(int(ranks[position]))
            later_n += 1
    return float(concordant / comparable) if comparable > 0 else np.nan


def breslow_baseline_hazard(
    margin: np.ndarray,
    exit_time: np.ndarray,
    event: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Training-only Breslow increments on a numerically centered margin."""
    time_array = np.asarray(exit_time, dtype=np.float64)
    event_array = np.asarray(event, dtype=bool)
    eta = np.asarray(margin, dtype=np.float64)
    if not (time_array.shape == event_array.shape == eta.shape):
        raise ValueError("Baseline-hazard axes are inconsistent.")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(eta)):
        raise ValueError("Baseline-hazard inputs must be finite.")
    margin_center = float(np.max(eta)) if eta.size else 0.0
    if int(event_array.sum()) == 0:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            margin_center,
        )

    order = np.argsort(-time_array, kind="mergesort")
    times = time_array[order]
    events = event_array[order]
    centered_eta = eta[order] - margin_center
    weights = np.exp(np.clip(centered_eta, -700.0, 0.0))
    group_start = np.flatnonzero(np.r_[True, times[1:] != times[:-1]])
    group_times = times[group_start]
    group_weight = np.add.reduceat(weights, group_start)
    risk_weight = np.cumsum(group_weight)
    event_count = np.add.reduceat(events.astype(np.int64), group_start)
    selected = np.flatnonzero(event_count > 0)
    increments = event_count[selected].astype(np.float64) / risk_weight[selected]
    return (
        np.asarray(group_times[selected][::-1], dtype=np.float64),
        np.asarray(increments[::-1], dtype=np.float64),
        margin_center,
    )


def linear_cox_params(ridge_lambda: float, seed: int) -> Dict[str, Any]:
    return {
        "booster": "gblinear",
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "updater": "coord_descent",
        "feature_selector": "cyclic",
        "eta": float(CFG.learning_rate),
        "lambda": float(ridge_lambda),
        "alpha": 0.0,
        "seed": int(seed),
        "nthread": int(CFG.nthread),
        "verbosity": 0,
        "validate_parameters": True,
    }


def booster_from_raw(raw: Optional[bytes]) -> Optional[xgb.Booster]:
    if raw is None:
        return None
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    return booster


def predict_margin_from_dmatrix(
    model: Mapping[str, Any],
    dmatrix: xgb.DMatrix,
    n_rows: int,
) -> np.ndarray:
    booster = booster_from_raw(model.get("booster_raw"))
    if booster is None:
        return np.full(
            int(n_rows),
            float(model.get("constant_margin", 0.0)),
            dtype=np.float64,
        )
    margin = np.asarray(
        booster.predict(dmatrix, output_margin=True), dtype=np.float64
    ).reshape(-1)
    if margin.size != int(n_rows) or not np.all(np.isfinite(margin)):
        raise RuntimeError("Linear CoxPH solver produced invalid margins.")
    return margin


def predict_margin(
    model: Mapping[str, Any],
    matrix: sparse.csr_matrix,
) -> np.ndarray:
    return predict_margin_from_dmatrix(
        model, make_dmatrix(matrix), matrix.shape[0]
    )


def fit_linear_cox_candidate(
    dtrain: xgb.DMatrix,
    train_time: np.ndarray,
    train_event: np.ndarray,
    dvalidation: xgb.DMatrix,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    ridge_lambda: float,
    seed: int,
) -> Dict[str, Any]:
    started = time.time()
    evaluation_history: Dict[str, Dict[str, List[float]]] = {}
    evals = [(dtrain, "train")]
    validation_has_events = bool(np.asarray(validation_event, dtype=bool).any())
    if validation_has_events:
        evals.append((dvalidation, "validation"))
    booster = xgb.train(
        params=linear_cox_params(ridge_lambda, seed),
        dtrain=dtrain,
        num_boost_round=int(CFG.training_rounds),
        evals=evals,
        evals_result=evaluation_history,
        verbose_eval=False,
    )
    train_margin = np.asarray(
        booster.predict(dtrain, output_margin=True), dtype=np.float64
    ).reshape(-1)
    validation_margin = np.asarray(
        booster.predict(dvalidation, output_margin=True), dtype=np.float64
    ).reshape(-1)
    if not np.all(np.isfinite(train_margin)) or not np.all(
        np.isfinite(validation_margin)
    ):
        raise RuntimeError("Linear CoxPH solver produced non-finite margins.")
    baseline_times, baseline_increments, margin_center = (
        breslow_baseline_hazard(train_margin, train_time, train_event)
    )
    train_history = evaluation_history.get("train", {}).get("cox-nloglik", [])
    validation_history = evaluation_history.get("validation", {}).get(
        "cox-nloglik", []
    )
    return {
        "booster_raw": bytes(booster.save_raw(raw_format="ubj")),
        "constant_margin": 0.0,
        "training_rounds": int(CFG.training_rounds),
        "ridge_lambda": float(ridge_lambda),
        "parameters": linear_cox_params(ridge_lambda, seed),
        "training_cindex": float(
            harrell_c_index(train_time, train_event, train_margin)
        ),
        "validation_cindex": float(
            harrell_c_index(validation_time, validation_event, validation_margin)
        ),
        "baseline_times_months": baseline_times,
        "baseline_increments": baseline_increments,
        "margin_center": float(margin_center),
        "training_n": int(len(train_time)),
        "training_events": int(np.asarray(train_event, dtype=bool).sum()),
        "validation_n": int(len(validation_time)),
        "validation_events": int(
            np.asarray(validation_event, dtype=bool).sum()
        ),
        "elapsed_seconds": float(time.time() - started),
        "last_training_cox_nloglik": (
            float(train_history[-1]) if train_history else np.nan
        ),
        "last_validation_cox_nloglik": (
            float(validation_history[-1]) if validation_history else np.nan
        ),
        "solver": "xgboost_gblinear_coord_descent",
        "fit_status": "fitted",
    }


def null_cox_model(
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    reason: str,
) -> Dict[str, Any]:
    train_margin = np.zeros(len(train_time), dtype=np.float64)
    validation_margin = np.zeros(len(validation_time), dtype=np.float64)
    baseline_times, baseline_increments, margin_center = (
        breslow_baseline_hazard(train_margin, train_time, train_event)
    )
    return {
        "booster_raw": None,
        "constant_margin": 0.0,
        "training_rounds": 0,
        "ridge_lambda": np.nan,
        "parameters": {},
        "training_cindex": float(
            harrell_c_index(train_time, train_event, train_margin)
        ),
        "validation_cindex": float(
            harrell_c_index(validation_time, validation_event, validation_margin)
        ),
        "baseline_times_months": baseline_times,
        "baseline_increments": baseline_increments,
        "margin_center": float(margin_center),
        "training_n": int(len(train_time)),
        "training_events": int(np.asarray(train_event, dtype=bool).sum()),
        "validation_n": int(len(validation_time)),
        "validation_events": int(
            np.asarray(validation_event, dtype=bool).sum()
        ),
        "elapsed_seconds": 0.0,
        "last_training_cox_nloglik": np.nan,
        "last_validation_cox_nloglik": np.nan,
        "solver": "null_breslow_only",
        "fit_status": str(reason),
    }


def endpoint_seed(endpoint_name: str) -> int:
    digest = hashlib.sha256(endpoint_name.encode("utf-8")).hexdigest()
    return int((CFG.seed + int(digest[:8], 16)) % (2**31 - 1))


def fit_linear_cox_search(
    train_matrix: sparse.csr_matrix,
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_matrix: sparse.csr_matrix,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    endpoint_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    training_events = int(np.asarray(train_event, dtype=bool).sum())
    validation_events = int(np.asarray(validation_event, dtype=bool).sum())
    if training_events < CFG.min_training_events_to_fit:
        model = null_cox_model(
            train_time,
            train_event,
            validation_time,
            validation_event,
            reason="null_low_training_events",
        )
        return model, [{
            "endpoint": endpoint_name,
            "ridge_lambda": np.nan,
            "training_events": training_events,
            "validation_events": validation_events,
            "validation_cindex": model["validation_cindex"],
            "selected": True,
            "fit_status": model["fit_status"],
        }]

    dtrain = make_dmatrix(train_matrix, train_time, train_event)
    dvalidation = make_dmatrix(
        validation_matrix, validation_time, validation_event
    )
    candidate_lambdas = [float(CFG.default_ridge_lambda)]
    if validation_events >= CFG.min_validation_events_to_select:
        candidate_lambdas.extend([
            float(value)
            for value in CFG.ridge_lambdas
            if float(value) != float(CFG.default_ridge_lambda)
        ])

    seed = endpoint_seed(endpoint_name)
    candidates: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []
    for candidate_index, ridge_lambda in enumerate(candidate_lambdas):
        model = fit_linear_cox_candidate(
            dtrain,
            train_time,
            train_event,
            dvalidation,
            validation_time,
            validation_event,
            ridge_lambda=ridge_lambda,
            seed=seed,
        )
        print(
            "[COX-VALIDATION] "
            f"endpoint={endpoint_name} candidate={candidate_index + 1}/"
            f"{len(candidate_lambdas)} ridge_lambda={ridge_lambda:g} "
            f"rounds={model['training_rounds']} "
            f"train_cindex={model['training_cindex']:.6f} "
            f"val_cindex={model['validation_cindex']:.6f} "
            f"val_cox_nloglik={model['last_validation_cox_nloglik']:.6f} "
            f"elapsed_seconds={model['elapsed_seconds']:.1f}",
            flush=True,
        )
        candidates.append(model)
        trace.append({
            "endpoint": endpoint_name,
            "candidate_index": int(candidate_index),
            "ridge_lambda": float(ridge_lambda),
            "training_events": training_events,
            "validation_events": validation_events,
            "training_cindex": model["training_cindex"],
            "validation_cindex": model["validation_cindex"],
            "training_rounds": model["training_rounds"],
            "elapsed_seconds": model["elapsed_seconds"],
            "selected": False,
            "fit_status": "ridge_candidate",
        })

    selected_index = 0
    if validation_events >= CFG.min_validation_events_to_select:
        best_score = float(candidates[0]["validation_cindex"])
        best_score = best_score if np.isfinite(best_score) else -np.inf
        for candidate_index, candidate in enumerate(candidates[1:], start=1):
            score = float(candidate["validation_cindex"])
            if np.isfinite(score) and score > best_score + 1e-15:
                best_score = score
                selected_index = candidate_index
    selected = candidates[selected_index]
    selected["fit_status"] = (
        "ridge_selected_by_validation_harrell_cindex"
        if validation_events >= CFG.min_validation_events_to_select
        else "default_ridge_low_validation_events"
    )
    selected["selection_metric"] = "validation Harrell C-index"
    trace[selected_index]["selected"] = True
    trace[selected_index]["fit_status"] = selected["fit_status"]
    return selected, trace

# --- Notebook cell 19 ---
def run_survival_utility_self_tests() -> None:
    labels = signed_cox_labels(
        np.asarray([1.0, 2.0]), np.asarray([True, False])
    )
    if not np.allclose(labels, np.asarray([1.0, -2.0])):
        raise AssertionError("Signed Cox-label self-test failed.")

    cindex = harrell_c_index(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([True, True, True]),
        np.asarray([3.0, 2.0, 1.0]),
    )
    if not np.isclose(cindex, 1.0):
        raise AssertionError("Harrell C-index self-test failed.")

    times, increments, center = breslow_baseline_hazard(
        np.zeros(3),
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([True, False, True]),
    )
    if (
        not np.allclose(times, np.asarray([1.0, 3.0]))
        or not np.allclose(increments, np.asarray([1.0 / 3.0, 1.0]))
        or not np.isclose(center, 0.0)
    ):
        raise AssertionError("Centered Breslow self-test failed.")

    # Fail early if the installed XGBoost build cannot run the deterministic
    # sparse gblinear Cox objective used by the publication model.
    matrix = sparse.csr_matrix(np.asarray([
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ], dtype=np.float32))
    event_time = np.asarray([1.0, 2.0, 3.0, 4.0])
    event = np.asarray([True, True, False, True])
    dmatrix = make_dmatrix(matrix, event_time, event)
    booster = xgb.train(
        linear_cox_params(CFG.default_ridge_lambda, CFG.seed),
        dmatrix,
        num_boost_round=3,
        verbose_eval=False,
    )
    margin = np.asarray(
        booster.predict(dmatrix, output_margin=True), dtype=float
    )
    if margin.shape != (4,) or not np.all(np.isfinite(margin)):
        raise AssertionError("gblinear Cox solver smoke test failed.")


run_survival_utility_self_tests()
print("Linear CoxPH numerical and solver self-tests passed.")
