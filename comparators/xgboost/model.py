# XGBoost — Boosted-tree Cox objective, search space, and numerical self-tests
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 16 ---
ROBUST_SEARCH_SPACE: Dict[str, Tuple[Any, ...]] = {
    "max_depth": (2, 3, 4, 5, 6, 8, 10),
    "eta": (0.01, 0.025, 0.05, 0.1, 0.2, 0.3),
    "min_child_weight": (1, 3, 5, 10, 20),
    "subsample": (0.6, 0.75, 0.85, 1.0),
    "colsample_bytree": (0.6, 0.75, 0.85, 1.0),
    "gamma": (0.0, 0.25, 0.5, 1.0, 5.0, 10.0),
    "lambda": (0.0, 1.0, 5.0, 10.0, 20.0),
    "alpha": (0.0, 0.1, 1.0, 5.0, 10.0),
}
V4_BASELINE_PARAMS: Dict[str, Any] = {
    "max_depth": 6,
    "eta": 0.3,
    "min_child_weight": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "gamma": 0.0,
    "lambda": 1.0,
    "alpha": 0.0,
}
REGULARIZED_ANCHOR_PARAMS: Dict[str, Any] = {
    "max_depth": 3,
    "eta": 0.05,
    "min_child_weight": 10,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "gamma": 0.25,
    "lambda": 5.0,
    "alpha": 0.1,
}


def signed_cox_labels(
    exit_time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    time_array = np.asarray(exit_time, dtype=np.float64)
    event_array = np.asarray(event, dtype=bool)
    if time_array.shape != event_array.shape:
        raise ValueError("Survival time and event axes are inconsistent.")
    if not np.all(np.isfinite(time_array)) or np.any(time_array < 0):
        raise ValueError("Survival times must be finite and non-negative.")
    safe_time = np.maximum(time_array, 1e-6)
    return np.where(event_array, safe_time, -safe_time).astype(np.float32)


class CoxMatchedDenseBatchIterator(xgb.DataIter):
    """Convert sparse numeric-zero semantics into bounded dense feature batches."""

    def __init__(self, matrix: sparse.csr_matrix, label: np.ndarray, batch_size: int):
        super().__init__(cache_prefix=None, release_data=True)
        self.matrix = matrix.tocsr()
        self.label = np.asarray(label, dtype=np.float32)
        self.batch_size = int(batch_size)
        self.position = 0
        self.batch = None
        if self.batch_size < 1 or self.label.shape != (self.matrix.shape[0],):
            raise ValueError("Invalid dense iterator batch size or label axis.")

    def reset(self):
        self.position = 0
        self.batch = None

    def next(self, input_data):
        if self.position >= self.matrix.shape[0]:
            return False
        stop = min(self.position + self.batch_size, self.matrix.shape[0])
        self.batch = np.ascontiguousarray(
            self.matrix[self.position:stop].toarray(), dtype=np.float32
        )
        if not np.isfinite(self.batch).all():
            raise ValueError("Non-finite canonical features in dense adapter.")
        input_data(data=self.batch, label=self.label[self.position:stop])
        self.position = stop
        return True


def make_dmatrix(
    matrix: sparse.csr_matrix,
    exit_time: Optional[np.ndarray] = None,
    event: Optional[np.ndarray] = None,
    reference: Optional[xgb.DMatrix] = None,
) -> xgb.DMatrix:
    """All unstored canonical summaries mean numeric zero, matching gblinear."""
    label = None
    if exit_time is not None or event is not None:
        if exit_time is None or event is None:
            raise ValueError("Both time and event are required for labels.")
        label = signed_cox_labels(exit_time, event)
        if label.shape != (matrix.shape[0],):
            raise ValueError("DMatrix feature and label axes differ.")
    if label is not None and matrix.shape[0] > 0:
        iterator = CoxMatchedDenseBatchIterator(matrix, label, CFG.dmatrix_batch_size)
        return xgb.QuantileDMatrix(
            iterator, missing=np.nan, nthread=CFG.nthread,
            max_bin=CFG.max_bin, ref=reference,
        )
    # Unlabelled calls are prediction batches, not whole-cohort conversions.
    if matrix.shape[0] > max(CFG.prediction_batch_size, CFG.dmatrix_batch_size):
        raise ValueError("Unlabelled DMatrix must be constructed in bounded prediction batches.")
    return xgb.DMatrix(
        np.ascontiguousarray(matrix.toarray(), dtype=np.float32),
        label=label, missing=np.nan, nthread=CFG.nthread,
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


def base_xgb_params(seed: int) -> Dict[str, Any]:
    return {
        "booster": "gbtree",
        "max_bin": int(CFG.max_bin),
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "tree_method": CFG.tree_method,
        "device": CFG.device,
        "seed": int(seed),
        "nthread": int(CFG.nthread),
        "verbosity": 0,
    }


def booster_from_raw(raw: Optional[bytes]) -> Optional[xgb.Booster]:
    if raw is None:
        return None
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    return booster


def predict_margin(
    model: Mapping[str, Any], matrix: sparse.csr_matrix,
) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([
        predict_margin_from_dmatrix(model, make_dmatrix(matrix[start:start + CFG.prediction_batch_size]),
                                    min(CFG.prediction_batch_size, matrix.shape[0] - start))
        for start in range(0, matrix.shape[0], CFG.prediction_batch_size)
    ])


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
    best_iteration = int(model["best_iteration"])
    margin = booster.predict(
        dmatrix,
        output_margin=True,
        iteration_range=(0, best_iteration + 1),
    )
    margin = np.asarray(margin, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(margin)):
        raise RuntimeError("XGBoost produced non-finite margins.")
    return margin


def fit_xgb_candidate(
    train_matrix: sparse.csr_matrix,
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_matrix: sparse.csr_matrix,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    parameters: Mapping[str, Any],
    seed: int,
    early_stopping_rounds: Optional[int] = None,
    training_dmatrix: Optional[xgb.DMatrix] = None,
    validation_dmatrix: Optional[xgb.DMatrix] = None,
) -> Dict[str, Any]:
    started = time.time()
    dtrain = training_dmatrix if training_dmatrix is not None else make_dmatrix(train_matrix, train_time, train_event)
    dvalidation = validation_dmatrix if validation_dmatrix is not None else make_dmatrix(
        validation_matrix, validation_time, validation_event, reference=dtrain
    )
    params = {**base_xgb_params(seed), **dict(parameters)}
    evals = [(dtrain, "train")]
    use_validation_early_stop = int(
        np.asarray(validation_event, dtype=bool).sum()
    ) > 0
    if use_validation_early_stop:
        evals.append((dvalidation, "validation"))
    patience = int(
        CFG.early_stopping_rounds
        if early_stopping_rounds is None else early_stopping_rounds
    )
    if patience < 1:
        raise ValueError("Candidate early-stopping patience must be positive.")
    evaluation_history: Dict[str, Dict[str, List[float]]] = {}
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=CFG.max_boost_rounds,
        evals=evals,
        early_stopping_rounds=(
            patience if use_validation_early_stop else None
        ),
        evals_result=evaluation_history,
        verbose_eval=int(CFG.validation_print_interval),
    )
    best_iteration = int(
        getattr(booster, "best_iteration", booster.num_boosted_rounds() - 1)
    )
    iteration_range = (0, best_iteration + 1)
    train_margin = np.asarray(
        booster.predict(
            dtrain, output_margin=True, iteration_range=iteration_range
        ),
        dtype=np.float64,
    )
    validation_margin = np.asarray(
        booster.predict(
            dvalidation, output_margin=True,
            iteration_range=iteration_range,
        ),
        dtype=np.float64,
    )
    baseline_times, baseline_increments, margin_center = breslow_baseline_hazard(
        train_margin, train_time, train_event
    )
    validation_cindex = harrell_c_index(
        validation_time, validation_event, validation_margin
    )
    training_cindex = harrell_c_index(
        train_time, train_event, train_margin
    )
    selected_booster = booster[: best_iteration + 1]
    return {
        "booster_raw": bytes(
            selected_booster.save_raw(raw_format="ubj")
        ),
        "constant_margin": 0.0,
        "best_iteration": best_iteration,
        "boosting_rounds": best_iteration + 1,
        "early_stopping_rounds": patience,
        "parameters": dict(parameters),
        "training_cindex": float(training_cindex),
        "validation_cindex": float(validation_cindex),
        "baseline_times_months": baseline_times,
        "baseline_increments": baseline_increments,
        "margin_center": float(margin_center),
        "training_n": int(train_matrix.shape[0]),
        "training_events": int(np.asarray(train_event, dtype=bool).sum()),
        "validation_n": int(validation_matrix.shape[0]),
        "validation_events": int(
            np.asarray(validation_event, dtype=bool).sum()
        ),
        "elapsed_seconds": float(time.time() - started),
        "last_training_cox_nloglik": float(
            evaluation_history["train"]["cox-nloglik"][
                min(
                    best_iteration,
                    len(
                        evaluation_history["train"]["cox-nloglik"]
                    ) - 1,
                )
            ]
        ),
        "last_validation_cox_nloglik": (
            float(
                evaluation_history["validation"]["cox-nloglik"][
                    min(
                        best_iteration,
                        len(
                            evaluation_history[
                                "validation"
                            ]["cox-nloglik"]
                        ) - 1,
                    )
                ]
            )
            if use_validation_early_stop else np.nan
        ),
    }


def null_xgb_model(
    train_matrix: sparse.csr_matrix,
    train_time: np.ndarray,
    train_event: np.ndarray,
    validation_matrix: sparse.csr_matrix,
    validation_time: np.ndarray,
    validation_event: np.ndarray,
    reason: str,
) -> Dict[str, Any]:
    train_margin = np.zeros(train_matrix.shape[0], dtype=np.float64)
    validation_margin = np.zeros(
        validation_matrix.shape[0], dtype=np.float64
    )
    baseline_times, baseline_increments, margin_center = breslow_baseline_hazard(
        train_margin, train_time, train_event
    )
    return {
        "booster_raw": None,
        "constant_margin": 0.0,
        "best_iteration": -1,
        "boosting_rounds": 0,
        "parameters": {},
        "training_cindex": float(harrell_c_index(
            train_time, train_event, train_margin
        )),
        "validation_cindex": float(harrell_c_index(
            validation_time, validation_event, validation_margin
        )),
        "baseline_times_months": baseline_times,
        "baseline_increments": baseline_increments,
        "margin_center": float(margin_center),
        "training_n": int(train_matrix.shape[0]),
        "training_events": int(np.asarray(train_event, dtype=bool).sum()),
        "validation_n": int(validation_matrix.shape[0]),
        "validation_events": int(
            np.asarray(validation_event, dtype=bool).sum()
        ),
        "elapsed_seconds": 0.0,
        "last_training_cox_nloglik": np.nan,
        "last_validation_cox_nloglik": np.nan,
        "fit_status": str(reason),
        "selected_variant": "null",
        "early_stopping_rounds": 0,
        "required_validation_cindex_gain": np.nan,
        "selected_validation_cindex_gain": np.nan,
        "v4_baseline_validation_cindex": np.nan,
        "paper_default_validation_cindex": np.nan,
    }


def endpoint_seed(endpoint_name: str) -> int:
    digest = hashlib.sha256(endpoint_name.encode("utf-8")).hexdigest()
    return int((CFG.seed + int(digest[:8], 16)) % (2**31 - 1))


def validation_gain_requirement(validation_events: int) -> float:
    return max(
        float(CFG.min_validation_cindex_gain),
        float(CFG.validation_event_gain_scale)
        / math.sqrt(max(int(validation_events), 1)),
    )


def fit_xgb_survival_search(
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
        model = null_xgb_model(
            train_matrix, train_time, train_event,
            validation_matrix, validation_time, validation_event,
            reason="null_low_training_events",
        )
        return model, [{
            "endpoint": endpoint_name,
            "variant": "null",
            "trial_number": -3,
            "training_events": training_events,
            "validation_events": validation_events,
            "validation_cindex": model["validation_cindex"],
            "selected": True,
            "fit_status": model["fit_status"],
        }]

    # Quantize each endpoint's eligible cohort once, then reuse across candidates.
    dtrain = make_dmatrix(train_matrix, train_time, train_event)
    dvalidation = make_dmatrix(validation_matrix, validation_time, validation_event, reference=dtrain)
    seed = endpoint_seed(endpoint_name)
    required_gain = validation_gain_requirement(validation_events)
    trace: List[Dict[str, Any]] = []

    baseline = fit_xgb_candidate(
        train_matrix, train_time, train_event,
        validation_matrix, validation_time, validation_event,
        parameters=V4_BASELINE_PARAMS,
        seed=seed,
        early_stopping_rounds=CFG.baseline_early_stopping_rounds,
        training_dmatrix=dtrain, validation_dmatrix=dvalidation,
    )
    baseline_score = float(baseline["validation_cindex"])
    baseline.update({
        "fit_status": "v4_baseline_benchmark",
        "selected_variant": "v4_baseline",
        "v4_baseline_validation_cindex": baseline_score,
        "paper_default_validation_cindex": baseline_score,
        "regularized_anchor_validation_cindex": np.nan,
        "required_validation_cindex_gain": float(required_gain),
        "selected_validation_cindex_gain": 0.0,
    })
    print(
        "[XGB-VALIDATION] "
        f"endpoint={endpoint_name} variant=v4_baseline trial=-1 "
        f"rounds={baseline['boosting_rounds']} "
        f"train_cindex={baseline['training_cindex']:.6f} "
        f"val_cindex={baseline_score:.6f} "
        f"val_cox_nloglik={baseline['last_validation_cox_nloglik']:.6f} "
        f"elapsed_seconds={baseline['elapsed_seconds']:.1f}",
        flush=True,
    )
    trace.append({
        "endpoint": endpoint_name,
        "variant": "v4_baseline",
        "trial_number": -1,
        "training_events": training_events,
        "validation_events": validation_events,
        "validation_cindex": baseline_score,
        "training_cindex": baseline["training_cindex"],
        "boosting_rounds": baseline["boosting_rounds"],
        "early_stopping_rounds": baseline["early_stopping_rounds"],
        "elapsed_seconds": baseline["elapsed_seconds"],
        "required_validation_cindex_gain": float(required_gain),
        "validation_cindex_gain_vs_v4": 0.0,
        "parameters_json": json.dumps(V4_BASELINE_PARAMS, sort_keys=True),
        "selected": validation_events < CFG.min_validation_events_to_select,
        "fit_status": (
            "v4_baseline_low_validation_events"
            if validation_events < CFG.min_validation_events_to_select
            else "v4_baseline_benchmark"
        ),
    })
    if validation_events < CFG.min_validation_events_to_select:
        baseline["fit_status"] = "v4_baseline_low_validation_events"
        return baseline, trace

    regularized = fit_xgb_candidate(
        train_matrix, train_time, train_event,
        validation_matrix, validation_time, validation_event,
        parameters=REGULARIZED_ANCHOR_PARAMS,
        seed=seed,
        early_stopping_rounds=CFG.early_stopping_rounds,
        training_dmatrix=dtrain, validation_dmatrix=dvalidation,
    )
    regularized_score = float(regularized["validation_cindex"])
    print(
        "[XGB-VALIDATION] "
        f"endpoint={endpoint_name} variant=regularized_anchor trial=-2 "
        f"rounds={regularized['boosting_rounds']} "
        f"train_cindex={regularized['training_cindex']:.6f} "
        f"val_cindex={regularized_score:.6f} "
        f"val_cox_nloglik={regularized['last_validation_cox_nloglik']:.6f} "
        f"elapsed_seconds={regularized['elapsed_seconds']:.1f}",
        flush=True,
    )
    regularized_gain = regularized_score - baseline_score
    trace.append({
        "endpoint": endpoint_name,
        "variant": "regularized_anchor",
        "trial_number": -2,
        "training_events": training_events,
        "validation_events": validation_events,
        "validation_cindex": regularized_score,
        "training_cindex": regularized["training_cindex"],
        "boosting_rounds": regularized["boosting_rounds"],
        "early_stopping_rounds": regularized["early_stopping_rounds"],
        "elapsed_seconds": regularized["elapsed_seconds"],
        "required_validation_cindex_gain": float(required_gain),
        "validation_cindex_gain_vs_v4": float(regularized_gain),
        "parameters_json": json.dumps(
            REGULARIZED_ANCHOR_PARAMS, sort_keys=True
        ),
        "selected": False,
        "fit_status": "regularized_anchor_benchmark",
    })

    baseline_selection_score = (
        baseline_score if np.isfinite(baseline_score) else -np.inf
    )
    best: Dict[str, Any] = {
        "score": baseline_selection_score,
        "trial": -1,
        "variant": "v4_baseline",
        "model": baseline,
    }

    def has_supported_selection_score(score: float) -> bool:
        if not np.isfinite(score):
            return False
        if not np.isfinite(baseline_score):
            return True
        return bool(score >= baseline_score + required_gain)

    if (
        has_supported_selection_score(regularized_score)
        and regularized_score > float(best["score"]) + 1e-15
    ):
        best.update({
            "score": regularized_score,
            "trial": -2,
            "variant": "regularized_anchor",
            "model": regularized,
        })

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        parameters = {
            name: trial.suggest_categorical(name, list(values))
            for name, values in ROBUST_SEARCH_SPACE.items()
        }
        model = fit_xgb_candidate(
            train_matrix, train_time, train_event,
            validation_matrix, validation_time, validation_event,
            parameters=parameters,
            seed=seed,
            early_stopping_rounds=CFG.early_stopping_rounds,
        training_dmatrix=dtrain, validation_dmatrix=dvalidation,
        )
        score = float(model["validation_cindex"])
        gain = score - baseline_score
        print(
            "[XGB-VALIDATION] "
            f"endpoint={endpoint_name} variant=robust_tuned "
            f"trial={trial.number + 1}/{CFG.tuning_trials} "
            f"rounds={model['boosting_rounds']} "
            f"train_cindex={model['training_cindex']:.6f} "
            f"val_cindex={score:.6f} "
            f"gain_vs_v4={gain:.6f} required_gain={required_gain:.6f} "
            f"val_cox_nloglik={model['last_validation_cox_nloglik']:.6f} "
            f"elapsed_seconds={model['elapsed_seconds']:.1f}",
            flush=True,
        )
        trace.append({
            "endpoint": endpoint_name,
            "variant": "robust_tuned",
            "trial_number": int(trial.number),
            "training_events": training_events,
            "validation_events": validation_events,
            "validation_cindex": score,
            "training_cindex": model["training_cindex"],
            "boosting_rounds": model["boosting_rounds"],
            "early_stopping_rounds": model["early_stopping_rounds"],
            "elapsed_seconds": model["elapsed_seconds"],
            "required_validation_cindex_gain": float(required_gain),
            "validation_cindex_gain_vs_v4": float(gain),
            "parameters_json": json.dumps(parameters, sort_keys=True),
            "selected": False,
            "fit_status": "robust_tpe_trial",
        })
        if (
            has_supported_selection_score(score)
            and score > float(best["score"]) + 1e-15
        ):
            best.update({
                "score": score,
                "trial": int(trial.number),
                "variant": "robust_tuned",
                "model": model,
            })
        return score if np.isfinite(score) else 0.5

    study.optimize(
        objective,
        n_trials=CFG.tuning_trials,
        n_jobs=1,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    selected = best["model"]
    selected["selection_metric"] = "validation Harrell C-index"
    selected_score = float(selected["validation_cindex"])
    selected_gain = selected_score - baseline_score
    selected.update({
        "selected_variant": str(best["variant"]),
        "v4_baseline_validation_cindex": baseline_score,
        "paper_default_validation_cindex": baseline_score,
        "regularized_anchor_validation_cindex": regularized_score,
        "required_validation_cindex_gain": float(required_gain),
        "selected_validation_cindex_gain": float(selected_gain),
    })
    if best["variant"] == "v4_baseline":
        selected["fit_status"] = "default_tree_selected_by_validation_harrell_cindex"
    elif best["variant"] == "regularized_anchor":
        selected["fit_status"] = "regularized_anchor_selected_over_v4"
    else:
        selected["fit_status"] = "robust_tpe_selected_over_v4"
        selected["optuna_best_trial"] = int(best["trial"])

    if CFG.retain_default_variant and best["variant"] != "v4_baseline":
        selected["paper_default_model"] = baseline

    for row in trace:
        is_selected = (
            (best["variant"] == "v4_baseline" and row["trial_number"] == -1)
            or (
                best["variant"] == "regularized_anchor"
                and row["trial_number"] == -2
            )
            or (
                best["variant"] == "robust_tuned"
                and row["variant"] == "robust_tuned"
                and int(row["trial_number"]) == int(best["trial"])
            )
        )
        if is_selected:
            row["selected"] = True
            row["fit_status"] = selected["fit_status"]

    if (
        np.isfinite(baseline_score)
        and selected_score < baseline_score - 1e-12
    ):
        raise AssertionError("Non-regression selection chose below the V4 anchor.")
    return selected, trace

# --- Notebook cell 18 ---
def run_survival_utility_self_tests() -> None:
    labels = signed_cox_labels(
        np.asarray([2.0, 3.0, 4.0]),
        np.asarray([True, False, True]),
    )
    if not np.array_equal(labels > 0, [True, False, True]):
        raise AssertionError("XGBoost censor-label encoding failed.")
    cindex = harrell_c_index(
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([True, True, True, False]),
        np.asarray([4.0, 3.0, 2.0, 1.0]),
    )
    if not np.isclose(cindex, 1.0):
        raise AssertionError("Harrell C-index direction test failed.")
    times, increments, margin_center = breslow_baseline_hazard(
        np.zeros(4),
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        np.asarray([True, True, False, False]),
    )
    if (
        not np.array_equal(times, np.asarray([1.0, 2.0]))
        or not np.allclose(increments, np.asarray([0.25, 1.0 / 3.0]))
    ):
        raise AssertionError("Breslow baseline test failed.")
    for shift in (0.0, 40.0, -40.0):
        _, shifted_increments, center = breslow_baseline_hazard(
            np.full(2, shift), np.full(2, 6.0), np.ones(2, dtype=bool)
        )
        risk = -np.expm1(-shifted_increments.sum() * np.exp(np.clip(shift - center, -30.0, 30.0)))
        if not np.isclose(risk, -np.expm1(-1.0)):
            raise AssertionError("Centered absolute-risk shift-invariance test failed.")
    test_features = sparse.csr_matrix(np.asarray([[0., 1.], [1., 0.], [0., 0.]], dtype=np.float32))
    test_matrix = make_dmatrix(test_features, np.asarray([1., 2., 3.]), np.asarray([True, True, False]))
    if int(test_matrix.num_nonmissing()) != 6:
        raise AssertionError("Dense adapter converted canonical zeros to missing values.")
    test_validation = make_dmatrix(test_features, np.asarray([1., 2., 3.]),
                                   np.asarray([True, True, False]), reference=test_matrix)
    test_booster = xgb.train({**base_xgb_params(CFG.seed), "max_depth": 1, "eta": 0.1},
                             test_matrix, num_boost_round=3, verbose_eval=False)
    if not np.isfinite(test_booster.predict(test_validation, output_margin=True)).all():
        raise AssertionError("Quantile Cox tree solver smoke test failed.")
    print({
        "survival_utility_self_tests": "passed",
        "perfect_order_cindex": cindex,
    })


run_survival_utility_self_tests()
