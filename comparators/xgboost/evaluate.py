# XGBoost — Defines test prediction, competing-risk metrics, tables, and participant bootstrap
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 22 ---
def load_canonical_test_samples() -> pd.DataFrame:
    if FROZEN_BUNDLE is None:
        raise RuntimeError(
            "A frozen bundle must be loaded before opening the test split."
        )
    require_files([CANONICAL_PATHS["test"]])
    observed = sha256_file(CANONICAL_PATHS["test"])
    expected = EXPECTED_HASHES.get("canonical_test_samples_sha256")
    if not expected or observed != expected:
        raise RuntimeError(
            "Canonical test pickle is missing from or differs from the CRESCENT manifest."
        )
    frame = pd.read_pickle(CANONICAL_PATHS["test"]).copy()
    verify_crescent_provenance("test", frame)
    frame["split"] = "test"
    validate_canonical_samples(frame, "test")
    compare_with_main_index_audit(frame, "test")

    train_ids = set(TRAIN_SAMPLES["eid"].astype(str))
    validation_ids = set(VAL_SAMPLES["eid"].astype(str))
    test_ids = set(frame["eid"].astype(str))
    if test_ids & train_ids or test_ids & validation_ids:
        raise RuntimeError("Participant leakage into canonical test split.")
    if FROZEN_BUNDLE["test_used_for_model_selection"]:
        raise RuntimeError("Frozen bundle reports test-set model selection.")
    if (
        FROZEN_BUNDLE["canonical_train_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["train"])
        or FROZEN_BUNDLE["canonical_val_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["val"])
    ):
        raise RuntimeError(
            "Frozen bundle was fitted from different canonical samples."
        )
    return frame.reset_index(drop=True)


def prepare_test_context(
    bundle: Mapping[str, Any],
) -> pd.DataFrame:
    """Set the frozen evaluation globals without fitting any model."""
    global FROZEN_BUNDLE
    global TEST_SAMPLES, TEST_HISTORY_AUDIT
    global TEST_FEATURE_BUILDER, X_TEST

    FROZEN_BUNDLE = dict(bundle)
    TEST_SAMPLES = load_canonical_test_samples()
    TEST_HISTORY_AUDIT = pd.DataFrame([
        history_length_audit(TEST_SAMPLES, "test")
    ])
    pd.concat(
        [HISTORY_AUDIT_TRAIN_VAL, TEST_HISTORY_AUDIT],
        ignore_index=True,
    ).to_csv(
        RUN_DIR / "full_history_audit_all_splits.csv",
        index=False,
    )

    TEST_FEATURE_BUILDER = FROZEN_BUNDLE["feature_builder"]
    x_test_raw = TEST_FEATURE_BUILDER.transform(TEST_SAMPLES)
    X_TEST = apply_frozen_feature_contract(x_test_raw)
    del x_test_raw
    if X_TEST.shape[1] != len(FROZEN_BUNDLE["feature_names"]):
        raise RuntimeError(
            "Test feature axis differs from the frozen model."
        )
    if not np.isfinite(X_TEST.data).all():
        raise ValueError(
            "Non-finite stored test XGBoost features."
        )
    print({
        "canonical_test_rows": len(TEST_SAMPLES),
        "test_feature_shape": X_TEST.shape,
        "test_history_audit": TEST_HISTORY_AUDIT.iloc[0].to_dict(),
        "frozen_bundle_sha256": sha256_file(MODEL_BUNDLE_PATH),
    })
    return TEST_SAMPLES

# --- Notebook cell 24 ---
PREDICTION_PATHS = {
    "adjusted": RUN_DIR / "test_risk_death_adjusted_dense.npy",
    "unadjusted": RUN_DIR / "test_risk_unadjusted_dense.npy",
    "death": RUN_DIR / "test_risk_death_dense.npy",
    "raw_condition_rate_gap": RUN_DIR
    / "test_condition_rate_at_gap.npy",
    "raw_death_rate_gap": RUN_DIR / "test_death_rate_at_gap.npy",
    "raw_rate_summary": RUN_DIR
    / "test_raw_landmark_rate_summary.csv",
    "raw_rate_auc": RUN_DIR
    / "test_raw_landmark_rate_auc_by_horizon.csv",
    "sequence_summary": RUN_DIR
    / "test_full_history_sequence_summary.csv",
    "manifest": RUN_DIR / "test_prediction_manifest.json",
    "row_key": RUN_DIR / "test_prediction_row_key.csv",
}


def baseline_increments_to_canonical_bins(
    model: Mapping[str, Any],
) -> np.ndarray:
    n_future_bins = (
        max(CFG.report_horizons_months) // BIN_WIDTH_MONTHS
    )
    result = np.zeros(n_future_bins, dtype=np.float64)
    times = np.asarray(
        model["baseline_times_months"], dtype=np.float64
    )
    increments = np.asarray(
        model["baseline_increments"], dtype=np.float64
    )
    if times.size != increments.size:
        raise ValueError("Baseline time/increment arrays are misaligned.")
    valid = (
        np.isfinite(times)
        & np.isfinite(increments)
        & (times >= 0.0)
        & (times <= max(CFG.report_horizons_months) + 1e-8)
        & (increments >= 0.0)
    )
    times = times[valid]
    increments = increments[valid]
    if times.size:
        bin_ends = np.arange(
            BIN_WIDTH_MONTHS,
            max(CFG.report_horizons_months) + BIN_WIDTH_MONTHS,
            BIN_WIDTH_MONTHS,
            dtype=np.float64,
        )
        bin_index = np.searchsorted(
            bin_ends, times, side="left"
        )
        bin_index = np.clip(bin_index, 0, n_future_bins - 1)
        np.add.at(result, bin_index, increments)
    return result


def piecewise_competing_risk_curves(
    condition_multiplier: np.ndarray,
    death_multiplier: np.ndarray,
    condition_baseline_bins: np.ndarray,
    death_baseline_bins: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    condition_multiplier = np.asarray(
        condition_multiplier, dtype=np.float64
    ).reshape(-1)
    death_multiplier = np.asarray(death_multiplier, dtype=np.float64).reshape(-1)
    n = len(condition_multiplier)
    grid_size = len(DENSE_HORIZONS_MONTHS)
    adjusted = np.zeros((n, grid_size), dtype=np.float64)
    unadjusted = np.zeros((n, grid_size), dtype=np.float64)
    death = np.zeros((n, grid_size), dtype=np.float64)

    survival = np.ones(n, dtype=np.float64)
    cumulative_condition = np.zeros(n, dtype=np.float64)
    cumulative_condition_hazard = np.zeros(n, dtype=np.float64)
    cumulative_death_hazard = np.zeros(n, dtype=np.float64)
    dense_lookup = {
        int(horizon): index
        for index, horizon in enumerate(DENSE_HORIZONS_MONTHS)
    }

    for bin_index in range(len(condition_baseline_bins)):
        condition_increment = (
            condition_multiplier * float(condition_baseline_bins[bin_index])
        )
        death_increment = death_multiplier * float(death_baseline_bins[bin_index])
        total_increment = condition_increment + death_increment
        failure = -np.expm1(-np.clip(total_increment, 0.0, 700.0))
        condition_probability = np.divide(
            condition_increment * failure,
            total_increment,
            out=np.zeros_like(total_increment),
            where=total_increment > 0,
        )
        cumulative_condition += survival * condition_probability
        survival *= np.exp(-np.clip(total_increment, 0.0, 700.0))
        cumulative_condition_hazard += condition_increment
        cumulative_death_hazard += death_increment

        horizon = (bin_index + 1) * BIN_WIDTH_MONTHS
        if horizon in dense_lookup:
            output_index = dense_lookup[horizon]
            adjusted[:, output_index] = cumulative_condition
            unadjusted[:, output_index] = -np.expm1(
                -np.clip(cumulative_condition_hazard, 0.0, 700.0)
            )
            death[:, output_index] = -np.expm1(
                -np.clip(cumulative_death_hazard, 0.0, 700.0)
            )

    return (
        np.clip(adjusted, 0.0, 1.0),
        np.clip(unadjusted, 0.0, 1.0),
        np.clip(death, 0.0, 1.0),
    )


def prediction_files_complete() -> bool:
    required = [
        PREDICTION_PATHS["adjusted"],
        PREDICTION_PATHS["death"],
        PREDICTION_PATHS["raw_condition_rate_gap"],
        PREDICTION_PATHS["raw_death_rate_gap"],
        PREDICTION_PATHS["raw_rate_summary"],
        PREDICTION_PATHS["sequence_summary"],
        PREDICTION_PATHS["manifest"],
        PREDICTION_PATHS["row_key"],
    ]
    if CFG.export_unadjusted_comparison:
        required.append(PREDICTION_PATHS["unadjusted"])
    return all(path.is_file() for path in required)


def load_saved_predictions() -> Dict[str, Any]:
    if not prediction_files_complete():
        raise FileNotFoundError(
            "Saved XGBoost survival prediction files are incomplete."
        )
    with PREDICTION_PATHS["manifest"].open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    checks = {
        "test_eid_axis_sha256": canonical_json_hash(
            TEST_SAMPLES["eid"].astype(str).tolist()
        ),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "canonical_test_samples_sha256": sha256_file(
            CANONICAL_PATHS["test"]
        ),
        "model_bundle_sha256": sha256_file(MODEL_BUNDLE_PATH),
    }
    for key, expected in checks.items():
        if manifest.get(key) != expected:
            raise RuntimeError(
                f"Saved prediction manifest mismatch for {key}."
            )
    if tuple(manifest.get("dense_horizons_months", [])) != DENSE_HORIZONS_MONTHS:
        raise RuntimeError("Saved prediction dense horizon grid differs.")
    if tuple(manifest.get("report_horizons_months", [])) != tuple(
        CFG.report_horizons_months
    ):
        raise RuntimeError("Saved prediction report horizons differ.")
    if int(manifest.get("gap_months", -1)) != int(CFG.gap_months):
        raise RuntimeError("Saved prediction lead-time gap differs.")
    if manifest.get("history_contract") != "all canonical pre-index events":
        raise RuntimeError("Saved prediction full-history contract differs.")
    if manifest.get("feature_mask_sha256") != sha256_array(
        FEATURE_MASK.astype(np.uint8)
    ):
        raise RuntimeError("Saved prediction feature-filter contract differs.")

    predictions: Dict[str, Any] = {
        "adjusted": np.load(PREDICTION_PATHS["adjusted"], mmap_mode="r"),
        "death": np.load(PREDICTION_PATHS["death"], mmap_mode="r"),
        "raw_condition_rate_gap": np.load(
            PREDICTION_PATHS["raw_condition_rate_gap"], mmap_mode="r"
        ),
        "raw_death_rate_gap": np.load(
            PREDICTION_PATHS["raw_death_rate_gap"], mmap_mode="r"
        ),
        "manifest": manifest,
    }
    predictions["unadjusted"] = (
        np.load(PREDICTION_PATHS["unadjusted"], mmap_mode="r")
        if PREDICTION_PATHS["unadjusted"].is_file()
        else None
    )
    expected_shape = (
        len(TEST_SAMPLES), K, len(DENSE_HORIZONS_MONTHS)
    )
    if predictions["adjusted"].shape != expected_shape:
        raise RuntimeError(
            "Adjusted XGBoost survival prediction shape is incorrect."
        )
    if predictions["death"].shape != (
        len(TEST_SAMPLES), len(DENSE_HORIZONS_MONTHS)
    ):
        raise RuntimeError(
            "Death XGBoost survival prediction shape is incorrect."
        )
    return predictions


def generate_predictions() -> Dict[str, Any]:
    if prediction_files_complete() and not CFG.overwrite_predictions:
        print(
            "XGBoost survival predictions exist; loading them. Set "
            "EHR_OVERWRITE_PREDICTIONS=YES to regenerate."
        )
        return load_saved_predictions()

    n = len(TEST_SAMPLES)
    grid_size = len(DENSE_HORIZONS_MONTHS)
    adjusted = np.lib.format.open_memmap(
        PREDICTION_PATHS["adjusted"],
        mode="w+", dtype=np.float32, shape=(n, K, grid_size),
    )
    unadjusted = None
    if CFG.export_unadjusted_comparison:
        unadjusted = np.lib.format.open_memmap(
            PREDICTION_PATHS["unadjusted"],
            mode="w+", dtype=np.float32, shape=(n, K, grid_size),
        )
    death = np.lib.format.open_memmap(
        PREDICTION_PATHS["death"],
        mode="w+", dtype=np.float32, shape=(n, grid_size),
    )
    raw_condition_rate = np.lib.format.open_memmap(
        PREDICTION_PATHS["raw_condition_rate_gap"],
        mode="w+", dtype=np.float32, shape=(n, K),
    )
    raw_death_rate = np.lib.format.open_memmap(
        PREDICTION_PATHS["raw_death_rate_gap"],
        mode="w+", dtype=np.float32, shape=(n,),
    )

    condition_models = FROZEN_BUNDLE["condition_models"]
    death_model = FROZEN_BUNDLE["death_model"]
    death_baseline_bins = baseline_increments_to_canonical_bins(
        death_model
    )
    condition_baseline_bins = [
        baseline_increments_to_canonical_bins(model)
        for model in condition_models
    ]
    # Predictions are retained only for participants condition-free, alive,
    # and observed at the six-month landmark. This is the same eligibility
    # transformation used by training, evaluation, and the frozen KM estimators.
    at_risk = make_condition_outcomes(TEST_SAMPLES)["eligible"]

    for start in range(0, n, CFG.prediction_batch_size):
        stop = min(start + CFG.prediction_batch_size, n)
        matrix = X_TEST[start:stop]
        dmatrix = make_dmatrix(matrix)
        death_eta = predict_margin_from_dmatrix(
            death_model, dmatrix, stop - start
        )
        death_multiplier = np.exp(np.clip(
            death_eta - float(death_model["margin_center"]), -30.0, 30.0
        ))
        raw_death_rate[start:stop] = (
            death_multiplier
            * death_baseline_bins[0]
            / float(BIN_WIDTH_MONTHS)
        ).astype(np.float32)

        # The death curve does not depend on condition and is stored once.
        zero_condition = np.zeros_like(death_baseline_bins)
        _, _, death_curve = piecewise_competing_risk_curves(
            np.ones(stop - start),
            death_multiplier,
            zero_condition,
            death_baseline_bins,
        )
        death[start:stop] = death_curve.astype(np.float32)

        for condition, model in enumerate(condition_models):
            condition_eta = predict_margin_from_dmatrix(
                model, dmatrix, stop - start
            )
            condition_multiplier = np.exp(
                np.clip(condition_eta - float(model["margin_center"]), -30.0, 30.0)
            )
            adjusted_curve, unadjusted_curve, _ = (
                piecewise_competing_risk_curves(
                    condition_multiplier,
                    death_multiplier,
                    condition_baseline_bins[condition],
                    death_baseline_bins,
                )
            )
            eligible_at_index = at_risk[start:stop, condition]
            adjusted_curve[~eligible_at_index] = 0.0
            unadjusted_curve[~eligible_at_index] = 0.0
            adjusted[start:stop, condition] = adjusted_curve.astype(
                np.float32
            )
            if unadjusted is not None:
                unadjusted[start:stop, condition] = (
                    unadjusted_curve.astype(np.float32)
                )
            rate = (
                condition_multiplier
                * condition_baseline_bins[condition][0]
                / float(BIN_WIDTH_MONTHS)
            )
            rate[~eligible_at_index] = 0.0
            raw_condition_rate[start:stop, condition] = rate.astype(
                np.float32
            )
        print(f"Predicted rows {start:,}:{stop:,} of {n:,}")

    adjusted.flush()
    death.flush()
    raw_condition_rate.flush()
    raw_death_rate.flush()
    if unadjusted is not None:
        unadjusted.flush()

    for name, array in (
        ("adjusted", adjusted),
        ("death", death),
        ("unadjusted", unadjusted),
    ):
        if array is None:
            continue
        if (
            not np.all(np.isfinite(array))
            or float(np.min(array)) < -1e-6
            or float(np.max(array)) > 1.0 + 1e-6
            or np.any(np.diff(array, axis=-1) < -2e-6)
        ):
            raise ValueError(f"Invalid or non-monotone {name} predictions.")

    pd.DataFrame({
        "row_index": np.arange(n, dtype=int),
        "eid": TEST_SAMPLES["eid"].astype(str),
        "index_dt": TEST_SAMPLES["index_dt"].astype(str),
        "index_age_years": TEST_SAMPLES["index_age_years"].astype(float),
        "index_offset_days_from_recruitment": TEST_SAMPLES[
            "index_offset_days_from_recruitment"
        ].astype(int),
    }).to_csv(PREDICTION_PATHS["row_key"], index=False)

    summary_rows = []
    for condition, condition_name in enumerate(CONDITION_NAMES):
        values = np.asarray(raw_condition_rate[:, condition], dtype=float)
        finite = values[np.isfinite(values)]
        summary_rows.append({
            "model_name": CFG.model_name,
            "score_type": "first_post_gap_bin_average_condition_rate",
            "condition_id": int(condition),
            "condition_name": condition_name,
            "eligible_n": int(at_risk[:, condition].sum()),
            "finite_n": int(finite.size),
            "rate_mean_per_month": (
                float(finite.mean()) if finite.size else np.nan
            ),
            "rate_sd_per_month": (
                float(finite.std(ddof=0)) if finite.size else np.nan
            ),
            "rate_median_per_month": (
                float(np.median(finite)) if finite.size else np.nan
            ),
            "approximately_unique_n": int(
                np.unique(np.round(finite, decimals=10)).size
            ) if finite.size else 0,
        })
    pd.DataFrame(summary_rows).to_csv(
        PREDICTION_PATHS["raw_rate_summary"], index=False
    )
    history_lengths = TEST_SAMPLES["history_event_ids"].map(len).to_numpy(int)
    pd.DataFrame({
        "eid": TEST_SAMPLES["eid"].astype(str),
        "available_pre_index_events": history_lengths,
        "events_used": history_lengths,
        "events_truncated": np.zeros_like(history_lengths),
        "prevalent_conditions": [
            int(np.sum(np.asarray(value, dtype=float) <= 0.5))
            for value in TEST_SAMPLES["at_risk"]
        ],
    }).to_csv(PREDICTION_PATHS["sequence_summary"], index=False)

    manifest = {
        "schema_version": CFG.schema_version,
        "model_name": CFG.model_name,
        "model_bundle_path": str(MODEL_BUNDLE_PATH),
        "model_bundle_sha256": sha256_file(MODEL_BUNDLE_PATH),
        "canonical_test_samples_sha256": sha256_file(
            CANONICAL_PATHS["test"]
        ),
        "test_eid_axis_sha256": canonical_json_hash(
            TEST_SAMPLES["eid"].astype(str).tolist()
        ),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "raw_feature_axis_sha256": FROZEN_BUNDLE[
            "raw_feature_axis_sha256"
        ],
        "feature_axis_sha256": FROZEN_BUNDLE["feature_axis_sha256"],
        "feature_mask_sha256": FROZEN_BUNDLE["feature_mask_sha256"],
        "test_rows": n,
        "conditions": K,
        "history_contract": "all canonical pre-index events",
        "events_discarded": 0,
        "gap_months": CFG.gap_months,
        "dense_horizons_months": list(DENSE_HORIZONS_MONTHS),
        "report_horizons_months": list(CFG.report_horizons_months),
        "xgboost_booster": "gbtree",
        "feature_scale_sha256": FROZEN_BUNDLE["feature_scale_sha256"],
        "evaluator_contract": FROZEN_BUNDLE["evaluator_contract"],
        "xgboost_objective": "survival:cox",
        "xgboost_prediction_scale": (
            "exp(clip(raw_margin - training_margin_center, -30, 30)); Cox V6 convention"
        ),
        "baseline_hazard": "training-only centered Breslow increments",
        "cif_integration": (
            "Breslow baseline increments aggregated to canonical six-month "
            "bins and combined with the exact symmetric "
            "piecewise-exponential condition/death formula"
        ),
        "primary_prediction": (
            "death-adjusted condition CIF from the canonical index with a "
            "six-month lead-time gap"
        ),
        "adjusted_path": str(PREDICTION_PATHS["adjusted"]),
        "unadjusted_path": (
            str(PREDICTION_PATHS["unadjusted"])
            if unadjusted is not None else None
        ),
        "death_path": str(PREDICTION_PATHS["death"]),
        "full_history_sequence_summary_path": str(
            PREDICTION_PATHS["sequence_summary"]
        ),
        "test_used_for_model_selection": False,
    }
    atomic_json_dump(manifest, PREDICTION_PATHS["manifest"])
    return load_saved_predictions()

# --- Notebook cell 26 ---
class CensoringKM:
    def __init__(self, exit_time: np.ndarray, censor_event: np.ndarray):
        time_array = np.asarray(exit_time, dtype=float)
        event_array = np.asarray(censor_event, dtype=bool)
        valid = np.isfinite(time_array) & (time_array >= 0)
        time_array = time_array[valid]
        event_array = event_array[valid]
        if time_array.size == 0:
            self.times = np.asarray([], dtype=float)
            self.survival_after = np.asarray([], dtype=float)
            return
        order = np.argsort(time_array, kind="mergesort")
        time_array = time_array[order]
        event_array = event_array[order].astype(np.int64)
        unique, first = np.unique(time_array, return_index=True)
        at_risk = time_array.size - first
        censor_counts = np.add.reduceat(event_array, first)
        factors = 1.0 - censor_counts / np.maximum(at_risk, 1)
        self.times = unique
        self.survival_after = np.cumprod(
            np.clip(factors, 0.0, 1.0)
        )

    def _query(
        self, query: np.ndarray | float, side: str, clip: bool
    ) -> np.ndarray:
        query_array = np.asarray(query, dtype=float)
        indices = (
            np.searchsorted(self.times, query_array, side=side) - 1
        )
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


def binary_window(
    eligible: np.ndarray,
    status: np.ndarray,
    exit_time: np.ndarray,
    horizon: float,
    km: CensoringKM,
) -> Dict[str, Any]:
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=float)
    case = (
        eligible & (status == 1)
        & (exit_time <= horizon + 1e-8)
    )
    competing = (
        eligible & (status == 2)
        & (exit_time <= horizon + 1e-8)
    )
    observed_through = (
        eligible & (exit_time >= horizon - 1e-8)
    )
    known = case | competing | observed_through
    y = case.astype(float)
    weights = np.zeros_like(exit_time, dtype=float)
    event_or_competing = case | competing
    weights[event_or_competing] = (
        1.0 / km.before(exit_time[event_or_competing])
    )
    weights[observed_through & ~event_or_competing] = (
        1.0 / float(km.before(horizon))
    )

    raw_g = np.ones_like(exit_time, dtype=float)
    raw_g[event_or_competing] = km.before_raw(
        exit_time[event_or_competing]
    )
    raw_g[observed_through & ~event_or_competing] = float(
        km.before_raw(horizon)
    )
    truncated = known & (raw_g < CFG.censor_survival_floor)
    return {
        "event_free_control": observed_through & ~case & ~competing,
        "eligible": eligible,
        "case": case,
        "competing": competing,
        "known": known,
        "observed_through": observed_through,
        "y": y,
        "weights": weights,
        "truncated": truncated,
        "raw_g": raw_g,
    }


def aalen_johansen_cif(
    exit_time: np.ndarray,
    status: np.ndarray,
    horizon: float,
) -> float:
    time_array = np.asarray(exit_time, dtype=float)
    status_array = np.asarray(status, dtype=np.int8)
    valid = np.isfinite(time_array) & (time_array >= 0)
    time_array = time_array[valid]
    status_array = status_array[valid]
    if time_array.size == 0:
        return np.nan
    order = np.argsort(time_array, kind="mergesort")
    time_array = time_array[order]
    status_array = status_array[order]
    unique, first = np.unique(time_array, return_index=True)
    at_risk = time_array.size - first
    primary = np.add.reduceat(
        (status_array == 1).astype(np.int64), first
    )
    all_events = np.add.reduceat(
        (status_array != 0).astype(np.int64), first
    )
    event_mask = (
        (unique <= horizon + 1e-8) & (all_events > 0)
    )
    if not np.any(event_mask):
        return 0.0
    hazard_all = all_events / np.maximum(at_risk, 1)
    survival_after = np.cumprod(
        np.clip(1.0 - hazard_all, 0.0, 1.0)
    )
    survival_before = np.concatenate(
        ([1.0], survival_after[:-1])
    )
    increments = (
        survival_before * primary / np.maximum(at_risk, 1)
    )
    return float(np.sum(increments[event_mask]))


def ipcw_weight_diagnostics(
    window: Mapping[str, np.ndarray]
) -> Dict[str, float]:
    known = np.asarray(window["known"], dtype=bool)
    weights = np.asarray(window["weights"], dtype=float)
    valid_weights = weights[
        known & np.isfinite(weights) & (weights > 0)
    ]
    if valid_weights.size == 0:
        return {
            "ipcw_weight_sum": 0.0,
            "ipcw_weight_mean": np.nan,
            "ipcw_weight_max": np.nan,
            "ipcw_effective_sample_size": 0.0,
            "ipcw_truncated_n": int(
                np.asarray(window["truncated"], dtype=bool).sum()
            ),
            "ipcw_truncated_fraction": np.nan,
        }
    weight_sum = float(valid_weights.sum())
    effective_n = (
        weight_sum ** 2 / float(np.sum(valid_weights ** 2))
    )
    truncated_n = int(
        np.asarray(window["truncated"], dtype=bool).sum()
    )
    return {
        "ipcw_weight_sum": weight_sum,
        "ipcw_weight_mean": float(valid_weights.mean()),
        "ipcw_weight_max": float(valid_weights.max()),
        "ipcw_effective_sample_size": float(effective_n),
        "ipcw_truncated_n": truncated_n,
        "ipcw_truncated_fraction": float(
            truncated_n / max(int(known.sum()), 1)
        ),
    }


def ipcw_ht_scores(
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    eligible_n: int,
) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.clip(
        np.asarray(prediction, dtype=float), 1e-7, 1.0 - 1e-7
    )
    weights = np.asarray(weights, dtype=float)
    valid = (
        np.isfinite(y) & np.isfinite(prediction)
        & np.isfinite(weights) & (weights > 0)
    )
    if eligible_n <= 0 or not valid.any():
        return {
            "brier_ipcw_ht": np.nan,
            "nbll_ipcw_ht": np.nan,
        }
    brier = np.sum(
        weights[valid] * (y[valid] - prediction[valid]) ** 2
    ) / eligible_n
    nbll = -np.sum(
        weights[valid]
        * (
            y[valid] * np.log(prediction[valid])
            + (1.0 - y[valid]) * np.log1p(-prediction[valid])
        )
    ) / eligible_n
    return {
        "brier_ipcw_ht": float(brier),
        "nbll_ipcw_ht": float(nbll),
    }


def competing_risk_auc(
    window: Mapping[str, np.ndarray],
    prediction: np.ndarray,
) -> Dict[str, float]:
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = known & np.isfinite(prediction) & (weights > 0)
    result = {
        "auc_ipcw_cr": np.nan,
        "auprc_ipcw_cr": np.nan,
    }
    if valid.any() and np.unique(y[valid].astype(int)).size == 2:
        result["auc_ipcw_cr"] = float(roc_auc_score(
            y[valid], prediction[valid], sample_weight=weights[valid]
        ))
        result["auprc_ipcw_cr"] = float(average_precision_score(
            y[valid], prediction[valid], sample_weight=weights[valid]
        ))
    return result


@njit(cache=False, nogil=True)
def _xgb_fenwick_add(tree, index, value):
    size = tree.shape[0]
    while index < size:
        tree[index] += value
        index += index & -index


@njit(cache=False, nogil=True)
def _xgb_fenwick_sum(tree, index):
    total = 0.0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=False, nogil=True)
def _xgb_wolbers_cindex_core(
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
                _xgb_fenwick_add(tree_a, risk_rank[row], 1.0)
                total_a += 1.0
        if current_time <= tau:
            for position in range(start, end):
                row = order[position]
                if status[row] == 1:
                    rank = risk_rank[row]
                    lower = _xgb_fenwick_sum(tree_a, rank - 1)
                    through = _xgb_fenwick_sum(tree_a, rank)
                    tied = through - lower
                    pair_weight = 1.0 / (g_before[row] * g_after[row])
                    weighted_pairs += pair_weight * total_a
                    weighted_concordant += pair_weight * (lower + 0.5 * tied)
                    raw_pairs += int(total_a)
        for position in range(start, end):
            row = order[position]
            if status[row] != 0:
                _xgb_fenwick_add(tree_a, risk_rank[row], 1.0)
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
                _xgb_fenwick_add(
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
                    lower = _xgb_fenwick_sum(tree_b, rank - 1)
                    through = _xgb_fenwick_sum(tree_b, rank)
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
        _xgb_wolbers_cindex_core(
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
    fast = _xgb_wolbers_cindex_core(
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
    empty = {
        "calibration_intercept_cloglog": np.nan,
        "calibration_slope_cloglog": np.nan,
    }
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.clip(
        np.asarray(prediction, dtype=float), 1e-7, 1.0 - 1e-7
    )
    valid = known & (weights > 0) & np.isfinite(prediction)
    if valid.sum() < 20 or np.unique(y[valid].astype(int)).size < 2:
        return empty
    predictor = np.log(-np.log1p(-prediction[valid]))
    target = y[valid]
    weight = weights[valid]

    def objective(parameters: np.ndarray) -> float:
        eta = np.clip(
            parameters[0] + parameters[1] * predictor,
            -20.0, 20.0,
        )
        probability = np.clip(
            -np.expm1(-np.exp(eta)), 1e-10, 1.0 - 1e-10
        )
        return float(-np.sum(
            weight
            * (
                target * np.log(probability)
                + (1.0 - target) * np.log1p(-probability)
            )
        ))

    fit = minimize(
        objective, np.asarray([0.0, 1.0]), method="BFGS"
    )
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
    empty = {
        "ici_ipcw_isotonic": np.nan,
        "e50_ipcw_isotonic": np.nan,
        "e90_ipcw_isotonic": np.nan,
    }
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=float)
    weights = np.asarray(window["weights"], dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = known & (weights > 0) & np.isfinite(prediction)
    if (
        valid.sum() < 20
        or np.unique(prediction[valid]).size < 2
    ):
        return empty, pd.DataFrame(columns=[
            "predicted_risk", "observed_risk_smooth"
        ])
    fit = IsotonicRegression(
        y_min=0.0, y_max=1.0, out_of_bounds="clip"
    )
    fit.fit(
        prediction[valid],
        y[valid],
        sample_weight=weights[valid],
    )
    fitted = fit.predict(prediction[valid])
    error = np.abs(fitted - prediction[valid])
    scalar = {
        "ici_ipcw_isotonic": float(np.average(
            error, weights=weights[valid]
        )),
        "e50_ipcw_isotonic": float(np.quantile(error, 0.50)),
        "e90_ipcw_isotonic": float(np.quantile(error, 0.90)),
    }
    grid = np.linspace(
        float(prediction[valid].min()),
        float(prediction[valid].max()),
        grid_size,
    )
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
    minimum = max(
        20,
        CFG.calibration_bins,
        CFG.min_calibration_group_n,
        CFG.min_calibration_bin_n,
    )
    if values.size < minimum:
        return []
    edges = np.unique(np.quantile(
        values, np.linspace(0.0, 1.0, CFG.calibration_bins + 1)
    ))
    if edges.size <= 2:
        bin_id = np.zeros(values.size, dtype=int)
    else:
        bin_id = np.searchsorted(
            edges[1:-1], values, side="right"
        )
    rows = []
    for current_bin in np.unique(bin_id):
        selected = bin_id == current_bin
        if int(selected.sum()) < CFG.min_calibration_bin_n:
            continue
        predicted_mean = float(values[selected].mean())
        observed = aalen_johansen_cif(
            times[selected], states[selected], float(horizon)
        )
        rows.append({
            "calibration_bin": int(current_bin),
            "n": int(selected.sum()),
            "predicted_mean": predicted_mean,
            "predicted_median": float(np.median(values[selected])),
            "predicted_min": float(values[selected].min()),
            "predicted_max": float(values[selected].max()),
            "observed_aj_cif": observed,
            "predicted_minus_observed": (
                predicted_mean - observed
                if np.isfinite(observed) else np.nan
            ),
            # CRESCENT sign convention (observed - predicted).
            "observed_minus_predicted": (
                observed - predicted_mean
                if np.isfinite(observed) else np.nan
            ),
        })
    return rows


def integrate_metric(
    values: Sequence[float], horizons: Sequence[int]
) -> float:
    value_array = np.asarray(values, dtype=float)
    horizon_array = np.asarray(horizons, dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(horizon_array)
    if valid.sum() < 2:
        return np.nan
    x = horizon_array[valid]
    y = value_array[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0:
        return np.nan
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is None:
        trapezoid = np.trapz
    return float(trapezoid(y, x) / span)



# -----------------------------------------------------------------------------
# CRESCENT-compatible naming helpers
# -----------------------------------------------------------------------------
CRESCENT_ALIAS_MAP = {
    "auroc": "auc_ipcw_cr",
    "auprc": "auprc_ipcw_cr",
    "brier_ipcw": "brier_ipcw_ht",
    "nbll_ipcw": "nbll_ipcw_ht",
}


def add_crescent_alias_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for alias, canonical in CRESCENT_ALIAS_MAP.items():
        if canonical in frame.columns and alias not in frame.columns:
            frame[alias] = frame[canonical]
    if "mean_predicted_cif" in frame.columns and "mean_predicted_risk" not in frame.columns:
        frame["mean_predicted_risk"] = frame["mean_predicted_cif"]
    return frame


def ipcw_event_rate(window: Mapping[str, np.ndarray]) -> float:
    valid = (
        np.asarray(window["known"], dtype=bool)
        & np.isfinite(window["weights"]) & (np.asarray(window["weights"]) > 0)
    )
    if not valid.any():
        return float("nan")
    return float(np.average(
        np.asarray(window["y"], dtype=float)[valid],
        weights=np.asarray(window["weights"], dtype=float)[valid],
    ))


def weighted_quantile(values, weights, quantile) -> float:
    """Identical to the CRESCENT `_weighted_quantile`."""
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not bool(valid.any()):
        return float("nan")
    values = values[valid]
    weights = weights[valid]
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    target = float(quantile) * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, target, side="left"))
    return float(values[min(index, values.size - 1)])


def build_calibration_error_summary(calibration: pd.DataFrame) -> pd.DataFrame:
    """CRESCENT `test_calibration_error_summary.csv`: AJ-decile ICI / E50 / E90."""
    rows = []
    for risk_type in ("death_adjusted", "unadjusted"):
        for horizon in CFG.report_horizons_months:
            if calibration.empty:
                sub = calibration
            else:
                sub = calibration[
                    calibration["risk_type"].eq(risk_type)
                    & calibration["horizon_months"].eq(int(horizon))
                ]
            if sub.empty and risk_type == "unadjusted":
                continue
            error = np.abs(pd.to_numeric(
                sub.get("observed_minus_predicted", pd.Series(dtype=float)), errors="coerce"
            ).to_numpy(float))
            weight = pd.to_numeric(
                sub.get("n", pd.Series(dtype=float)), errors="coerce"
            ).to_numpy(float)
            keep = np.isfinite(error)
            error, weight = error[keep], weight[keep]
            if error.size:
                rows.append({
                    "horizon_months": int(horizon), "risk_type": risk_type,
                    "ici": float(np.average(error, weights=weight)),
                    "e50": weighted_quantile(error, weight, 0.50),
                    "e90": weighted_quantile(error, weight, 0.90),
                    "n_calibration_bins": int(error.size),
                    "calibration_weight_n": int(weight.sum()),
                })
            else:
                rows.append({
                    "horizon_months": int(horizon), "risk_type": risk_type,
                    "ici": np.nan, "e50": np.nan, "e90": np.nan,
                    "n_calibration_bins": 0, "calibration_weight_n": 0,
                })
    return pd.DataFrame(rows)


def build_crescent_summary(
    micro: pd.DataFrame,
    per_condition: pd.DataFrame,
    calibration_error: pd.DataFrame,
) -> pd.DataFrame:
    """CRESCENT-schema `test_summary_publication.csv` (micro/macro by horizon)."""
    rows = []
    for risk_type in micro["risk_type"].unique():
        for horizon in CFG.report_horizons_months:
            m = micro[(micro["horizon_months"] == horizon) & (micro["risk_type"] == risk_type)]
            sub = per_condition[
                (per_condition["horizon_months"] == horizon)
                & (per_condition["risk_type"] == risk_type)
            ]
            cal = calibration_error[
                (calibration_error["horizon_months"] == horizon)
                & (calibration_error["risk_type"] == risk_type)
            ]
            mr = m.iloc[0] if len(m) else pd.Series(dtype=float)
            rows.append({
                "horizon_months": int(horizon), "risk_type": risk_type,
                "micro_auroc": float(mr.get("auroc", np.nan)),
                "micro_auprc": float(mr.get("auprc", np.nan)),
                "micro_brier_ipcw": float(mr.get("brier_ipcw", np.nan)),
                "micro_nbll_ipcw": float(mr.get("nbll_ipcw", np.nan)),
                "macro_auroc": float(sub["auroc"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_auprc": float(sub["auprc"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_brier_ipcw": float(sub["brier_ipcw"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_nbll_ipcw": float(sub["nbll_ipcw"].mean(skipna=True)) if len(sub) else np.nan,
                "macro_ctd_ipcw": np.nan,
                "ctd_status": "disabled_not_validly_implemented",
                "macro_cindex_ipcw_cr": (
                    float(sub["cindex_ipcw_cr"].mean(skipna=True))
                    if len(sub) and "cindex_ipcw_cr" in sub.columns else np.nan
                ),
                "mean_predicted_risk": float(sub["mean_predicted_risk"].mean(skipna=True)) if len(sub) else np.nan,
                "n_conditions_with_auroc": int(sub["auroc"].notna().sum()) if len(sub) else 0,
                "n_conditions": int(K),
                "ici": float(cal["ici"].iloc[0]) if len(cal) else np.nan,
                "e50": float(cal["e50"].iloc[0]) if len(cal) else np.nan,
                "e90": float(cal["e90"].iloc[0]) if len(cal) else np.nan,
            })
    summary = pd.DataFrame(rows)
    integrated_rows = []
    for risk_type in summary["risk_type"].unique():
        sub = summary[summary["risk_type"] == risk_type]
        row = {"horizon_months": "integrated_mean_over_reported_horizons",
               "risk_type": risk_type, "ctd_status": "disabled_not_validly_implemented",
               "macro_ctd_ipcw": np.nan, "n_conditions": int(K)}
        for column in summary.columns:
            if column in row or column in {"horizon_months", "risk_type"}:
                continue
            row[column] = float(sub[column].mean(skipna=True))
        row["n_conditions_with_auroc"] = int(sub["n_conditions_with_auroc"].mean(skipna=True))
        integrated_rows.append(row)
    return pd.concat([summary, pd.DataFrame(integrated_rows)], ignore_index=True, sort=False)

# --- Notebook cell 28 ---
def evaluate_all_metrics(
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> Dict[str, pd.DataFrame]:
    train_outcomes = make_condition_outcomes(TRAIN_SAMPLES)
    test_outcomes = make_condition_outcomes(test_frame)
    condition_kms = [
        CensoringKM(
            train_outcomes["exit_time"][
                train_outcomes["eligible"][:, condition], condition
            ],
            train_outcomes["status"][
                train_outcomes["eligible"][:, condition], condition
            ] == 0,
        )
        for condition in range(K)
    ]
    condition_cindex_kms = [
        CensoringKM(
            train_outcomes["exit_time_untruncated"][
                train_outcomes["eligible"][:, condition], condition
            ],
            train_outcomes["status_untruncated"][
                train_outcomes["eligible"][:, condition], condition
            ] == 0,
        )
        for condition in range(K)
    ]

    train_null = np.full(
        (K, len(DENSE_HORIZONS_MONTHS)), np.nan, dtype=float
    )
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
    micro_rows: List[Dict[str, Any]] = []
    dense_point_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(CFG.seed + 2026)
    dense_index = {
        int(horizon): index
        for index, horizon in enumerate(DENSE_HORIZONS_MONTHS)
    }

    for risk_type, source in risk_sources:
        risk_cube = predictions[source]
        for horizon in CFG.report_horizons_months:
            grid_index = dense_index[int(horizon)]
            micro_y: List[np.ndarray] = []
            micro_p: List[np.ndarray] = []
            micro_w: List[np.ndarray] = []
            micro_eligible = 0
            micro_known = 0
            micro_cases = 0
            micro_competing = 0
            for condition in range(K):
                eligible = test_outcomes["eligible"][:, condition]
                status = test_outcomes["status"][:, condition]
                exit_time = test_outcomes["exit_time"][:, condition]
                km = condition_kms[condition]
                window = binary_window(
                    eligible, status, exit_time, float(horizon), km
                )
                prediction = np.asarray(
                    risk_cube[:, condition, grid_index], dtype=float
                )
                eligible_n = int(eligible.sum())
                _valid = (
                    window["known"] & np.isfinite(prediction)
                    & np.isfinite(window["weights"]) & (window["weights"] > 0)
                )
                micro_y.append(window["y"][_valid])
                micro_p.append(prediction[_valid])
                micro_w.append(window["weights"][_valid])
                micro_eligible += eligible_n
                micro_known += int(window["known"].sum())
                micro_cases += int(window["case"].sum())
                micro_competing += int(window["competing"].sum())

                discrimination = competing_risk_auc(
                    window, prediction
                )
                scores = ipcw_ht_scores(
                    window["y"], prediction, window["weights"], eligible_n
                )
                null_prediction = np.full_like(
                    prediction, train_null[condition, grid_index]
                )
                null_scores = ipcw_ht_scores(
                    window["y"],
                    null_prediction,
                    window["weights"],
                    eligible_n,
                )
                null_brier = null_scores["brier_ipcw_ht"]
                scaled_brier = (
                    1.0 - scores["brier_ipcw_ht"] / null_brier
                    if np.isfinite(null_brier) and null_brier > 0
                    else np.nan
                )
                cindex = competing_risk_cindex(
                    eligible,
                    test_outcomes["status_untruncated"][:, condition],
                    test_outcomes["exit_time_untruncated"][:, condition],
                    prediction,
                    float(horizon),
                    condition_cindex_kms[condition],
                    rng,
                )
                observed = aalen_johansen_cif(
                    exit_time[eligible],
                    status[eligible],
                    float(horizon),
                )
                mean_prediction = (
                    float(prediction[eligible].mean())
                    if eligible_n else np.nan
                )
                calibration_in_large = (
                    mean_prediction - observed
                    if np.isfinite(mean_prediction)
                    and np.isfinite(observed)
                    else np.nan
                )
                observed_to_expected = (
                    observed / mean_prediction
                    if np.isfinite(observed)
                    and np.isfinite(mean_prediction)
                    and mean_prediction > 0
                    else np.nan
                )
                calibration_parameters = (
                    calibration_intercept_slope_cloglog(
                        window, prediction
                    )
                )
                smooth_scalar, curve = smooth_isotonic_calibration(
                    window, prediction
                )

                per_condition_rows.append({
                    "model_name": CFG.model_name,
                    "horizon_months": int(horizon),
                    "horizon_years": float(horizon / 12.0),
                    "risk_type": risk_type,
                    "condition_id": int(condition),
                    "condition_name": CONDITION_NAMES[condition],
                    "eligible_n": eligible_n,
                    "known_n": int(window["known"].sum()),
                    "event_n": int(window["case"].sum()),
                    "death_competing_n": int(
                        window["competing"].sum()
                    ),
                    "observed_event_free_n": int(
                        window["event_free_control"].sum()
                    ),
                    "admin_censored_before_horizon_n": int(
                        (eligible & ~window["known"]).sum()
                    ),
                    "mean_predicted_cif": mean_prediction,
                    "event_rate": ipcw_event_rate(window),
                    "observed_aj_cif": observed,
                    "calibration_in_large_pred_minus_obs": (
                        calibration_in_large
                    ),
                    "observed_to_expected_ratio": (
                        observed_to_expected
                    ),
                    **discrimination,
                    **scores,
                    "brier_null_ipcw_ht": null_brier,
                    "scaled_brier_ipcw_ht": scaled_brier,
                    **cindex,
                    **calibration_parameters,
                    **smooth_scalar,
                    **ipcw_weight_diagnostics(window),
                })

                for row in grouped_aj_calibration(
                    prediction, eligible, exit_time, status, int(horizon)
                ):
                    calibration_rows.append({
                        "model_name": CFG.model_name,
                        "horizon_months": int(horizon),
                        "horizon_years": float(horizon / 12.0),
                        "risk_type": risk_type,
                        "condition_id": int(condition),
                        "condition_name": CONDITION_NAMES[condition],
                        **row,
                    })
                if not curve.empty:
                    curve.insert(0, "condition_name", CONDITION_NAMES[condition])
                    curve.insert(0, "condition_id", int(condition))
                    curve.insert(0, "risk_type", risk_type)
                    curve.insert(0, "horizon_months", int(horizon))
                    curve.insert(0, "model_name", CFG.model_name)
                    smooth_rows.extend(curve.to_dict("records"))

            # Pooled (micro) metrics across all eligible condition-person pairs.
            if micro_y:
                my = np.concatenate(micro_y)
                mp = np.concatenate(micro_p)
                mw = np.concatenate(micro_w)
                micro_auc = competing_risk_auc(
                    {"known": np.ones(my.shape, dtype=bool), "y": my, "weights": mw}, mp
                )
                micro_scores = ipcw_ht_scores(my, mp, mw, micro_eligible)
                micro_rows.append({
                    "horizon_months": int(horizon),
                    "risk_type": risk_type,
                    "auroc": micro_auc["auc_ipcw_cr"],
                    "auprc": micro_auc["auprc_ipcw_cr"],
                    "event_rate": (
                        float(np.average(my, weights=mw)) if my.size else np.nan
                    ),
                    "brier_ipcw": micro_scores["brier_ipcw_ht"],
                    "nbll_ipcw": micro_scores["nbll_ipcw_ht"],
                    "weight_sum": float(mw.sum()),
                    "normalization_n": int(micro_eligible),
                    "eligible_pairs": int(micro_eligible),
                    "known_pairs": int(micro_known),
                    "condition_events": int(micro_cases),
                    "death_competing_events": int(micro_competing),
                })
            del micro_y, micro_p, micro_w

        for condition in range(K):
            eligible = test_outcomes["eligible"][:, condition]
            status = test_outcomes["status"][:, condition]
            exit_time = test_outcomes["exit_time"][:, condition]
            km = condition_kms[condition]
            eligible_n = int(eligible.sum())
            brier_values: List[float] = []
            null_brier_values: List[float] = []
            nbll_values: List[float] = []
            auc_values: List[float] = []
            calibration_values: List[float] = []

            for grid_index, horizon in enumerate(
                DENSE_HORIZONS_MONTHS
            ):
                window = binary_window(
                    eligible, status, exit_time, float(horizon), km
                )
                prediction = np.asarray(
                    risk_cube[:, condition, grid_index], dtype=float
                )
                scores = ipcw_ht_scores(
                    window["y"], prediction, window["weights"], eligible_n
                )
                null_prediction = np.full_like(
                    prediction, train_null[condition, grid_index]
                )
                null_scores = ipcw_ht_scores(
                    window["y"],
                    null_prediction,
                    window["weights"],
                    eligible_n,
                )
                auc_pair = competing_risk_auc(window, prediction)
                auc_value = auc_pair["auc_ipcw_cr"]
                observed = aalen_johansen_cif(
                    exit_time[eligible],
                    status[eligible],
                    float(horizon),
                )
                mean_prediction = (
                    float(prediction[eligible].mean())
                    if eligible_n else np.nan
                )
                if risk_type == "death_adjusted":
                    dense_point_rows.append({
                        "model_name": CFG.model_name,
                        "risk_type": risk_type,
                        "condition_id": int(condition),
                        "condition_name": CONDITION_NAMES[condition],
                        "horizon_months": int(horizon),
                        "eligible_n": eligible_n,
                        "known_n": int(window["known"].sum()),
                        "event_n": int(window["case"].sum()),
                        "death_competing_n": int(window["competing"].sum()),
                        "mean_predicted_cif": mean_prediction,
                        "observed_aj_cif": observed,
                        "absolute_calibration_error": (
                            abs(mean_prediction - observed)
                            if np.isfinite(mean_prediction) and np.isfinite(observed)
                            else np.nan
                        ),
                        "auc_ipcw_cr": auc_pair["auc_ipcw_cr"],
                        "auprc_ipcw_cr": auc_pair["auprc_ipcw_cr"],
                        "brier_ipcw_ht": scores["brier_ipcw_ht"],
                        "brier_null_ipcw_ht": null_scores["brier_ipcw_ht"],
                        "nbll_ipcw_ht": scores["nbll_ipcw_ht"],
                    })
                brier_values.append(scores["brier_ipcw_ht"])
                null_brier_values.append(
                    null_scores["brier_ipcw_ht"]
                )
                nbll_values.append(scores["nbll_ipcw_ht"])
                auc_values.append(auc_value)
                calibration_values.append(
                    abs(mean_prediction - observed)
                    if np.isfinite(mean_prediction)
                    and np.isfinite(observed)
                    else np.nan
                )

            ibs = integrate_metric(
                brier_values, DENSE_HORIZONS_MONTHS
            )
            ibs_null = integrate_metric(
                null_brier_values, DENSE_HORIZONS_MONTHS
            )
            integrated_rows.append({
                "model_name": CFG.model_name,
                "risk_type": risk_type,
                "condition_id": int(condition),
                "condition_name": CONDITION_NAMES[condition],
                "integration_start_months": int(
                    DENSE_HORIZONS_MONTHS[0]
                ),
                "integration_end_months": int(
                    DENSE_HORIZONS_MONTHS[-1]
                ),
                "grid_step_months": BIN_WIDTH_MONTHS,
                "n_grid_points": len(DENSE_HORIZONS_MONTHS),
                "ibs_ipcw_ht": ibs,
                "ibs_null_ipcw_ht": ibs_null,
                "scaled_ibs_ipcw_ht": (
                    1.0 - ibs / ibs_null
                    if np.isfinite(ibs)
                    and np.isfinite(ibs_null)
                    and ibs_null > 0
                    else np.nan
                ),
                "inbll_ipcw_ht": integrate_metric(
                    nbll_values, DENSE_HORIZONS_MONTHS
                ),
                "iauc_ipcw_cr": integrate_metric(
                    auc_values, DENSE_HORIZONS_MONTHS
                ),
                "integrated_abs_calibration_error": integrate_metric(
                    calibration_values, DENSE_HORIZONS_MONTHS
                ),
            })

    per_condition = add_crescent_alias_columns(pd.DataFrame(per_condition_rows))
    calibration = pd.DataFrame(calibration_rows)
    smooth = pd.DataFrame(smooth_rows)
    integrated = pd.DataFrame(integrated_rows)

    macro_rows: List[Dict[str, Any]] = []
    summary_metrics = [
        "auc_ipcw_cr", "auprc_ipcw_cr", "cindex_ipcw_cr",
        "brier_ipcw_ht", "scaled_brier_ipcw_ht", "nbll_ipcw_ht",
        "calibration_in_large_pred_minus_obs",
        "calibration_slope_cloglog", "ici_ipcw_isotonic",
        "observed_aj_cif", "mean_predicted_cif",
    ]
    for (risk_type, horizon), group in per_condition.groupby(
        ["risk_type", "horizon_months"], sort=True
    ):
        row: Dict[str, Any] = {
            "model_name": CFG.model_name,
            "risk_type": risk_type,
            "horizon_months": int(horizon),
            "horizon_years": float(horizon / 12.0),
            "n_conditions": K,
            "conditions_with_auc": int(
                group["auc_ipcw_cr"].notna().sum()
            ),
        }
        for metric in summary_metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"macro_mean_{metric}"] = float(
                values.mean(skipna=True)
            )
            row[f"macro_median_{metric}"] = float(
                values.median(skipna=True)
            )
            row[f"macro_q25_{metric}"] = float(values.quantile(0.25))
            row[f"macro_q75_{metric}"] = float(values.quantile(0.75))
        macro_rows.append(row)
    macro = pd.DataFrame(macro_rows)

    train_death = make_death_outcomes(TRAIN_SAMPLES)
    test_death = make_death_outcomes(test_frame)
    death_km = CensoringKM(
        train_death["exit_time"][train_death["eligible"]],
        train_death["status"][train_death["eligible"]] == 0,
    )
    death_cindex_km = CensoringKM(
        train_death["exit_time_untruncated"][train_death["eligible"]],
        train_death["status_untruncated"][train_death["eligible"]] == 0,
    )
    death_rows: List[Dict[str, Any]] = []
    death_brier: List[float] = []
    death_null_brier: List[float] = []
    death_nbll: List[float] = []
    death_auc: List[float] = []

    for grid_index, horizon in enumerate(DENSE_HORIZONS_MONTHS):
        window = binary_window(
            test_death["eligible"],
            test_death["status"],
            test_death["exit_time"],
            float(horizon),
            death_km,
        )
        prediction = np.asarray(
            predictions["death"][:, grid_index], dtype=float
        )
        eligible_n = int(test_death["eligible"].sum())
        scores = ipcw_ht_scores(
            window["y"], prediction, window["weights"], eligible_n
        )
        train_null_death = aalen_johansen_cif(
            train_death["exit_time"][train_death["eligible"]],
            train_death["status"][train_death["eligible"]],
            float(horizon),
        )
        null_prediction = np.full_like(
            prediction, train_null_death
        )
        null_scores = ipcw_ht_scores(
            window["y"],
            null_prediction,
            window["weights"],
            eligible_n,
        )
        discrimination = competing_risk_auc(window, prediction)
        observed = aalen_johansen_cif(
            test_death["exit_time"][test_death["eligible"]],
            test_death["status"][test_death["eligible"]],
            float(horizon),
        )
        mean_prediction = (
            float(prediction[test_death["eligible"]].mean())
            if eligible_n else np.nan
        )
        death_brier.append(scores["brier_ipcw_ht"])
        death_null_brier.append(null_scores["brier_ipcw_ht"])
        death_nbll.append(scores["nbll_ipcw_ht"])
        death_auc.append(discrimination["auc_ipcw_cr"])

        if int(horizon) in CFG.report_horizons_months:
            cindex = competing_risk_cindex(
                test_death["eligible"],
                test_death["status_untruncated"],
                test_death["exit_time_untruncated"],
                prediction,
                float(horizon),
                death_cindex_km,
                rng,
            )
            death_rows.append({
                "model_name": CFG.model_name,
                "horizon_months": int(horizon),
                "horizon_years": float(horizon / 12.0),
                "eligible_n": eligible_n,
                "known_n": int(window["known"].sum()),
                "death_event_n": int(window["case"].sum()),
                "admin_censored_before_horizon_n": int(
                    (
                        test_death["eligible"] & ~window["known"]
                    ).sum()
                ),
                "mean_predicted_death_risk": mean_prediction,
                "event_rate": ipcw_event_rate(window),
                "observed_death_risk": observed,
                "calibration_in_large_pred_minus_obs": (
                    mean_prediction - observed
                    if np.isfinite(mean_prediction)
                    and np.isfinite(observed)
                    else np.nan
                ),
                "observed_to_expected_ratio": (
                    observed / mean_prediction
                    if np.isfinite(observed)
                    and np.isfinite(mean_prediction)
                    and mean_prediction > 0
                    else np.nan
                ),
                **discrimination,
                **scores,
                "brier_null_ipcw_ht": null_scores["brier_ipcw_ht"],
                "scaled_brier_ipcw_ht": (
                    1.0
                    - scores["brier_ipcw_ht"]
                    / null_scores["brier_ipcw_ht"]
                    if np.isfinite(null_scores["brier_ipcw_ht"])
                    and null_scores["brier_ipcw_ht"] > 0
                    else np.nan
                ),
                **cindex,
                **calibration_intercept_slope_cloglog(
                    window, prediction
                ),
                **ipcw_weight_diagnostics(window),
            })

    death = add_crescent_alias_columns(pd.DataFrame(death_rows))
    death_ibs = integrate_metric(
        death_brier, DENSE_HORIZONS_MONTHS
    )
    death_ibs_null = integrate_metric(
        death_null_brier, DENSE_HORIZONS_MONTHS
    )
    death_integrated = pd.DataFrame([{
        "model_name": CFG.model_name,
        "integration_start_months": int(DENSE_HORIZONS_MONTHS[0]),
        "integration_end_months": int(DENSE_HORIZONS_MONTHS[-1]),
        "ibs_ipcw_ht": death_ibs,
        "ibs_null_ipcw_ht": death_ibs_null,
        "scaled_ibs_ipcw_ht": (
            1.0 - death_ibs / death_ibs_null
            if np.isfinite(death_ibs)
            and np.isfinite(death_ibs_null)
            and death_ibs_null > 0
            else np.nan
        ),
        "inbll_ipcw_ht": integrate_metric(
            death_nbll, DENSE_HORIZONS_MONTHS
        ),
        "iauc_ipcw": integrate_metric(
            death_auc, DENSE_HORIZONS_MONTHS
        ),
    }])

    micro = pd.DataFrame(micro_rows)
    calibration_error = build_calibration_error_summary(calibration)
    return {
        "micro": micro,
        "calibration_error": calibration_error,
        "dense_point": pd.DataFrame(dense_point_rows),
        "per_condition": per_condition,
        "integrated": integrated,
        "calibration": calibration,
        "smooth_calibration": smooth,
        "macro": macro,
        "death": death,
        "death_integrated": death_integrated,
    }




def save_raw_landmark_rate_auc() -> None:
    train_outcomes = make_condition_outcomes(TRAIN_SAMPLES)
    test_outcomes = make_condition_outcomes(TEST_SAMPLES)
    condition_kms = [
        CensoringKM(
            train_outcomes["exit_time"][
                train_outcomes["eligible"][:, condition], condition
            ],
            train_outcomes["status"][
                train_outcomes["eligible"][:, condition], condition
            ] == 0,
        )
        for condition in range(K)
    ]
    rows: List[Dict[str, Any]] = []
    scores = np.asarray(
        TEST_PREDICTIONS["raw_condition_rate_gap"], dtype=float
    )
    for horizon in CFG.report_horizons_months:
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
            rows.append({
                "model_name": CFG.model_name,
                "score_type": (
                    "first_post_gap_bin_average_condition_rate"
                ),
                "horizon_months": int(horizon),
                "condition_id": int(condition),
                "condition_name": CONDITION_NAMES[condition],
                "eligible_n": int(eligible.sum()),
                "known_n": int(window["known"].sum()),
                "event_n": int(window["case"].sum()),
                "death_competing_n": int(
                    window["competing"].sum()
                ),
                **competing_risk_auc(
                    window, scores[:, condition]
                ),
            })
    pd.DataFrame(rows).to_csv(
        PREDICTION_PATHS["raw_rate_auc"], index=False
    )

# --- Notebook cell 30 ---
def save_compatibility_prediction_npz(
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> Path:
    dense_index = {
        int(horizon): index
        for index, horizon in enumerate(DENSE_HORIZONS_MONTHS)
    }
    payload: Dict[str, Any] = {
        "eid": test_frame["eid"].astype(str).to_numpy(),
        "condition_names": np.asarray(CONDITION_NAMES, dtype=object),
        "horizons_months": np.asarray(
            CFG.report_horizons_months, dtype=np.int32
        ),
        "entry_delay_months": np.asarray(CFG.gap_months, dtype=np.float32),
        "gap_months": np.asarray(CFG.gap_months, dtype=np.float32),  # legacy alias
        "delayed_entry_schema_version": np.asarray(DELAYED_ENTRY_SCHEMA_VERSION),
        "index_age_years": test_frame[
            "index_age_years"
        ].to_numpy(np.float32),
        "index_dt": test_frame[
            "index_dt"
        ].astype(str).to_numpy(dtype=object),
        "index_offset_days_from_recruitment": test_frame[
            "index_offset_days_from_recruitment"
        ].to_numpy(np.int32),
        "at_risk": np.stack(
            test_frame["at_risk"].to_numpy()
        ).astype(np.float32),
        "event_obs": np.stack(
            test_frame["event_obs"].to_numpy()
        ).astype(np.float32),
        "event_bin": np.stack(
            test_frame["event_bin"].to_numpy()
        ).astype(np.int16),
        "t_tilde": np.stack(
            test_frame["t_tilde"].to_numpy()
        ).astype(np.float32),
        "C_months": test_frame["C_months"].to_numpy(np.float32),
        "C_bin": test_frame["C_bin"].to_numpy(np.int16),
        "death_observed_post_index": test_frame[
            "death_observed_post_index"
        ].to_numpy(np.float32),
        "death_time_months": test_frame[
            "death_time_months"
        ].to_numpy(np.float32),
        "death_bin": test_frame["death_bin"].to_numpy(np.int16),
        "cr_at_risk": np.stack(
            test_frame["cr_at_risk"].to_numpy()
        ).astype(np.float32),
        "cr_event_type": np.stack(
            test_frame["cr_event_type"].to_numpy()
        ).astype(np.int8),
        "cr_event_time_months": np.stack(
            test_frame["cr_event_time_months"].to_numpy()
        ).astype(np.float32),
    }
    for horizon in CFG.report_horizons_months:
        grid_index = dense_index[int(horizon)]
        payload[f"risk_death_adjusted_h{horizon}"] = np.asarray(
            predictions["adjusted"][:, :, grid_index],
            dtype=np.float32,
        )
        if predictions.get("unadjusted") is not None:
            payload[f"risk_unadjusted_h{horizon}"] = np.asarray(
                predictions["unadjusted"][:, :, grid_index],
                dtype=np.float32,
            )
        payload[f"risk_death_h{horizon}"] = np.asarray(
            predictions["death"][:, grid_index],
            dtype=np.float32,
        )
    path = RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
    np.savez_compressed(path, **payload)
    return path


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


TEST_PERFORMANCE_FILES = {
    "per_condition": RUN_DIR
    / "test_per_condition_metrics_by_horizon.csv",
    "integrated": RUN_DIR
    / "test_integrated_metrics_by_condition.csv",
    "calibration": RUN_DIR
    / "test_aj_calibration_by_horizon.csv",
    "smooth_calibration": RUN_DIR
    / "test_smooth_calibration_curves.csv",
    "macro": RUN_DIR / "test_macro_summary_by_horizon.csv",
    "death": RUN_DIR / "test_death_head_metrics_by_horizon.csv",
    "death_with_ci": RUN_DIR / "test_death_head_metrics_by_horizon_with_ci.csv",
    "micro": RUN_DIR / "test_micro_metrics_by_horizon.csv",
    "calibration_error": RUN_DIR / "test_calibration_error_summary.csv",
    "dense_point": RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv",
    "dense_prediction_npz": RUN_DIR
    / "test_dense_prediction_arrays_for_integrated_metrics.npz",
    "integrated_metadata": RUN_DIR / "test_integrated_metrics_metadata.json",
    "publication_extended": RUN_DIR / "test_summary_publication_xgboost_extended.csv",
    "death_integrated": RUN_DIR
    / "test_death_integrated_metrics.csv",
    "publication": RUN_DIR / "test_summary_publication.csv",
    "prediction_npz": RUN_DIR
    / "test_prediction_arrays_death_adjusted.npz",
}


def save_evaluation_outputs(
    result: Mapping[str, pd.DataFrame],
    test_frame: pd.DataFrame,
    predictions: Mapping[str, Any],
) -> None:
    result["per_condition"].to_csv(
        TEST_PERFORMANCE_FILES["per_condition"], index=False
    )
    result["integrated"].to_csv(
        TEST_PERFORMANCE_FILES["integrated"], index=False
    )
    result["calibration"].to_csv(
        TEST_PERFORMANCE_FILES["calibration"], index=False
    )
    result["smooth_calibration"].to_csv(
        TEST_PERFORMANCE_FILES["smooth_calibration"], index=False
    )
    result["macro"].to_csv(
        TEST_PERFORMANCE_FILES["macro"], index=False
    )
    result["death"].to_csv(
        TEST_PERFORMANCE_FILES["death"], index=False
    )
    death_with_ci = result["death"].copy()
    death_with_ci["ci_computed"] = False
    death_with_ci["ci_lower"] = np.nan
    death_with_ci["ci_upper"] = np.nan
    death_with_ci.to_csv(TEST_PERFORMANCE_FILES["death_with_ci"], index=False)
    result["micro"].to_csv(TEST_PERFORMANCE_FILES["micro"], index=False)
    result["calibration_error"].to_csv(
        TEST_PERFORMANCE_FILES["calibration_error"], index=False
    )
    result["dense_point"].to_csv(
        TEST_PERFORMANCE_FILES["dense_point"], index=False
    )
    result["death_integrated"].to_csv(
        TEST_PERFORMANCE_FILES["death_integrated"], index=False
    )

    primary_macro = result["macro"].query(
        "risk_type == 'death_adjusted'"
    ).copy()
    primary_integrated = result["integrated"].query(
        "risk_type == 'death_adjusted'"
    ).copy()
    integrated_summary = pd.DataFrame([{
        "model_name": CFG.model_name,
        "risk_type": "death_adjusted",
        "horizon_months": "integrated_12_to_120",
        "n_conditions": K,
        "macro_mean_ibs_ipcw_ht": primary_integrated[
            "ibs_ipcw_ht"
        ].mean(skipna=True),
        "macro_mean_scaled_ibs_ipcw_ht": primary_integrated[
            "scaled_ibs_ipcw_ht"
        ].mean(skipna=True),
        "macro_mean_inbll_ipcw_ht": primary_integrated[
            "inbll_ipcw_ht"
        ].mean(skipna=True),
        "macro_mean_iauc_ipcw_cr": primary_integrated[
            "iauc_ipcw_cr"
        ].mean(skipna=True),
        "macro_mean_integrated_abs_calibration_error": (
            primary_integrated[
                "integrated_abs_calibration_error"
            ].mean(skipna=True)
        ),
    }])
    extended_publication = pd.concat(
        [primary_macro, integrated_summary],
        ignore_index=True,
        sort=False,
    )
    extended_publication.to_csv(
        TEST_PERFORMANCE_FILES["publication_extended"], index=False
    )
    # CRESCENT-schema summary (micro_*/macro_* by horizon, ici/e50/e90).
    publication = build_crescent_summary(
        result["micro"], result["per_condition"], result["calibration_error"]
    )
    publication.to_csv(
        TEST_PERFORMANCE_FILES["publication"], index=False
    )

    save_compatibility_prediction_npz(test_frame, predictions)
    np.savez_compressed(
        TEST_PERFORMANCE_FILES["dense_prediction_npz"],
        eid=test_frame["eid"].astype(str).to_numpy(dtype=object),
        condition_names=np.asarray(CONDITION_NAMES, dtype=object),
        dense_horizons_months=np.asarray(DENSE_HORIZONS_MONTHS, dtype=np.int32),
        entry_delay_months=np.asarray(CFG.gap_months, dtype=np.float32),
        gap_months=np.asarray(CFG.gap_months, dtype=np.float32),
        delayed_entry_schema_version=np.asarray(DELAYED_ENTRY_SCHEMA_VERSION),
        risk_death_adjusted=np.asarray(predictions["adjusted"], dtype=np.float32),
    )
    atomic_json_dump({
        "schema_version": "ehr-competing-risk-canonical-integrated-evaluation.v1",
        "model_name": CFG.model_name,
        "risk_type": "death_adjusted",
        "entry_delay_months": int(CFG.gap_months),
        "dense_horizons_months": [int(v) for v in DENSE_HORIZONS_MONTHS],
        "integration_method": "trapezoidal mean over the canonical six-month grid",
        "censoring_distribution": (
            "condition-specific administrative-censoring Kaplan-Meier "
            "estimated from canonical train_samples"
        ),
        "censor_survival_floor": float(CFG.censor_survival_floor),
        "estimand": (
            "Condition incidence during (6, 6+horizon] months from the original index, "
            "before death, conditional on being alive, condition-free, and observed "
            "strictly beyond the six-month delayed-entry landmark. Horizons are months after entry."
        ),
        "delayed_entry_contract": DELAYED_ENTRY_CONTRACT,
        "n_test_participants": int(len(test_frame)),
        "n_conditions": int(K),
    }, TEST_PERFORMANCE_FILES["integrated_metadata"])

    metric_definitions = {
        "schema_version": (
            "ehr-competing-risk-canonical-evaluation.v3.0"
        ),
        "model": CFG.model_name,
        "estimand": (
            "Incident condition during (0 months, horizon] after the locked "
            "six-month landmark and before death; horizons are measured after the gap."
        ),
        "history_contract": (
            "Every canonical pre-index event is summarized; no sequence "
            "truncation is applied. Event-feature filtering is fitted on "
            "the training split only."
        ),
        "time_origin": (
            "main-model locked participant-specific canonical index"
        ),
        "gap_months": CFG.gap_months,
        "entry_delay_months": CFG.gap_months,
        "delayed_entry_contract": DELAYED_ENTRY_CONTRACT,
        "report_horizons_months": list(CFG.report_horizons_months),
        "dense_integration_horizons_months": list(
            DENSE_HORIZONS_MONTHS
        ),
        "censoring_distribution": (
            "Condition-specific administrative-censoring Kaplan-Meier "
            "estimated from the canonical training split and applied "
            "unchanged to test metrics."
        ),
        "auc_ipcw_cr": (
            "IPCW cumulative/dynamic competing-risk AUROC; cases have the "
            "condition by t, and controls are condition-free through t or "
            "have competing death."
        ),
        "brier_ipcw_ht": (
            "Horvitz-Thompson IPCW Brier score divided by the number "
            "eligible at the lead-time landmark."
        ),
        "nbll_ipcw_ht": (
            "Horvitz-Thompson IPCW Bernoulli negative log-likelihood."
        ),
        "observed_aj_cif": (
            "Aalen-Johansen observed cumulative incidence with death "
            "competing."
        ),
        "calibration_in_large_pred_minus_obs": (
            "Mean predicted CIF minus observed AJ CIF; ideal is zero."
        ),
        "calibration_intercept_slope_cloglog": (
            "IPCW complementary-log-log calibration intercept and slope; "
            "ideals are zero and one."
        ),
        "ibs_ipcw_ht": (
            "Trapezoidal mean IPCW Brier score over the 12-to-120-month "
            "dense six-month grid."
        ),
        "confidence_intervals": (
            "Participant-level percentile bootstrap with the frozen model "
            "bundle, predictions, condition axis, and train-derived "
            "censoring distributions fixed. The exact competing-risk "
            "C-index CI is reported at 60 months after the lead-time gap."
        ),
    }
    atomic_json_dump(
        metric_definitions,
        RUN_DIR / "test_metric_definitions.json",
    )

    evaluation_manifest = {
        "schema_version": CFG.schema_version,
        "evaluation_schema_version": (
            "ehr-competing-risk-canonical-evaluation.v3.0"
        ),
        "model": CFG.model_name,
        "test_rows": len(test_frame),
        "test_eid_axis_sha256": canonical_json_hash(
            test_frame["eid"].astype(str).tolist()
        ),
        "canonical_test_samples_sha256": sha256_file(
            CANONICAL_PATHS["test"]
        ),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "feature_axis_sha256": FROZEN_BUNDLE[
            "feature_axis_sha256"
        ],
        "raw_feature_axis_sha256": FROZEN_BUNDLE[
            "raw_feature_axis_sha256"
        ],
        "feature_mask_sha256": FROZEN_BUNDLE[
            "feature_mask_sha256"
        ],
        "prediction_manifest_sha256": sha256_file(
            PREDICTION_PATHS["manifest"]
        ),
        "model_bundle_sha256": sha256_file(MODEL_BUNDLE_PATH),
        "crescent_run_dir": str(MAIN_RUN_DIR),
        "crescent_model_variant": str(INPUT_MANIFEST.get("model_variant")),
        "crescent_provenance_all_splits": [
            verify_crescent_provenance(split) for split in ("train", "val", "test")
        ],
        "same_index_as_main_model": True,
        "same_split_as_main_model": True,
        "same_condition_axis_as_main_model": True,
        "same_targets_as_main_model": True,
        "canonical_target_metadata_sha256": sha256_file(
            CANONICAL_PATHS["target_metadata"]
        ),
        "canonical_vocab_scalers_sha256": sha256_file(
            CANONICAL_PATHS["vocab_scalers"]
        ),
        "canonical_input_manifest_sha256": sha256_file(
            CANONICAL_PATHS["input_manifest"]
        ),
        "outcome_horizon_rule": (
            "events after max report horizon are administratively censored; "
            "they are never moved to the horizon"
        ),
        "prediction_at_risk_axis": "cr_at_risk",
        "test_used_for_model_selection": False,
        "history_contract": "all canonical pre-index events",
        "events_discarded": 0,
        "report_horizons_months": list(CFG.report_horizons_months),
        "gap_months": CFG.gap_months,
        "primary_metric": (
            "macro mean integrated IPCW Brier score"
        ),
        "primary_prediction": (
            "death-adjusted condition cumulative incidence"
        ),
    }
    atomic_json_dump(
        evaluation_manifest,
        RUN_DIR / "test_evaluation_manifest.json",
    )
    artifact_hash_table().to_csv(
        RUN_DIR / "artifact_sha256.csv", index=False
    )

    print("\nPrimary death-adjusted macro results:")
    print(primary_macro.to_string(index=False))
    print("\nPrimary integrated condition results:")
    columns = [
        "condition_id", "condition_name", "ibs_ipcw_ht",
        "scaled_ibs_ipcw_ht", "inbll_ipcw_ht", "iauc_ipcw_cr",
        "integrated_abs_calibration_error",
    ]
    print(primary_integrated[columns].head(20).to_string(index=False))


def verify_against_crescent_reference(per_condition: pd.DataFrame) -> Dict[str, Any]:
    """Cohort/target identity against the CRESCENT run (eid order, conditions, risk sets)."""
    report: Dict[str, Any] = {"reference_run_dir": str(MAIN_RUN_DIR)}
    reference_npz = MAIN_RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
    reference_metrics = MAIN_RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
    if reference_npz.is_file():
        with np.load(reference_npz, allow_pickle=True) as ref:
            ref_eid = ref["eid"].astype(str).tolist()
            ref_conditions = ref["condition_names"].astype(str).tolist()
            ref_entry = float(ref["entry_delay_months"]) if "entry_delay_months" in ref else None
        if ref_eid != TEST_SAMPLES["eid"].astype(str).tolist():
            raise RuntimeError("Test participant order differs from the CRESCENT prediction NPZ.")
        if ref_conditions != CONDITION_NAMES:
            raise RuntimeError("Condition order differs from the CRESCENT prediction NPZ.")
        if ref_entry is not None and abs(ref_entry - float(CFG.gap_months)) > 1e-8:
            raise RuntimeError("Entry delay differs from the CRESCENT run.")
        report["eid_and_condition_axes_match_crescent_npz"] = True
    else:
        report["eid_and_condition_axes_match_crescent_npz"] = None
        print("NOTE: CRESCENT test_prediction_arrays_death_adjusted.npz not found; "
              "axis check against CRESCENT predictions skipped.", flush=True)
    if reference_metrics.is_file():
        keys = ["horizon_months", "condition_name"]
        ref = pd.read_csv(reference_metrics)
        if "risk_type" in ref.columns:
            ref = ref[ref["risk_type"].astype(str).eq("death_adjusted")]
        ours = per_condition[per_condition["risk_type"].eq("death_adjusted")]
        merged = ours[keys + ["eligible_n", "event_n", "death_competing_n"]].merge(
            ref[keys + ["eligible_n", "event_n", "death_competing_n"]],
            on=keys, how="outer", suffixes=("_xgb", "_crescent"), indicator=True,
        )
        if not merged["_merge"].eq("both").all():
            raise RuntimeError("Condition/horizon rows differ from the CRESCENT metrics table.")
        eligible_gap = (merged["eligible_n_xgb"] != merged["eligible_n_crescent"])
        if eligible_gap.any():
            raise RuntimeError(
                "Eligible risk sets differ from CRESCENT in "
                f"{int(eligible_gap.sum())} condition-horizon rows (first: "
                f"{merged[eligible_gap].head(3).to_dict('records')})."
            )
        event_gap = int((
            (merged["event_n_xgb"] != merged["event_n_crescent"])
            | (merged["death_competing_n_xgb"] != merged["death_competing_n_crescent"])
        ).sum())
        report["eligible_n_identical_to_crescent"] = True
        report["rows_with_event_count_difference"] = event_gap
        if event_gap:
            print(f"WARNING: {event_gap} condition-horizon rows have different event/competing "
                  "counts from CRESCENT; inspect before reporting.", flush=True)
    else:
        report["eligible_n_identical_to_crescent"] = None
        print("NOTE: CRESCENT test_per_condition_metrics_by_horizon.csv not found; "
              "risk-set comparison skipped (run CRESCENT evaluation, then re-run verification).",
              flush=True)
    return report


def verify_test_outputs() -> Path:
    required = {
        **TEST_PERFORMANCE_FILES,
        "prediction_manifest": PREDICTION_PATHS["manifest"],
        "prediction_row_key": PREDICTION_PATHS["row_key"],
        "raw_rate_auc": PREDICTION_PATHS["raw_rate_auc"],
        "sequence_summary": PREDICTION_PATHS["sequence_summary"],
        "model_bundle": MODEL_BUNDLE_PATH,
    }
    if CFG.workflow == "train_and_evaluate":
        required.update({
            "fit_summary": FIT_SUMMARY_PATH,
            "training_trace": TRAINING_TRACE_PATH,
            "top_gain_features": TOP_FEATURES_PATH,
        })
    missing = [
        str(path) for path in required.values()
        if not Path(path).is_file() or Path(path).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            "Required XGBoost survival artifacts are missing or empty:\n"
            + "\n".join(missing)
        )

    per_condition = pd.read_csv(required["per_condition"])
    required_metric_columns = {
        "model_name", "horizon_months", "risk_type",
        "condition_id", "condition_name", "eligible_n", "event_n",
        "death_competing_n", "auc_ipcw_cr", "brier_ipcw_ht",
        "observed_aj_cif", "mean_predicted_cif",
        "auroc", "auprc", "brier_ipcw", "nbll_ipcw", "mean_predicted_risk",
    }
    missing_columns = (
        required_metric_columns - set(per_condition.columns)
    )
    if missing_columns:
        raise RuntimeError(
            f"Per-condition output lacks columns: {missing_columns}"
        )
    observed_horizons = sorted(
        per_condition["horizon_months"].dropna().astype(int).unique()
    )
    if observed_horizons != sorted(CFG.report_horizons_months):
        raise RuntimeError("Metric output has the wrong horizon axis.")
    if set(per_condition["condition_name"]) != set(CONDITION_NAMES):
        raise RuntimeError("Metric output has the wrong condition axis.")

    with np.load(required["prediction_npz"], allow_pickle=True) as payload:
        if payload["eid"].astype(str).tolist() != (
            TEST_SAMPLES["eid"].astype(str).tolist()
        ):
            raise RuntimeError("Compatibility NPZ participant order differs.")
        if payload["condition_names"].astype(str).tolist() != CONDITION_NAMES:
            raise RuntimeError("Compatibility NPZ condition order differs.")
        for horizon in CFG.report_horizons_months:
            key = f"risk_death_adjusted_h{horizon}"
            expected_shape = (len(TEST_SAMPLES), K)
            if key not in payload or payload[key].shape != expected_shape:
                raise RuntimeError(
                    f"Compatibility NPZ is missing valid {key}."
                )

    reference_check = verify_against_crescent_reference(per_condition)

    completion = RUN_DIR / "TEST_EVALUATION_COMPLETE.json"
    atomic_json_dump({
        "status": "complete",
        "model": CFG.model_name,
        "schema_version": CFG.schema_version,
        "test_rows": len(TEST_SAMPLES),
        "conditions": K,
        "report_horizons_months": list(CFG.report_horizons_months),
        "crescent_reference_check": reference_check,
        "same_split_index_targets_and_schema": True,
        "test_used_for_model_selection": False,
        "required_artifacts": {
            key: {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for key, path in required.items()
        },
    }, completion)
    print({"verification": "passed", "completion": str(completion)})
    return completion

# --- Notebook cell 32 ---
# =============================================================================
# Participant-level bootstrap confidence intervals
# Additive long-form CI schema; frozen XGBoost survival model and train censoring KMs.
# =============================================================================

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
CI_MODEL_NAME = CFG.model_name


CI_POINT_METRICS = (
    "auc_ipcw_cr",
    "auprc_ipcw_cr",
    "brier_ipcw_ht",
    "brier_null_ipcw_ht",
    "scaled_brier_ipcw_ht",
    "nbll_ipcw_ht",
    "mean_predicted_cif",
    "observed_aj_cif",
    "calibration_in_large_pred_minus_obs",
    "observed_to_expected_ratio",
    "absolute_calibration_error",
)
CI_INTEGRATED_METRICS = (
    "ibs_ipcw_ht",
    "ibs_null_ipcw_ht",
    "scaled_ibs_ipcw_ht",
    "inbll_ipcw_ht",
    "iauc_ipcw_cr",
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
    Generate percentile CIs for every canonical scalar point/integrated metric.

    Test participants are resampled as clusters. The fitted XGBoost survival model,
    predictions, condition axis, and train-derived censoring KMs remain frozen.
    """
    dense_prediction_path = PREDICTION_PATHS["adjusted"]
    point_condition_path = (
        RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
    )
    dense_point_path = point_condition_path
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

    report_horizons = tuple(int(x) for x in CFG.ci_point_horizons_months)
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

    selected_raw = os.getenv("EHR_XGB_CI_CONDITION_IDS", "").strip()
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
        raise ValueError("EHR_XGB_CI_CONDITION_IDS contains duplicates.")
    invalid_conditions = [
        value
        for value in selected_condition_ids
        if value < 0 or value >= K
    ]
    if invalid_conditions:
        raise ValueError(
            "EHR_XGB_CI_CONDITION_IDS contains out-of-range values: "
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
        cindex_censor_event = (
            train_outcomes["status_untruncated"][selected_train, condition] == 0
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
                cindex_censor_event,
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
                test_outcomes["status_untruncated"][index, condition],
                exit_time_untruncated,
                tau_prediction,
                float(tau),
                condition_cindex_kms[condition],
            )
            cindex_event_n = int(
                (
                    eligible
                    & (test_outcomes["status_untruncated"][index, condition] == 1)
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
    print("XGBOOST SURVIVAL PARTICIPANT-BOOTSTRAP CONFIDENCE INTERVALS")
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
    cindex_reference = pd.read_csv(point_condition_path).query(
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
        "schema_version": "xgboost-survival-participant-bootstrap-ci.v2",
        "implementation_version": "six-month-lead-full-history-xgboost-canonical-a10-v7-cox-matched",
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
        "integrated_auroc_included": True,
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
    _atomic_csv(
        macro_ci,
        RUN_DIR / "test_bootstrap_summary_cis.csv",
    )
    _atomic_csv(
        macro_replicates,
        RUN_DIR / "test_bootstrap_replicates_summary.csv",
    )

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

    # Preserve the attached main model's compatibility filenames and base layout.
    publication_with_ci = pd.read_csv(
        RUN_DIR / "test_summary_publication.csv"
    )
    if "risk_type" in publication_with_ci.columns:
        publication_with_ci["ci_computed"] = publication_with_ci["risk_type"].astype(str).eq("death_adjusted")
    else:
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
            if str(row.get("risk_type", "death_adjusted")) != "death_adjusted":
                continue
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
        RUN_DIR / "test_summary_publication_with_ci.csv",
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
            "reference_cindex_file": str(point_condition_path),
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

    print("=" * 88)
    print("XGBOOST SURVIVAL PARTICIPANT-BOOTSTRAP CIs COMPLETE")
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

def run_model_test_evaluation(
    bundle: Mapping[str, Any],
    evaluation_mode: str,
) -> Dict[str, pd.DataFrame]:
    """Run frozen-bundle inference and preserve all existing point outputs."""
    global TEST_PREDICTIONS, TEST_RESULTS, TEST_COMPLETION_FILE
    prepare_test_context(bundle)
    TEST_PREDICTIONS = generate_predictions()
    TEST_RESULTS = evaluate_all_metrics(
        TEST_SAMPLES,
        TEST_PREDICTIONS,
    )
    save_raw_landmark_rate_auc()
    save_evaluation_outputs(
        TEST_RESULTS,
        TEST_SAMPLES,
        TEST_PREDICTIONS,
    )
    TEST_COMPLETION_FILE = verify_test_outputs()
    atomic_json_dump(
        {
            "schema_version": CFG.schema_version,
            "evaluation_mode": evaluation_mode,
            "model_bundle_path": str(MODEL_BUNDLE_PATH),
            "model_bundle_sha256": sha256_file(MODEL_BUNDLE_PATH),
            "completed_unix_time": time.time(),
        },
        RUN_DIR / "test_evaluation_mode.json",
    )
    return TEST_RESULTS


def run_full_results(
    bundle: Mapping[str, Any],
    evaluation_mode: str,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Generate every existing point result plus the complete additive CI set."""
    point_results = run_model_test_evaluation(
        bundle,
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
            "model_bundle_path": str(MODEL_BUNDLE_PATH),
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
