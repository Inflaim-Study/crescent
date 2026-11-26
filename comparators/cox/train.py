# Penalised Cox — Defines validation selection, fitting, and bundle freezing
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 21 ---
MODEL_BUNDLE_PATH = (
    Path(CFG.bundle_path).expanduser().resolve()
    if CFG.bundle_path
    else RUN_DIR / "frozen_coxph_full_history_bundle.pkl"
)
TRAINING_TRACE_PATH = RUN_DIR / "coxph_ridge_selection_trace.csv"
FIT_SUMMARY_PATH = RUN_DIR / "coxph_fit_summary.csv"
TOP_COEFFICIENTS_PATH = RUN_DIR / "coxph_top_absolute_coefficients.csv"


def fit_endpoint(
    train_outcome: Mapping[str, np.ndarray],
    validation_outcome: Mapping[str, np.ndarray],
    condition: Optional[int],
    endpoint_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if condition is None:
        train_eligible = np.asarray(train_outcome["eligible"], dtype=bool)
        validation_eligible = np.asarray(
            validation_outcome["eligible"], dtype=bool
        )
        train_status = np.asarray(train_outcome["status"], dtype=np.int8)
        validation_status = np.asarray(
            validation_outcome["status"], dtype=np.int8
        )
        train_time = np.asarray(train_outcome["exit_time"], dtype=np.float64)
        validation_time = np.asarray(
            validation_outcome["exit_time"], dtype=np.float64
        )
    else:
        train_eligible = np.asarray(
            train_outcome["eligible"][:, condition], dtype=bool
        )
        validation_eligible = np.asarray(
            validation_outcome["eligible"][:, condition], dtype=bool
        )
        train_status = np.asarray(
            train_outcome["status"][:, condition], dtype=np.int8
        )
        validation_status = np.asarray(
            validation_outcome["status"][:, condition], dtype=np.int8
        )
        train_time = np.asarray(
            train_outcome["exit_time"][:, condition], dtype=np.float64
        )
        validation_time = np.asarray(
            validation_outcome["exit_time"][:, condition], dtype=np.float64
        )

    return fit_linear_cox_search(
        train_matrix=X_TRAIN[train_eligible],
        train_time=train_time[train_eligible],
        train_event=(train_status[train_eligible] == 1),
        validation_matrix=X_VAL[validation_eligible],
        validation_time=validation_time[validation_eligible],
        validation_event=(validation_status[validation_eligible] == 1),
        endpoint_name=endpoint_name,
    )


def extract_linear_coefficients(model: Mapping[str, Any]) -> Optional[np.ndarray]:
    booster = booster_from_raw(model.get("booster_raw"))
    if booster is None:
        return None
    try:
        dumps = booster.get_dump(dump_format="json")
        payload = json.loads(dumps[0])
        coefficients = np.asarray(payload["weight"], dtype=np.float64).reshape(-1)
    except Exception as exc:
        warnings.warn(
            f"Could not extract gblinear coefficients: {exc}", RuntimeWarning
        )
        return None
    if coefficients.size != len(FEATURE_NAMES):
        warnings.warn(
            "gblinear coefficient count differs from the frozen feature axis; "
            "the fitted model is retained but the coefficient table is omitted.",
            RuntimeWarning,
        )
        return None
    return coefficients


def top_coefficient_rows(
    model: Mapping[str, Any],
    endpoint_type: str,
    condition_id: Optional[int],
    condition_name: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    coefficients_scaled = extract_linear_coefficients(model)
    if coefficients_scaled is None:
        return []
    coefficients_original = coefficients_scaled / FEATURE_SCALE
    order = np.argsort(-np.abs(coefficients_original), kind="mergesort")[:limit]
    rows: List[Dict[str, Any]] = []
    for rank, feature_index in enumerate(order, start=1):
        coefficient = float(coefficients_original[feature_index])
        rows.append({
            "model_name": CFG.model_name,
            "endpoint_type": endpoint_type,
            "condition_id": condition_id,
            "condition_name": condition_name,
            "rank_by_absolute_coefficient": int(rank),
            "feature_index": int(feature_index),
            "feature_name": FEATURE_NAMES[int(feature_index)],
            "coefficient_scaled_feature": float(
                coefficients_scaled[feature_index]
            ),
            "coefficient_original_feature_unit": coefficient,
            "hazard_ratio_per_original_unit": float(
                np.exp(np.clip(coefficient, -50.0, 50.0))
            ),
            "feature_training_scale": float(FEATURE_SCALE[feature_index]),
        })
    return rows


def validate_frozen_bundle_contract(bundle: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "condition_names", "condition_models", "death_model",
        "raw_feature_names", "feature_names", "feature_mask", "feature_scale",
        "raw_feature_axis_sha256", "feature_axis_sha256",
        "feature_mask_sha256", "feature_scale_sha256",
        "canonical_train_samples_sha256", "canonical_val_samples_sha256",
        "condition_axis_sha256", "config", "test_used_for_model_selection",
        "history_contract", "solver",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise RuntimeError(f"Frozen CoxPH bundle is missing keys: {missing}")
    if bundle["schema_version"] != CFG.schema_version:
        raise RuntimeError("Frozen CoxPH bundle schema differs from this notebook.")
    if bundle["condition_names"] != CONDITION_NAMES:
        raise RuntimeError("Frozen CoxPH bundle has a different condition axis.")
    if bundle["raw_feature_names"] != RAW_FEATURE_NAMES:
        raise RuntimeError("Frozen CoxPH bundle has a different raw feature axis.")
    if bundle["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Frozen CoxPH bundle has a different selected feature axis.")
    if not np.array_equal(
        np.asarray(bundle["feature_mask"], dtype=bool), FEATURE_MASK
    ):
        raise RuntimeError("Frozen CoxPH bundle has a different feature filter.")
    if not np.array_equal(
        np.asarray(bundle["feature_scale"], dtype=np.float64), FEATURE_SCALE
    ):
        raise RuntimeError("Frozen CoxPH bundle has different training-only scales.")
    hash_checks = {
        "raw_feature_axis_sha256": canonical_json_hash(RAW_FEATURE_NAMES),
        "feature_axis_sha256": canonical_json_hash(FEATURE_NAMES),
        "feature_mask_sha256": sha256_array(FEATURE_MASK.astype(np.uint8)),
        "feature_scale_sha256": sha256_array(FEATURE_SCALE),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
    }
    for key, expected in hash_checks.items():
        if bundle.get(key) != expected:
            raise RuntimeError(f"Frozen CoxPH bundle hash mismatch for {key}.")
    if bundle.get("test_used_for_model_selection"):
        raise RuntimeError("Frozen CoxPH bundle reports test-set model selection.")
    if bundle.get("history_contract") != "all canonical pre-index events":
        raise RuntimeError("Frozen CoxPH bundle is not a full-history model.")
    if bundle.get("solver") != "xgboost_gblinear_coord_descent":
        raise RuntimeError("Frozen CoxPH bundle uses a different optimizer contract.")
    if len(bundle["condition_models"]) != K:
        raise RuntimeError("Frozen CoxPH bundle has the wrong number of endpoints.")
    if (
        bundle["canonical_train_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["train"])
        or bundle["canonical_val_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["val"])
    ):
        raise RuntimeError(
            "Frozen CoxPH bundle was fitted from different canonical samples."
        )

    bundle_config = bundle.get("config", {})
    contract_checks: Dict[str, Any] = {
        "gap_months": int(CFG.gap_months),
        "report_horizons_months": tuple(CFG.report_horizons_months),
        "integration_start_months": int(CFG.integration_start_months),
        "ci_point_horizons_months": tuple(CFG.ci_point_horizons_months),
        "min_calibration_group_n": int(CFG.min_calibration_group_n),
        "min_calibration_bin_n": int(CFG.min_calibration_bin_n),
        "include_extended_history_features": bool(
            CFG.include_extended_history_features
        ),
        "min_event_feature_nonzero": int(CFG.min_event_feature_nonzero),
        "ridge_lambdas": tuple(float(value) for value in CFG.ridge_lambdas),
        "default_ridge_lambda": float(CFG.default_ridge_lambda),
        "training_rounds": int(CFG.training_rounds),
        "learning_rate": float(CFG.learning_rate),
    }
    for key, expected in contract_checks.items():
        observed = bundle_config.get(key)
        if key == "report_horizons_months" and observed is not None:
            observed = tuple(int(value) for value in observed)
        if key == "ridge_lambdas" and observed is not None:
            observed = tuple(float(value) for value in observed)
        if key == "ci_point_horizons_months" and observed is not None:
            observed = tuple(int(value) for value in observed)
        if observed != expected:
            raise RuntimeError(
                f"Frozen CoxPH bundle contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    optional_hashes = {
        "target_metadata_sha256": CANONICAL_PATHS["target_metadata"],
        "vocab_scalers_sha256": CANONICAL_PATHS["vocab_scalers"],
        "input_manifest_sha256": CANONICAL_PATHS["input_manifest"],
        "source_config_sha256": CANONICAL_PATHS["source_config"],
        "dataloader_metadata_sha256": CANONICAL_PATHS["dataloader_metadata"],
        "target_config_sha256": CANONICAL_PATHS["target_config"],
    }
    for key, path in optional_hashes.items():
        observed = bundle.get(key)
        if observed is not None and observed != sha256_file(path):
            raise RuntimeError(f"Frozen CoxPH bundle mismatch for {key}.")


def _fit_summary_row(
    model: Mapping[str, Any],
    endpoint_type: str,
    condition_id: Optional[int],
    condition_name: str,
) -> Dict[str, Any]:
    return {
        "model_name": CFG.model_name,
        "endpoint_type": endpoint_type,
        "condition_id": condition_id,
        "condition_name": condition_name,
        "fit_status": model["fit_status"],
        "solver": model["solver"],
        "training_n": model["training_n"],
        "training_events": model["training_events"],
        "validation_n": model["validation_n"],
        "validation_events": model["validation_events"],
        "training_cindex": model["training_cindex"],
        "validation_cindex": model["validation_cindex"],
        "ridge_lambda": model["ridge_lambda"],
        "training_rounds": model["training_rounds"],
        "elapsed_seconds": model["elapsed_seconds"],
        "baseline_event_times": int(len(model["baseline_times_months"])),
        "margin_center": model["margin_center"],
        "last_training_cox_nloglik": model["last_training_cox_nloglik"],
        "last_validation_cox_nloglik": model["last_validation_cox_nloglik"],
    }


def _model_manifest_summary(model: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: model.get(key)
        for key in (
            "fit_status", "solver", "training_n", "training_events",
            "validation_n", "validation_events", "training_cindex",
            "validation_cindex", "ridge_lambda", "training_rounds",
            "elapsed_seconds", "margin_center",
        )
    } | {
        "baseline_event_times": int(len(model["baseline_times_months"])),
        "fitted_linear_model": model.get("booster_raw") is not None,
    }


def train_and_freeze_bundle() -> Dict[str, Any]:
    if MODEL_BUNDLE_PATH.is_file() and not CFG.overwrite_model:
        print(
            "Frozen CoxPH bundle exists; loading it. Set "
            "EHR_COX_OVERWRITE_MODEL=YES to refit."
        )
        with MODEL_BUNDLE_PATH.open("rb") as handle:
            bundle = pickle.load(handle)
        validate_frozen_bundle_contract(bundle)
        return bundle

    condition_models: List[Dict[str, Any]] = []
    all_trace: List[Dict[str, Any]] = []
    fit_summary: List[Dict[str, Any]] = []
    coefficient_rows: List[Dict[str, Any]] = []

    for condition, condition_name in enumerate(CONDITION_NAMES):
        print(
            f"[{condition + 1}/{K}] fitting full-history linear CoxPH: "
            f"{condition_name}",
            flush=True,
        )
        model, trace = fit_endpoint(
            TRAIN_CONDITION_OUTCOMES,
            VAL_CONDITION_OUTCOMES,
            condition=condition,
            endpoint_name=f"condition::{condition_name}",
        )
        condition_models.append(model)
        all_trace.extend(trace)
        fit_summary.append(_fit_summary_row(
            model, "condition", int(condition), condition_name
        ))
        coefficient_rows.extend(top_coefficient_rows(
            model, "condition", int(condition), condition_name
        ))

    print("Fitting shared full-history death CoxPH model", flush=True)
    death_model, death_trace = fit_endpoint(
        TRAIN_DEATH_OUTCOMES,
        VAL_DEATH_OUTCOMES,
        condition=None,
        endpoint_name="death",
    )
    all_trace.extend(death_trace)
    fit_summary.append(_fit_summary_row(
        death_model, "death", None, "Death"
    ))
    coefficient_rows.extend(top_coefficient_rows(
        death_model, "death", None, "Death"
    ))

    pd.DataFrame(all_trace).to_csv(TRAINING_TRACE_PATH, index=False)
    pd.DataFrame(fit_summary).to_csv(FIT_SUMMARY_PATH, index=False)
    coefficient_columns = [
        "model_name", "endpoint_type", "condition_id", "condition_name",
        "rank_by_absolute_coefficient", "feature_index", "feature_name",
        "coefficient_scaled_feature", "coefficient_original_feature_unit",
        "hazard_ratio_per_original_unit", "feature_training_scale",
    ]
    pd.DataFrame(
        coefficient_rows, columns=coefficient_columns
    ).to_csv(TOP_COEFFICIENTS_PATH, index=False)

    bundle = {
        "schema_version": CFG.schema_version,
        "model_name": CFG.model_name,
        "config": asdict(CFG),
        "condition_names": CONDITION_NAMES,
        "condition_models": condition_models,
        "death_model": death_model,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_mask": FEATURE_MASK,
        "feature_scale": FEATURE_SCALE,
        "raw_feature_axis_sha256": canonical_json_hash(RAW_FEATURE_NAMES),
        "feature_axis_sha256": canonical_json_hash(FEATURE_NAMES),
        "feature_mask_sha256": sha256_array(FEATURE_MASK.astype(np.uint8)),
        "feature_scale_sha256": sha256_array(FEATURE_SCALE),
        "canonical_train_samples_sha256": sha256_file(
            CANONICAL_PATHS["train"]
        ),
        "canonical_val_samples_sha256": sha256_file(
            CANONICAL_PATHS["val"]
        ),
        "target_metadata_sha256": sha256_file(
            CANONICAL_PATHS["target_metadata"]
        ),
        "vocab_scalers_sha256": sha256_file(
            CANONICAL_PATHS["vocab_scalers"]
        ),
        "input_manifest_sha256": sha256_file(
            CANONICAL_PATHS["input_manifest"]
        ),
        "source_config_sha256": sha256_file(
            CANONICAL_PATHS["source_config"]
        ),
        "dataloader_metadata_sha256": sha256_file(
            CANONICAL_PATHS["dataloader_metadata"]
        ),
        "target_config_sha256": (
            sha256_file(CANONICAL_PATHS["target_config"])
            if CANONICAL_PATHS["target_config"].is_file() else None
        ),
        "train_eid_axis_sha256": canonical_json_hash(
            TRAIN_SAMPLES["eid"].astype(str).tolist()
        ),
        "val_eid_axis_sha256": canonical_json_hash(
            VAL_SAMPLES["eid"].astype(str).tolist()
        ),
        "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
        "history_contract": "all canonical pre-index events",
        "feature_filter": "training-only grouped event-frequency filter",
        "feature_scaling": "training-only sparse StandardScaler(with_mean=False)",
        "model_family": "ridge-penalized linear Cox proportional hazards",
        "solver": "xgboost_gblinear_coord_descent",
        "booster": "gblinear",
        "objective": "survival:cox",
        "feature_selector": "cyclic",
        "l1_penalty": 0.0,
        "selection_metric": "validation Harrell C-index",
        "minimum_validation_events_for_tuning": int(
            CFG.min_validation_events_to_select
        ),
        "baseline_hazard_estimator": "training-only centered Breslow",
        "xgboost_version": xgb.__version__,
        "test_used_for_model_selection": False,
        "created_unix_time": time.time(),
    }
    validate_frozen_bundle_contract(bundle)
    MODEL_BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_pickle_dump(bundle, MODEL_BUNDLE_PATH)

    bundle_manifest = {
        "schema_version": bundle["schema_version"],
        "model_name": bundle["model_name"],
        "model_family": bundle["model_family"],
        "solver": bundle["solver"],
        "booster": bundle["booster"],
        "objective": bundle["objective"],
        "history_contract": bundle["history_contract"],
        "config": bundle["config"],
        "condition_names": bundle["condition_names"],
        "raw_feature_count": len(RAW_FEATURE_NAMES),
        "selected_feature_count": len(FEATURE_NAMES),
        "raw_feature_axis_sha256": bundle["raw_feature_axis_sha256"],
        "feature_axis_sha256": bundle["feature_axis_sha256"],
        "feature_mask_sha256": bundle["feature_mask_sha256"],
        "feature_scale_sha256": bundle["feature_scale_sha256"],
        "canonical_train_samples_sha256": bundle[
            "canonical_train_samples_sha256"
        ],
        "canonical_val_samples_sha256": bundle[
            "canonical_val_samples_sha256"
        ],
        "condition_axis_sha256": bundle["condition_axis_sha256"],
        "condition_models": [
            _model_manifest_summary(model) for model in condition_models
        ],
        "death_model": _model_manifest_summary(death_model),
        "test_used_for_model_selection": False,
        "created_unix_time": bundle["created_unix_time"],
    }
    atomic_json_dump(
        bundle_manifest,
        RUN_DIR / "frozen_coxph_full_history_bundle_manifest.json",
    )
    print({
        "frozen_bundle": str(MODEL_BUNDLE_PATH),
        "sha256": sha256_file(MODEL_BUNDLE_PATH),
        "test_loaded": False,
    })
    return bundle


def load_frozen_bundle() -> Dict[str, Any]:
    """Load and validate a pretrained full-history CoxPH bundle without fitting."""
    require_files([MODEL_BUNDLE_PATH])
    with MODEL_BUNDLE_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    validate_frozen_bundle_contract(bundle)
    print(f"Evaluation only: loaded frozen CoxPH bundle {MODEL_BUNDLE_PATH}")
    return bundle


FROZEN_BUNDLE: Optional[Dict[str, Any]] = None
