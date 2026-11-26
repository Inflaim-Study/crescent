# XGBoost — Defines validation tuning, fitting, and bundle freezing
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 20 ---
MODEL_BUNDLE_PATH = (
    Path(CFG.bundle_path).expanduser().resolve()
    if CFG.bundle_path
    else RUN_DIR / "frozen_xgboost_survival_full_history_bundle.pkl"
)
TRAINING_TRACE_PATH = RUN_DIR / "xgboost_survival_tuning_trace.csv"
FIT_SUMMARY_PATH = RUN_DIR / "xgboost_survival_fit_summary.csv"
TOP_FEATURES_PATH = RUN_DIR / "xgboost_survival_top_gain_features.csv"


def fit_endpoint(
    train_outcome: Mapping[str, np.ndarray],
    validation_outcome: Mapping[str, np.ndarray],
    condition: Optional[int],
    endpoint_name: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if condition is None:
        train_eligible = np.asarray(
            train_outcome["eligible"], dtype=bool
        )
        validation_eligible = np.asarray(
            validation_outcome["eligible"], dtype=bool
        )
        train_status = np.asarray(train_outcome["status"], dtype=np.int8)
        validation_status = np.asarray(
            validation_outcome["status"], dtype=np.int8
        )
        train_time = np.asarray(
            train_outcome["exit_time"], dtype=np.float64
        )
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

    return fit_xgb_survival_search(
        train_matrix=X_TRAIN[train_eligible],
        train_time=train_time[train_eligible],
        train_event=(train_status[train_eligible] == 1),
        validation_matrix=X_VAL[validation_eligible],
        validation_time=validation_time[validation_eligible],
        validation_event=(validation_status[validation_eligible] == 1),
        endpoint_name=endpoint_name,
    )


def top_gain_rows(
    model: Mapping[str, Any],
    endpoint_type: str,
    condition_id: Optional[int],
    condition_name: str,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    booster = booster_from_raw(model.get("booster_raw"))
    if booster is None:
        return []
    gain = booster.get_score(importance_type="gain")
    parsed: List[Tuple[int, float]] = []
    for key, value in gain.items():
        if not key.startswith("f") or not key[1:].isdigit():
            continue
        feature_index = int(key[1:])
        if not 0 <= feature_index < len(FEATURE_NAMES):
            raise RuntimeError("Booster feature index is outside the axis.")
        parsed.append((feature_index, float(value)))
    selected = sorted(parsed, key=lambda item: -item[1])[:limit]
    rows: List[Dict[str, Any]] = []
    for rank, (feature_index, value) in enumerate(selected, start=1):
        rows.append({
            "model_name": CFG.model_name,
            "endpoint_type": endpoint_type,
            "condition_id": condition_id,
            "condition_name": condition_name,
            "rank_by_gain": rank,
            "feature_index": int(feature_index),
            "feature_name": FEATURE_NAMES[int(feature_index)],
            "mean_gain": float(value),
        })
    return rows


def validate_frozen_bundle_contract(bundle: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "condition_names", "condition_models", "death_model",
        "feature_builder", "raw_feature_names", "feature_names", "feature_mask",
        "raw_feature_axis_sha256", "feature_axis_sha256", "feature_mask_sha256",
        "canonical_train_samples_sha256", "canonical_val_samples_sha256",
        "condition_axis_sha256", "config", "test_used_for_model_selection",
        "history_contract", "feature_scale", "feature_scale_sha256",
        "tree_input_zero_semantics", "baseline_margin_contract", "evaluator_contract",
    }
    missing = sorted(required.difference(bundle))
    if missing:
        raise RuntimeError(f"Frozen bundle is missing keys: {missing}")
    if bundle["schema_version"] != CFG.schema_version:
        raise RuntimeError("Frozen bundle schema differs from this notebook.")
    if bundle["condition_names"] != CONDITION_NAMES:
        raise RuntimeError("Frozen bundle has a different condition axis.")
    if bundle["raw_feature_names"] != RAW_FEATURE_NAMES:
        raise RuntimeError("Frozen bundle has a different raw feature axis.")
    if bundle["feature_names"] != FEATURE_NAMES:
        raise RuntimeError("Frozen bundle has a different feature axis.")
    if not np.array_equal(
        np.asarray(bundle["feature_mask"], dtype=bool), FEATURE_MASK
    ):
        raise RuntimeError("Frozen bundle has a different training-only feature filter.")
    if not np.array_equal(np.asarray(bundle["feature_scale"], dtype=np.float64), FEATURE_SCALE):
        raise RuntimeError("Frozen bundle has different training-only feature scales.")
    if bundle["feature_scale_sha256"] != sha256_array(FEATURE_SCALE):
        raise RuntimeError("Frozen bundle feature-scale hash is invalid.")
    if bundle["tree_input_zero_semantics"] != "dense zero-filled numeric summaries":
        raise RuntimeError("Frozen tree input zero semantics differ.")
    if bundle["baseline_margin_contract"] != "Cox V6 training-max centering; prediction clip [-30,30]":
        raise RuntimeError("Frozen baseline centering contract differs.")
    if bundle["evaluator_contract"] != "Cox V6 scalar metrics and support rules":
        raise RuntimeError("Frozen evaluator contract differs.")
    for model in list(bundle["condition_models"]) + [bundle["death_model"]]:
        if "margin_center" not in model or not np.isfinite(float(model["margin_center"])):
            raise RuntimeError("Frozen endpoint lacks a finite baseline margin center.")
    if bundle["raw_feature_axis_sha256"] != canonical_json_hash(RAW_FEATURE_NAMES):
        raise RuntimeError("Frozen bundle raw feature hash is invalid.")
    if bundle["feature_axis_sha256"] != canonical_json_hash(FEATURE_NAMES):
        raise RuntimeError("Frozen bundle feature hash is invalid.")
    if bundle["feature_mask_sha256"] != sha256_array(FEATURE_MASK.astype(np.uint8)):
        raise RuntimeError("Frozen bundle feature-mask hash is invalid.")
    if bundle["condition_axis_sha256"] != canonical_json_hash(CONDITION_NAMES):
        raise RuntimeError("Frozen bundle condition hash is invalid.")
    if bundle.get("test_used_for_model_selection"):
        raise RuntimeError("Frozen bundle reports test-set model selection.")
    if bundle.get("history_contract") != "all canonical pre-index events":
        raise RuntimeError("Frozen bundle is not a full-history model.")
    if (
        bundle["canonical_train_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["train"])
        or bundle["canonical_val_samples_sha256"]
        != sha256_file(CANONICAL_PATHS["val"])
    ):
        raise RuntimeError("Frozen bundle was fitted from different canonical samples.")

    bundle_config = bundle.get("config", {})
    contract_checks = {
        "gap_months": int(CFG.gap_months),
        "report_horizons_months": tuple(CFG.report_horizons_months),
        "integration_start_months": int(CFG.integration_start_months),
        "ci_point_horizons_months": tuple(CFG.ci_point_horizons_months),
        "min_event_feature_nonzero": int(CFG.min_event_feature_nonzero),
        "min_calibration_group_n": int(CFG.min_calibration_group_n),
        "min_calibration_bin_n": int(CFG.min_calibration_bin_n),
        "include_extended_history_features": bool(
            CFG.include_extended_history_features
        ),
    }
    for key, expected in contract_checks.items():
        observed = bundle_config.get(key)
        if key == "report_horizons_months" and observed is not None:
            observed = tuple(int(value) for value in observed)
        if key == "ci_point_horizons_months" and observed is not None:
            observed = tuple(int(value) for value in observed)
        if observed != expected:
            raise RuntimeError(
                f"Frozen bundle contract mismatch for {key}: "
                f"observed={observed!r}, expected={expected!r}."
            )

    optional_hashes = {
        "target_metadata_sha256": CANONICAL_PATHS["target_metadata"],
        "vocab_scalers_sha256": CANONICAL_PATHS["vocab_scalers"],
        "input_manifest_sha256": CANONICAL_PATHS["input_manifest"],
        "target_config_sha256": CANONICAL_PATHS["target_config"],
        "source_config_sha256": CANONICAL_PATHS["source_config"],
        "dataloader_metadata_sha256": CANONICAL_PATHS["dataloader_metadata"],
    }
    for key, path in optional_hashes.items():
        observed = bundle.get(key)
        if observed is not None and observed != sha256_file(path):
            raise RuntimeError(f"Frozen bundle contract mismatch for {key}.")


def train_and_freeze_bundle() -> Dict[str, Any]:
    if MODEL_BUNDLE_PATH.is_file() and not CFG.overwrite_model:
        print(
            "Frozen XGBoost bundle exists; loading it. Set "
            "EHR_XGB_OVERWRITE_MODEL=YES to refit."
        )
        with MODEL_BUNDLE_PATH.open("rb") as handle:
            bundle = pickle.load(handle)
        validate_frozen_bundle_contract(bundle)
        return bundle

    condition_models: List[Dict[str, Any]] = []
    all_trace: List[Dict[str, Any]] = []
    fit_summary: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []

    for condition, condition_name in enumerate(CONDITION_NAMES):
        print(
            f"[{condition + 1}/{K}] fitting condition XGBoost survival: "
            f"{condition_name}"
        )
        model, trace = fit_endpoint(
            TRAIN_CONDITION_OUTCOMES,
            VAL_CONDITION_OUTCOMES,
            condition=condition,
            endpoint_name=f"condition::{condition_name}",
        )
        default_validation_cindex = float(
            model.get("v4_baseline_validation_cindex", np.nan)
        )
        selected_validation_cindex = float(model.get("validation_cindex", np.nan))
        if (
            np.isfinite(default_validation_cindex)
            and np.isfinite(selected_validation_cindex)
            and selected_validation_cindex < default_validation_cindex - 1e-12
        ):
            raise AssertionError(
                f"{condition_name}: selected model is worse than the exact V4 anchor "
                f"({selected_validation_cindex} < {default_validation_cindex})."
            )

        condition_models.append(model)
        all_trace.extend(trace)
        fit_summary.append({
            "model_name": CFG.model_name,
            "endpoint_type": "condition",
            "condition_id": int(condition),
            "condition_name": condition_name,
            **{
                key: model[key] for key in (
                    "fit_status", "selected_variant", "training_n",
                    "training_events", "validation_n",
                    "validation_events", "training_cindex",
                    "validation_cindex", "boosting_rounds",
                    "best_iteration", "early_stopping_rounds",
                    "required_validation_cindex_gain",
                    "selected_validation_cindex_gain", "elapsed_seconds",
                )
            },
            "parameters_json": json.dumps(
                model["parameters"], sort_keys=True
            ),
            "baseline_event_times": int(
                len(model["baseline_times_months"])
            ),
        })
        top_rows.extend(top_gain_rows(
            model, "condition", condition, condition_name
        ))

    print("Fitting shared death XGBoost survival model")
    death_model, death_trace = fit_endpoint(
        TRAIN_DEATH_OUTCOMES,
        VAL_DEATH_OUTCOMES,
        condition=None,
        endpoint_name="death",
    )
    default_death_cindex = float(
        death_model.get("v4_baseline_validation_cindex", np.nan)
    )
    selected_death_cindex = float(death_model.get("validation_cindex", np.nan))
    if (
        np.isfinite(default_death_cindex)
        and np.isfinite(selected_death_cindex)
        and selected_death_cindex < default_death_cindex - 1e-12
    ):
        raise AssertionError(
            "Death model selected below the exact V4-anchor validation C-index "
            f"({selected_death_cindex} < {default_death_cindex})."
        )

    all_trace.extend(death_trace)
    fit_summary.append({
        "model_name": CFG.model_name,
        "endpoint_type": "death",
        "condition_id": None,
        "condition_name": "Death",
        **{
            key: death_model[key] for key in (
                "fit_status", "selected_variant", "training_n",
                "training_events", "validation_n",
                "validation_events", "training_cindex",
                "validation_cindex", "boosting_rounds",
                "best_iteration", "elapsed_seconds", "margin_center",
            )
        },
        "parameters_json": json.dumps(
            death_model["parameters"], sort_keys=True
        ),
        "baseline_event_times": int(
            len(death_model["baseline_times_months"])
        ),
    })
    top_rows.extend(top_gain_rows(
        death_model, "death", None, "Death"
    ))

    pd.DataFrame(all_trace).to_csv(TRAINING_TRACE_PATH, index=False)
    pd.DataFrame(fit_summary).to_csv(FIT_SUMMARY_PATH, index=False)
    pd.DataFrame(
        top_rows,
        columns=[
            "model_name", "endpoint_type", "condition_id",
            "condition_name", "rank_by_gain", "feature_index",
            "feature_name", "mean_gain",
        ],
    ).to_csv(TOP_FEATURES_PATH, index=False)

    bundle = {
        "schema_version": CFG.schema_version,
        "model_name": CFG.model_name,
        "config": asdict(CFG),
        "condition_names": CONDITION_NAMES,
        "condition_models": condition_models,
        "death_model": death_model,
        "feature_builder": FEATURE_BUILDER,
        "raw_feature_names": RAW_FEATURE_NAMES,
        "feature_names": FEATURE_NAMES,
        "feature_mask": FEATURE_MASK,
        "feature_scale": FEATURE_SCALE,
        "feature_scale_sha256": sha256_array(FEATURE_SCALE),
        "tree_input_zero_semantics": "dense zero-filled numeric summaries",
        "baseline_margin_contract": "Cox V6 training-max centering; prediction clip [-30,30]",
        "evaluator_contract": "Cox V6 scalar metrics and support rules",
        "source_gap_time_origin_audit": SOURCE_GAP_AUDIT,
        "cox_reference_alignment": COX_REFERENCE_ALIGNMENT,
        "raw_feature_axis_sha256": canonical_json_hash(RAW_FEATURE_NAMES),
        "feature_axis_sha256": canonical_json_hash(FEATURE_NAMES),
        "feature_mask_sha256": sha256_array(FEATURE_MASK.astype(np.uint8)),
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
        "booster": "gbtree",
        "objective": "survival:cox",
        "model_family": "boosted-tree cause-specific Cox proportional hazards",
        "feature_filter": "training-only grouped event-frequency filter",
        "scaling": "training-only StandardScaler(with_mean=False)",
        "robust_search_space": ROBUST_SEARCH_SPACE,
        "v4_baseline_parameters": V4_BASELINE_PARAMS,
        "regularized_anchor_parameters": REGULARIZED_ANCHOR_PARAMS,
        "selection_metric": "validation Harrell C-index",
        "early_stopping_metric": "validation cox-nloglik",
        "v4_baseline_competes_in_selection": True,
        "non_regression_gain_guard": False,
        "minimum_validation_events_for_tuning": int(
            CFG.min_validation_events_to_select
        ),
        "last_measured_value_definition": (
            "most recent occurrence with an observed finite numeric value"
        ),
        "baseline_hazard_estimator": "training-only centered Breslow, Cox V6 convention",
        "xgboost_version": xgb.__version__,
        "optuna_version": optuna.__version__,
        "test_used_for_model_selection": False,
        "created_unix_time": time.time(),
    }
    validate_frozen_bundle_contract(bundle)
    atomic_pickle_dump(bundle, MODEL_BUNDLE_PATH)
    atomic_json_dump({
        key: value
        for key, value in bundle.items()
        if key not in {
            "condition_models", "death_model", "feature_builder",
            "raw_feature_names", "feature_names", "feature_mask", "feature_scale",
        }
    }, RUN_DIR / "frozen_xgboost_survival_full_history_bundle_manifest.json")
    print({
        "frozen_bundle": str(MODEL_BUNDLE_PATH),
        "sha256": sha256_file(MODEL_BUNDLE_PATH),
        "test_loaded": False,
    })
    return bundle


def load_frozen_bundle() -> Dict[str, Any]:
    """Load and validate a pretrained model bundle without fitting."""
    require_files([MODEL_BUNDLE_PATH])
    with MODEL_BUNDLE_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    validate_frozen_bundle_contract(bundle)
    print(
        f"Evaluation only: loaded frozen bundle {MODEL_BUNDLE_PATH}",
        flush=True,
    )
    return bundle


FROZEN_BUNDLE: Optional[Dict[str, Any]] = None
