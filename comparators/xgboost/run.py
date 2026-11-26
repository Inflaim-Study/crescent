# XGBoost — Runs the selected workflow (train-and-evaluate or evaluation-only)
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 34 ---
TRAIN_AND_EVALUATE_RESULTS: Optional[
    Dict[str, Dict[str, pd.DataFrame]]
] = None
if CFG.workflow == "train_and_evaluate":
    print({
        "workflow": CFG.workflow,
        "main_run_dir": str(MAIN_RUN_DIR),
        "output_dir": str(RUN_DIR),
        "conditions": K,
        "report_horizons_months": CFG.report_horizons_months,
        "cindex_tau_months": CFG.ci_cindex_tau_months,
        "bootstrap_replicates": CFG.ci_bootstrap_n,
    }, flush=True)
    SELECTED_BUNDLE = train_and_freeze_bundle()
    TRAIN_AND_EVALUATE_RESULTS = run_full_results(
        SELECTED_BUNDLE,
        evaluation_mode="train_and_evaluate",
    )
    print(
        f"Train-and-evaluate complete. Full results: {RUN_DIR}",
        flush=True,
    )
else:
    print(
        "Skipped train-and-evaluate cell (WORKFLOW is evaluation_only).",
        flush=True,
    )

# --- Notebook cell 36 ---
EVALUATION_ONLY_RESULTS: Optional[
    Dict[str, Dict[str, pd.DataFrame]]
] = None
if CFG.workflow == "evaluation_only":
    print({
        "workflow": CFG.workflow,
        "bundle": str(MODEL_BUNDLE_PATH),
        "main_run_dir": str(MAIN_RUN_DIR),
        "output_dir": str(RUN_DIR),
        "conditions": K,
        "report_horizons_months": CFG.report_horizons_months,
        "cindex_tau_months": CFG.ci_cindex_tau_months,
        "bootstrap_replicates": CFG.ci_bootstrap_n,
        "training_called": False,
    }, flush=True)
    SELECTED_BUNDLE = load_frozen_bundle()
    EVALUATION_ONLY_RESULTS = run_full_results(
        SELECTED_BUNDLE,
        evaluation_mode="evaluation_only",
    )
    print(
        f"Evaluation-only complete. Full results: {RUN_DIR}",
        flush=True,
    )
else:
    print(
        "Skipped evaluation-only cell (WORKFLOW is train_and_evaluate).",
        flush=True,
    )
