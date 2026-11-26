# Delphi-2M-inspired — Runs the selected workflow (train-and-evaluate or evaluation-only)
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 33 ---
TRAIN_AND_EVALUATE_RESULTS: Optional[
    Dict[str, Dict[str, pd.DataFrame]]
] = None
if CFG.workflow == "train_and_evaluate":
    print({
        "workflow": CFG.workflow,
        "main_run_dir": str(MAIN_RUN_DIR),
        "output_dir": str(RUN_DIR),
        "train_rows": len(TRAIN_SAMPLES),
        "validation_rows": len(VAL_SAMPLES),
        "conditions": K,
        "report_horizons_months": CFG.report_horizons_months,
        "cindex_tau_months": CFG.ci_cindex_tau_months,
        "bootstrap_replicates": CFG.ci_bootstrap_n,
    }, flush=True)
    SELECTED_CHECKPOINT = train_model()
    TRAIN_AND_EVALUATE_RESULTS = run_full_results(
        SELECTED_CHECKPOINT,
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

# --- Notebook cell 35 ---
EVALUATION_ONLY_RESULTS: Optional[
    Dict[str, Dict[str, pd.DataFrame]]
] = None
if CFG.workflow == "evaluation_only":
    SELECTED_CHECKPOINT = resolved_checkpoint_path()
    print({
        "workflow": CFG.workflow,
        "checkpoint": str(SELECTED_CHECKPOINT),
        "main_run_dir": str(MAIN_RUN_DIR),
        "output_dir": str(RUN_DIR),
        "conditions": K,
        "report_horizons_months": CFG.report_horizons_months,
        "cindex_tau_months": CFG.ci_cindex_tau_months,
        "bootstrap_replicates": CFG.ci_bootstrap_n,
        "training_called": False,
    }, flush=True)
    EVALUATION_ONLY_RESULTS = run_full_results(
        SELECTED_CHECKPOINT,
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
