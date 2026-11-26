# Penalised Cox

A ridge-penalised linear Cox model fitted per condition using XGBoost's linear booster (`gblinear`) on fixed summaries of static covariates and full pre-index history, paired with a training-only Breslow baseline hazard to produce death-adjusted cumulative incidence across the 58 conditions on the six-month delayed-entry task.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 1, 3, 5, 7 | Sets configuration, checks package versions, and defines hashing/persistence helpers |
| 2 | `data.py` | 9, 11, 13, 15 | Validates the CRESCENT input contract, builds full-history tabular features, and constructs landmark competing-risk outcomes |
| 3 | `model.py` | 17, 19 | Defines the ridge-penalised linear Cox solver, Breslow baseline estimator, and numerical self-tests |
| 4 | `train.py` | 21 | Defines validation penalty selection, model fitting, and bundle freezing |
| 5 | `evaluate.py` | 23, 25, 28, 30, 32, 34 | Defines test loading, death-adjusted CIF prediction, competing-risk metrics, tables, and participant bootstrap |
| 6 | `run.py` | 36, 38 | Executes the selected workflow (`train_and_evaluate` or `evaluation_only`) |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` containing `prepared_data_source_audit.csv`, `prepared_splits_manifest.json`, canonical sample pickles, vocabularies, and manifests.
- **Extra packages:** `xgboost`, `scipy`, `joblib`, `numba`.
- **Hardware:** multi-threaded CPU.

## Running

Stages 1–5 only define functions and contracts; training and evaluation execute in Stage 6 (`run.py`), so `--stages 1-4` does not train a model. Run from the repository root using a fresh output folder:

```bash
export EHR_MAIN_RUN_DIR=runs/CRESCENT_Final
export EHR_COX_OUTPUT=runs/cox
python run_comparator.py cox
```

To run evaluation only on an existing frozen bundle, edit `WORKFLOW = "evaluation_only"` (and, if the bundle is outside the output folder, set `PRETRAINED_BUNDLE = "..."`) near the top of `config.py`, then run `python run_comparator.py cox`. Note that `config.py` overwrites `EHR_COX_WORKFLOW` and `EHR_COX_BUNDLE` from those two lines.

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Ridge penalty grid ($\lambda$) | `(0.01, 0.1, 1.0, 10.0)` (fallback `0.1`) | `RIDGE_LAMBDAS` / `EHR_COX_RIDGE_LAMBDAS`, `EHR_COX_DEFAULT_RIDGE_LAMBDA` |
| Training rounds | `100` | `COX_TRAINING_ROUNDS` / `EHR_COX_TRAINING_ROUNDS` |
| Learning rate | `0.5` | `EHR_COX_LEARNING_RATE` |
| Minimum events (train / val) | `10` / `20` | `EHR_COX_MIN_TRAIN_EVENTS`, `EHR_COX_MIN_VAL_EVENTS` |
| Random seed | `42` | `EHR_SEED` |
| CPU threads | `min(16, cpu_count)` | `COX_NTHREAD` / `EHR_COX_NTHREAD` |
| Bootstrap replicates / seed | `200` / `101` | `CI_BOOTSTRAP_N` / `EHR_COX_CI_BOOTSTRAP_N`, `EHR_COX_CI_BOOTSTRAP_SEED` |
| Concordance horizon ($	au$) | `60` months | `CINDEX_HORIZON_MONTHS` / `EHR_COX_CI_CINDEX_TAU_MONTHS` |

## Outputs

Written under `runs/cox` (git-ignored; contains participant-level data and must not be committed):
- **Frozen bundle:** `frozen_coxph_full_history_bundle.pkl`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz` and dense-grid prediction arrays
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_summary_publication.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration export:** `test_aj_calibration_by_horizon.csv`, `test_calibration_error_summary.csv`

## Notes

- Runs older than the current CRESCENT export (missing `prepared_data_source_audit.csv` or `prepared_splits_manifest.json`) are rejected.
- `EHR_CRESCENT_BASE_DIR` defaults to `.`; `EHR_MAIN_RUN_DIR` defaults to `runs/CRESCENT_Final`.
- This comparator runs its own copy of the competing-risk evaluation code.
