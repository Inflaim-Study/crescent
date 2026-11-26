# XGBoost

Gradient-boosted decision trees with a Cox partial-likelihood objective fitted per condition. It shares the same fixed full-history features, training-only Breslow baseline hazard, and evaluation conventions as the penalised Cox comparator so the comparison isolates the learner.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 1, 2, 4, 6 | Sets configuration, checks package versions, and defines hashing/persistence helpers |
| 2 | `data.py` | 8, 10, 12, 14 | Validates the CRESCENT input contract, builds Cox-matched full-history features, and constructs landmark competing-risk outcomes |
| 3 | `model.py` | 16, 18 | Defines the boosted-tree Cox objective, Optuna search space, default candidate, and numerical self-tests |
| 4 | `train.py` | 20 | Defines validation tuning, fitting, and bundle freezing |
| 5 | `evaluate.py` | 22, 24, 26, 28, 30, 32 | Defines test loading, death-adjusted CIF prediction, competing-risk metrics, tables, and participant bootstrap |
| 6 | `run.py` | 34, 36 | Executes the selected workflow (`train_and_evaluate` or `evaluation_only`) |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final`. XGBoost does **not** require a completed Cox run.
- **Extra packages:** `xgboost`, `optuna`, `scipy`, `joblib`, `numba`.
- **Hardware:** multi-threaded CPU (or GPU via `EHR_XGB_DEVICE`).

## Running

Stages 1–5 only define functions and contracts; training and evaluation execute in Stage 6 (`run.py`), so `--stages 1-4` does not train a model. Run from the repository root using a fresh output folder:

```bash
export EHR_MAIN_RUN_DIR=runs/CRESCENT_Final
export EHR_XGB_OUTPUT=runs/xgboost
python run_comparator.py xgboost
```

To run evaluation only on an existing frozen bundle, edit `WORKFLOW = "evaluation_only"` (and, if needed, `PRETRAINED_BUNDLE = "..."`) near the top of `config.py`, then run `python run_comparator.py xgboost`. Note that `config.py` overwrites `EHR_XGB_WORKFLOW` and `EHR_XGB_BUNDLE` from those two lines.

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Optuna tuning trials | `40` | `TUNING_TRIALS` / `EHR_XGB_TUNING_TRIALS` |
| Maximum boosting rounds | `10000` | `MAX_BOOST_ROUNDS` / `EHR_XGB_MAX_BOOST_ROUNDS` |
| Default candidate | Prespecified regularised parameter set always evaluated | `REGULARIZED_ANCHOR_PARAMS` in `model.py` |
| Tree method / device | `hist` / `cpu` | `EHR_XGB_TREE_METHOD`, `EHR_XGB_DEVICE` |
| CPU threads | `min(16, cpu_count)` | `EHR_XGB_NTHREAD` |
| Random seed | `42` | `EHR_SEED` |
| Optional Cox parity check | Unset (optional) | `EHR_COX_REFERENCE_OUTPUT` |

## Outputs

Written under `runs/xgboost` (git-ignored; contains participant-level data and must not be committed):
- **Frozen bundle:** `frozen_xgboost_survival_full_history_bundle.pkl`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz` and dense-grid prediction arrays
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_summary_publication.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration export:** `test_aj_calibration_by_horizon.csv`, `test_calibration_error_summary.csv`

## Notes

- Setting `EHR_COX_REFERENCE_OUTPUT` to a completed Cox run only verifies input and contract alignment; it is not required to train or evaluate XGBoost.
- This comparator runs its own copy of the competing-risk evaluation code.
