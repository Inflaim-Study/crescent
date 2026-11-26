# Dynamic-DeepHit-inspired

Event embeddings processed by a unidirectional GRU with additive temporal attention, condition-specific attention, and a shared death head. It is trained on the exact cause-specific piecewise-exponential post-entry likelihood (`1.0 * disease NLL + 0.10 * death NLL`) with empirical post-entry rate priors and no ranking loss.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 2 | Configures source/output paths, workflow mode, and fixed hyperparameters |
| 2 | `data.py` | 5 | Builds shared predictors, delayed-entry targets, and dataloaders |
| 3 | `model.py` | 7 | Defines event embeddings, GRU, attention readers, death head, and post-entry likelihood |
| 4 | `train.py` | 9, 11, 13 | Defines full-validation scoring, trains for 50,000 updates, and exports test predictions |
| 5 | `evaluate.py` | 15, 17, 19, 21, 23, 25, 27 | Reloads saved predictions, computes locked test metrics, dense-grid metrics, bootstrap CIs, calibration exports, and paired CRESCENT comparison |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` (overridable via `EHR_MAIN_RUN_DIR`).
- **Extra packages:** none beyond the CRESCENT environment (`torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `numba`).
- **Hardware:** a single NVIDIA GPU (CUDA) for training; CPU or GPU for evaluation.

## Running

Run from the repository root using a fresh output folder:

```bash
export EHR_MAIN_RUN_DIR=runs/CRESCENT_Final
export EHR_DDH_OUTPUT=runs/dynamic_deephit
python run_comparator.py dynamic_deephit
```

For evaluation only on a completed run (skips training in `train.py`, regenerates/loads predictions from `best_model_by_val_loss.pt`, and runs all evaluation stages), or to resume training:

```bash
# Evaluation only (run all stages from stage 1)
export EHR_DDH_WORKFLOW=evaluation_only
python run_comparator.py dynamic_deephit

# Resume interrupted training
export EHR_DDH_RESUME=runs/dynamic_deephit/last_checkpoint.pt
python run_comparator.py dynamic_deephit
```

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Event embedding / GRU hidden width | `256` / `512` | `model.py` |
| Updates / batch size | `50000` / `128` | `MAX_UPDATES`, `BATCH_SIZE` in `config.py` |
| Peak LR / cosine floor / weight decay | `5e-5` / `5e-6` / `0.01` | `LR`, `LR_FLOOR`, `WEIGHT_DECAY` in `config.py` |
| Warm-up / validation NLL interval | `1000` / `1000` updates | `WARMUP`, `LOSS_INTERVAL` in `config.py` |
| Dropout | `0.10` | `model.py` |
| Random seed | `42` | `EHR_SEED` |
| Device | `cuda:0` if available, else `cpu` | `EHR_DEVICE` |
| Figure export overrides | Optional | `FIG2_COLOR_KEY_SOURCE`, `FIG2_COMPARATOR_METRICS_PATH` |

## Outputs

Written under `runs/dynamic_deephit` (git-ignored; contains participant-level data and must not be committed):
- **Checkpoints:** `best_model_by_val_loss.pt`, `last_checkpoint.pt`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz`, `test_dense_prediction_arrays_for_integrated_metrics.npz`
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration & paired comparison:** `figure2_revised_inputs_2yr/`, paired CRESCENT delta tables

## Notes

- Always invoke `run_comparator.py dynamic_deephit` from Stage 1 (using `EHR_DDH_WORKFLOW=evaluation_only` to skip training), because stages share one namespace.
- This comparator runs its own copy of the competing-risk evaluation code.
