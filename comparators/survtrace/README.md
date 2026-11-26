# SurvTRACE-aligned

A SurvTRACE-style feature tokenizer and Transformer encoder with piecewise-exponential interval hazard heads, trained on the exact six-month delayed-entry post-entry likelihood (`1.0 * disease NLL + 0.10 * death NLL`) and selected by minimum validation joint NLL.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 2 | Loads shared CRESCENT task settings and sets SurvTRACE hyperparameters |
| 2 | `data.py` | 4, 6 | Verifies canonical train/validation splits and builds predictor tensors |
| 3 | `model.py` | 8 | Defines the feature tokenizer, Transformer encoder, interval hazard heads, and forward checks |
| 4 | `train.py` | 10 | Trains for 50,000 updates with EMA weights and selects the best validation NLL checkpoint |
| 5 | `evaluate.py` | 12, 14, 16, 18, 20 | Runs locked test scoring, dense-grid metrics, participant bootstrap, calibration export, and paired CRESCENT comparison |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` (plus CRESCENT's stage-7 bootstrap outputs for the paired comparison at the end of `evaluate.py`).
- **Extra packages:** none beyond the CRESCENT environment (`torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `numba`).
- **Hardware:** a single NVIDIA GPU (CUDA).

## Running

`SOURCE` is a fixed relative path (`runs/CRESCENT_Final`), so run from the repository root using a fresh output folder:

```bash
export EHR_SURVTRACE_OUTPUT=runs/survtrace
python run_comparator.py survtrace
```

For evaluation only on a completed checkpoint, or to resume training:

```bash
# Evaluation only
export EHR_SURVTRACE_WORKFLOW=evaluation_only
export EHR_SURVTRACE_CHECKPOINT=runs/survtrace/best_model_by_val_loss.pt
python run_comparator.py survtrace

# Resume interrupted training
export EHR_SURVTRACE_RESUME=runs/survtrace/last_checkpoint.pt
python run_comparator.py survtrace
```

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Encoder width (`d_model`) | `128` | `EHR_SURVTRACE_D_MODEL` |
| Transformer layers / heads | `3` / `8` | `EHR_SURVTRACE_N_LAYERS`, `EHR_SURVTRACE_N_HEADS` |
| Feed-forward width | `512` | `EHR_SURVTRACE_FF_DIM` |
| Dropout | Inherited from CRESCENT (`0.20`) | `EHR_SURVTRACE_DROPOUT` |
| Max feature tokens | `4096` | `EHR_SURVTRACE_MAX_FEATURE_TOKENS` |
| Updates / validation cadence | `50000` / every `1000` updates | `cfg.MAX_ITERATIONS`, `cfg.VALIDATION_INTERVAL` in `config.py` |
| Batch size, LR, weight decay, EMA | Inherited from `runs/CRESCENT_Final/config.json` | Loaded in `config.py` |
| Device | `auto` (`cuda:0` if available) | `EHR_DEVICE` |

## Outputs

Written under `runs/survtrace` (git-ignored; contains participant-level data and must not be committed):
- **Checkpoints:** `best_model_by_val_loss.pt`, `last_checkpoint.pt`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz`, `test_dense_prediction_arrays_for_integrated_metrics.npz`
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration & paired comparison:** `figure2_revised_inputs_2yr/`, paired CRESCENT delta tables

## Notes

- Must be run from the repository root so `runs/CRESCENT_Final` resolves.
- This comparator runs its own copy of the competing-risk evaluation code.
