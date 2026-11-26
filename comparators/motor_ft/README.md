# MOTOR-FT

Stage-2 fine-tuning of a locally pretrained MOTOR encoder (6-block local-causal Transformer, width 768, 12 heads, rotary age encoding) paired with a newly initialised factorised 22-interval competing-risk survival head. Only the encoder weights transfer from stage 1; the stage-1 8-bin projection is discarded.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 2 | Loads shared CRESCENT task settings and MOTOR architecture constants |
| 2 | `data.py` | 4, 6 | Verifies canonical train/validation splits and constructs predictors |
| 3 | `model.py` | 8 | Defines the MOTOR backbone, 22-bin head, stage-1 provenance checks and weight transfer, and forward checks |
| 4 | `train.py` | 10 | Fine-tunes for 50,000 updates on the post-entry likelihood with EMA and validation NLL selection |
| 5 | `evaluate.py` | 12, 14, 16, 18, 20 | Runs locked test scoring, dense-grid metrics, participant bootstrap, calibration export, and paired CRESCENT comparison |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` and a completed stage-1 MOTOR pretraining directory (`MOTOR_PRETRAIN_RUN_DIR`, default `runs/motor_pretrain`).
- **Extra packages:** none beyond the CRESCENT environment (`torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `numba`).
- **Hardware:** a single NVIDIA GPU (CUDA).

## Running

`SOURCE` is a fixed relative path (`runs/CRESCENT_Final`), so run from the repository root using a fresh output folder:

```bash
export MOTOR_PRETRAIN_RUN_DIR=runs/motor_pretrain
export MOTOR_FINETUNE_RUN_DIR=runs/motor_ft
python run_comparator.py motor_ft
```

Additional modes:

```bash
# Evaluation only on a completed fine-tuned checkpoint
export MOTOR_WORKFLOW=evaluation_only
python run_comparator.py motor_ft

# Resume interrupted fine-tuning
export MOTOR_RESUME_CHECKPOINT=runs/motor_ft/last_checkpoint.pt
python run_comparator.py motor_ft

# Random-initialisation control (use a separate output folder)
export MOTOR_INIT=scratch
export MOTOR_FINETUNE_RUN_DIR=runs/motor_scratch
python run_comparator.py motor_ft
```

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Backbone (layers / width / heads / FF) | `6` / `768` / `12` / `3072` | `CFG` in `config.py` |
| Attention width / dropout / survival dim | `26` / `0.10` / `256` | `CFG` in `config.py` |
| Updates / validation NLL cadence | `50000` / every `1000` updates | `cfg.MAX_ITERATIONS`, `cfg.VALIDATION_INTERVAL` |
| Monitoring AUROC cadence | Every `5000` updates (monitoring only) | `cfg.VAL_PERFORMANCE_INTERVAL` |
| Batch size, LR, weight decay, EMA | Inherited from `runs/CRESCENT_Final/config.json` | Loaded in `config.py` |
| Stage-1 checkpoint override | `MOTOR_PRETRAIN_RUN_DIR` (or `MOTOR_PRETRAIN_CHECKPOINT`, `MOTOR_PRETRAIN_SOURCE_DIR`) | `model.py` |
| Device | `auto` (`cuda:0` if available) | `EHR_DEVICE` |

## Outputs

Written under `runs/motor_ft` (git-ignored; contains participant-level data and must not be committed):
- **Checkpoints:** `best_model_by_val_loss.pt`, `last_checkpoint.pt`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz`, `test_dense_prediction_arrays_for_integrated_metrics.npz`
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration & paired comparison:** `figure2_revised_inputs_2yr/`, paired CRESCENT delta tables

## Notes

- **Stage-1 pretraining code is not yet included** (`pretrain.py` is a placeholder). A completed stage-1 checkpoint is required unless running the `MOTOR_INIT=scratch` control.
- Training and validation provenance are verified, and test participants are excluded, before transferring stage-1 weights.
- Must be run from the repository root so `runs/CRESCENT_Final` resolves.
- Stage-1 pretraining represents additional compute beyond the 50,000 fine-tuning updates.
- This comparator runs its own copy of the competing-risk evaluation code.
