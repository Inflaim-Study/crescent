# Delphi-2M-inspired

A GPT-style causal autoregressive Transformer over the full pre-index event stream trained with next-event and exponential waiting-time likelihoods, then queried to produce landmark rates and death-adjusted cumulative incidence on the CRESCENT cohort. This is a local implementation inspired by [Delphi-2M](https://github.com/gerstung-lab/Delphi) (MIT-licensed code), adapted to the common six-month delayed-entry task rather than the released Delphi-2M model or weights.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 1, 2, 4, 6 | Reads CRESCENT `config.json`, validates horizons/grid, and sets model and precision options |
| 2 | `data.py` | 8, 10, 12, 14 | Loads canonical splits, constructs full event-stream sequences, and builds autoregressive datasets |
| 3 | `model.py` | 16, 18 | Defines the causal Transformer and next-event / exponential waiting-time likelihood |
| 4 | `train.py` | 19 | Defines 50,000-update training, EMA, and checkpoint selection |
| 5 | `predict.py` | 21 | Defines frozen test loading and landmark rate/CIF prediction |
| 6 | `evaluate.py` | 23, 25, 27, 29, 31 | Defines competing-risk metrics, dense-grid evaluation, publication tables, and participant bootstrap |
| 7 | `run.py` | 33, 35 | Executes the selected workflow (`train_and_evaluate` or `evaluation_only`) |
| 8 | `extras.py` | 37, 39 | Exports supported-horizon integrated summaries (12–84 and 12–96 months) and optional validation-recalibrated sensitivity |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` (must contain `config.json` declaring horizons `(12, 24, 60, 120)`, integration start `12`, and one-month-capped index).
- **Extra packages:** none beyond the CRESCENT environment (`torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `numba`).
- **Hardware:** a single NVIDIA GPU (CUDA).

## Running

Stages 1–6 only define functions and contracts; training and evaluation execute in Stage 7 (`run.py`), followed by optional summaries in Stage 8 (`extras.py`), so `--stages 1-4` does not train a model. Run from the repository root using a fresh output folder:

```bash
export EHR_MAIN_RUN_DIR=runs/CRESCENT_Final
export EHR_DELPHI_OUTPUT=runs/delphi_2m
python run_comparator.py delphi_2m
```

To run evaluation only on an existing checkpoint, edit `WORKFLOW = "evaluation_only"` (and, if the checkpoint is outside the output folder, set `PRETRAINED_CHECKPOINT = "..."`) near the top of `config.py`, then run `python run_comparator.py delphi_2m`. Note that `config.py` overwrites `EHR_DELPHI_WORKFLOW`, `EHR_DELPHI_CHECKPOINT`, `EHR_DEVICE`, and `EHR_SEED` from its top-level variables.

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Layers / heads / width / FF | `12` / `12` / `120` / `480` | `EHR_DELPHI_N_LAYERS`, `_N_HEADS`, `_D_MODEL`, `_D_FF` |
| Block size / dropout | `96` / `0.10` | `EHR_DELPHI_BLOCK_SIZE`, `EHR_DELPHI_DROPOUT` |
| Updates / batch size | `50000` / `128` | `EHR_DELPHI_MAX_ITERATIONS`, `EHR_BATCH_SIZE` |
| Learning rate / weight decay | `3e-4` / `0.1` | `EHR_LEARNING_RATE`, `EHR_WEIGHT_DECAY` |
| Warm-up / EMA decay | `1000` / `0.999` | `EHR_WARMUP_ITERATIONS`, `EHR_DELPHI_EMA_DECAY` |
| Resume checkpoint | Unset | `EHR_DELPHI_RESUME` |
| Recalibrated sensitivity sidecar | `YES` | `EHR_DELPHI_RUN_RECALIBRATED_SENSITIVITY` (Stage 8, `extras.py`) |

## Outputs

Written under `runs/delphi_2m` (git-ignored; contains participant-level data and must not be committed):
- **Checkpoints:** `best_checkpoint.pt`, `last_checkpoint.pt`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz` and dense-grid prediction arrays
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_summary_publication.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Extras (`extras.py`):** `test_integrated_macro_supported_windows_delphi.csv` and sidecar subfolder `validation_recalibrated_sensitivity/` (never replaces primary outputs)

## Notes

- Inspired by [Delphi-2M](https://github.com/gerstung-lab/Delphi) (MIT code licence); no original Delphi-2M weights are used.
- The notebook's disabled legacy head-to-head audit cell (cell 41) was omitted.
- This comparator runs its own copy of the competing-risk evaluation code.
