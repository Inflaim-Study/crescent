# Gemma 4 E4B (frozen encoder)

Frozen Gemma 4 E4B comparator paired with the matched CRESCENT decoder. Event-aligned hidden states are extracted once from each participant's serialised pre-landmark clinical history and cached on disk; this folder trains and evaluates the CRESCENT decoder on those cached states while keeping all Gemma weights frozen.

## Stages

| # | File | Notebook cells | What it does |
| :--- | :--- | :--- | :--- |
| 1 | `config.py` | 2, 4 | Configures paths, phase control, lock-before-test protocol, and parity checks against CRESCENT |
| 2 | `data.py` | 6, 7, 9, 12 | Audits canonical train/validation splits, builds datasets, and attaches cached frozen Gemma states |
| 3 | `model.py` | 14, 16 | Instantiates the matched CRESCENT decoder with `GemmaSequenceAdapter` and checkpoint identity helpers |
| 4 | `train.py` | 18, 19, 21, 23 | Trains or resumes the decoder, locks validation-only selection, and verifies locks before test loading |
| 5 | `evaluate.py` | 25, 27, 31, 33, 36, 39 | Runs locked test evaluation, dense-grid metrics, participant bootstrap, and calibration export |

## Requirements

- **Inputs:** a completed CRESCENT run in `runs/CRESCENT_Final` (`GEMMA_CRESCENT_RUN_DIR`) and a completed frozen-state cache directory (`GEMMA_CACHE_DIR`, default `runs/gemma_e4b/GEMMA4_E4B_cache_V7`).
- **Extra packages:** none beyond the CRESCENT environment (`torch`, `numpy`, `pandas`, `scikit-learn`, `joblib`, `numba`) for reading cached states; state extraction requires `transformers` with `Gemma4TextModel` and a local copy of Gemma 4 E4B weights.
- **Hardware:** a single NVIDIA GPU (CUDA).

## Running

Run from the repository root using a fresh output directory:

```bash
export GEMMA_CRESCENT_RUN_DIR=runs/CRESCENT_Final
export GEMMA_OUTPUT_ROOT=runs/gemma_e4b
export GEMMA_PHASE=all
python run_comparator.py gemma_e4b
```

Phase control via `GEMMA_PHASE`:
- `all` (default): trains seed `GEMMA_SEED` (or reuses a completed run), locks validation selection, then runs test evaluation (requires the test cache to exist).
- `train`: trains and validates without parsing the test split.
- `select`: performs validation-only selection locking (`GEMMA_SELECT_MODE=lock` or `grid`).
- `evaluate`: verifies selection locks and evaluates on the held-out test split.

To resume interrupted training, set `GEMMA_RESUME=1` with `GEMMA_PHASE=train`.

## Key settings

| Setting | Default | Where set |
| :--- | :--- | :--- |
| Cached layer | `final` (`final` or `twothirds`) | `GEMMA_LAYER` |
| Load cache into RAM | `1` (enabled; `0` uses memmap) | `GEMMA_CACHE_IN_RAM` |
| Seed / hyperparameter ID | `42` / `final` | `GEMMA_SEED`, `GEMMA_HPID` |
| GPU index selector | Unset (single visible GPU `cuda:0`) | `GEMMA_GPU` |
| Event masking & covariate dropout | `0.0` (disabled for cached states) | Fixed in `config.py` |
| Decoder & training recipe | Inherited from `GEMMA_CRESCENT_RUN_DIR/config.json` (overridable via `CRESCENT_*` / `A10_*`) | Loaded in `config.py` |
| Figure export overrides | Optional | `FIG2_COLOR_KEY_SOURCE`, `FIG2_COMPARATOR_METRICS_PATH` |

## Outputs

Written under `runs/gemma_e4b` (git-ignored; contains participant-level data and must not be committed):
- **Checkpoints & locks:** `best_model_by_val_loss.pt`, `last_model.pt`, `run_selection_locked.json`
- **Test predictions (participant-level):** `test_prediction_arrays_death_adjusted.npz`, `test_dense_prediction_arrays_for_integrated_metrics.npz`
- **Metric tables & bootstrap CIs:** `test_per_condition_metrics_by_horizon.csv`, `test_integrated_metrics_by_condition.csv`, `test_macro_metrics_with_ci.csv`, `test_bootstrap_replicates_metrics.csv.gz`
- **Calibration export:** `figure2_revised_inputs_2yr/`

## Notes

- **State extraction code is not yet included** (`embed.py` is a placeholder). A pre-extracted cache in `GEMMA_CACHE_DIR` is required to run these stages.
- Gemma 4 E4B weights are not included in this repository and are never downloaded by this code; retain the Apache-2.0 licence accompanying your local checkpoint.
- Event masking and covariate dropout are disabled because cached contextual states cannot support equivalent token masking.
- This comparator runs its own copy of the competing-risk evaluation code.
