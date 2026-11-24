# Comparators

Each comparator follows the same layout as the CRESCENT code, with one extra file where it has an extra stage.
All of them use the same prepared splits, condition order, delayed-entry risk sets and horizons, and are scored
with the shared code in `evaluation/`.

| Folder | Model | Extra stage |
|---|---|---|
| `cox/` | Penalised Cox | - |
| `xgboost/` | XGBoost | - |
| `survtrace/` | SurvTRACE-aligned | - |
| `dynamic_deephit/` | Dynamic-DeepHit-inspired | - |
| `motor_ft/` | MOTOR-FT | `pretrain.py` |
| `delphi_2m/` | Delphi-2M-inspired | `simulate.py` |
| `gemma_e4b/` | Frozen Gemma 4 E4B | `embed.py` |
| `crescent_pt/` | CRESCENT-PT | `pretrain.py` |

Each `train.py` writes its test predictions in the same format as CRESCENT
(`test_prediction_arrays_death_adjusted.npz`) so that `evaluation/` can score every model identically.
See `docs/comparators.md` for a short description of each model.
