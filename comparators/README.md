# Comparators

The eight comparator models from the paper (§2.6). Each folder contains the code that produced the published results, split from the original notebooks into stage files that run in order in one shared namespace. Only machine-specific default paths were changed (along with generic environment wording in three comments/messages; Delphi's disabled legacy audit cell was omitted).

| Folder | Model | Status |
| :--- | :--- | :--- |
| `cox/` | Penalised Cox | Included |
| `xgboost/` | XGBoost (same features and baseline hazard as Cox) | Included |
| `survtrace/` | SurvTRACE-aligned | Included |
| `dynamic_deephit/` | Dynamic-DeepHit-inspired | Included |
| `motor_ft/` | MOTOR-FT | Fine-tuning included; pretraining to follow |
| `delphi_2m/` | Delphi-2M-inspired | Included |
| `gemma_e4b/` | Frozen Gemma 4 E4B | Decoder training & evaluation included; state extraction to follow |
| `crescent_pt/` | CRESCENT-PT | To follow |

## Running

All comparators read the inputs prepared by a completed CRESCENT run (stages 1–2 of `run_crescent.py`, written to `runs/CRESCENT_Final`), so run CRESCENT first. Run from the repository root:

```bash
python run_comparator.py --list            # available comparators
python run_comparator.py cox --list        # stages for one comparator
python run_comparator.py cox               # run all stages
```

Each folder's `README.md` lists its stages, settings, and environment variables. Use a new output folder for every run. Outputs are written under `runs/`, contain participant-level information, and must not be committed.

Each comparator currently runs its own copy of the evaluation code, as in the original analysis. A later release will score all models with one shared evaluator.
