# Data preparation

Builds the prepared files that `run_crescent.py` and every comparator read. Requires approved UK Biobank
access. Paper environment: Python 3.11.4, Polars 1.37.1.

| Step | File | Output |
|---|---|---|
| 1 | `extract.py` | Primary-care records, recruitment covariates and death dates |
| 2 | `build_cohort.py` | Diagnosis histories, baseline predictor events and outcomes |
| 3 | `make_splits.py` | `train.csv`, `val.csv`, `test.csv`, `splits_manifest.json`, `static_covariate_vocabulary.json` |

Output folder: `splits_229950_one_month_cap/` (never commit it). Field definitions: see `docs/model.md`.

Expected counts: 229,950 participants; training 161,020 (14 centres), validation 22,962 (6 centres),
test 45,968 (7 centres); 58 fitted conditions.
