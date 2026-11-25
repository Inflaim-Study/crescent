"""Copy the aggregate Fig. 2 inputs from `runs/` into `figures/figure2/source_data/`.

Run from anywhere after every model has been evaluated on the locked test set:

    python figures/figure2/collect_source_data.py

Model outputs are read from the repository's `runs/` folder (git-ignored), one folder per model
(see figures/README.md). Set CRESCENT_RUNS_DIR to read them from another location.

Only aggregate tables are copied: per-condition metrics, macro estimates with bootstrap CIs,
overall condition prevalence and the condition abbreviation key. The script stops if a table has
an `eid` column, and lists any count between 1 and 9 for review before anything is committed.
"""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path(os.environ.get("CRESCENT_RUNS_DIR", REPO / "runs")).expanduser().resolve()
DEST = HERE / "source_data"

# Model name used in plot.py -> run folder under runs/ (folder names follow comparators/).
MODEL_RUNS = {
    "Cox": "cox",
    "XGBoostSurvival": "xgboost",
    "Survtrace": "survtrace",
    "DDH": "dynamic_deephit",
    "Motor": "motor_ft",
    "Delphi-2M": "delphi_2m",
    "Gemma": "gemma_e4b",
    "CRESCENT-PT": "crescent_pt",
    "CRESCENT": "CRESCENT_Final",
}
MODEL_TABLES = (
    "test_per_condition_metrics_by_horizon.csv",
    "test_integrated_metrics_by_condition.csv",
    "test_cindex_by_condition_with_ci.csv",
    "test_macro_metrics_with_ci.csv",
)
EXTRA = {
    RUNS / "CRESCENT_Final" / "figure1_overall_condition_prevalence.csv":
        DEST / "figure1_overall_condition_prevalence.csv",
    RUNS / "metadata" / "con_abb.json": DEST / "Other_data" / "con_abb.json",
}


def check_csv(path):
    """Stop on participant identifiers; report small counts."""
    frame = pd.read_csv(path)
    if any(str(c).strip().lower() == "eid" for c in frame.columns):
        sys.exit(f"STOP: {path} has an eid column; it must not be published.")
    flagged = []
    for column in frame.columns:
        name = str(column).lower()
        if not (name.endswith("_n") or name in {"n", "n_pos", "count"}):
            continue
        small = int(pd.to_numeric(frame[column], errors="coerce").between(1, 9).sum())
        if small:
            flagged.append(f"{column}: {small} value(s) between 1 and 9")
    return flagged


def main():
    jobs = [(RUNS / run / table, DEST / model / table)
            for model, run in MODEL_RUNS.items() for table in MODEL_TABLES]
    jobs += list(EXTRA.items())

    missing = [str(src) for src, _ in jobs if not src.is_file()]
    if missing:
        sys.exit("Missing inputs:\n  " + "\n  ".join(missing))

    report = []
    for src, dst in jobs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.suffix == ".csv":
            report += [f"{dst.relative_to(HERE)}  {line}" for line in check_csv(dst)]
        print("copied", dst.relative_to(HERE))

    print(f"\n{len(jobs)} files copied to {DEST}")
    if report:
        print("\nSmall counts to review before committing:")
        print("\n".join("  " + line for line in report))
    else:
        print("No counts between 1 and 9.")


if __name__ == "__main__":
    main()
