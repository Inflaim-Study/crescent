"""Copy the Fig. 5 export from `runs/explainability/figure5/` into `figures/figure5/source_data/`.

    python figures/figure5/collect_source_data.py

The export (manifest.json and the CSVs it lists) is written by the Fig. 5 data generator in
`explainability/` into the repository's `runs/` folder (git-ignored; see figures/README.md).
Set CRESCENT_RUNS_DIR to read it from another location.

The four `fig5D_patient_*` tables describe two individual test participants (panels e, f) and
are never copied; plot.py draws panels a-d and the extended-data figures without them.

Absolute machine paths in manifest.json are reduced to file names; the file hashes that plot.py
checks are unchanged. The script stops if a table has an `eid` column, and lists any count between
1 and 9 for review before anything is committed.
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path(os.environ.get("CRESCENT_RUNS_DIR", REPO / "runs")).expanduser().resolve()
EXPORT = RUNS / "explainability" / "figure5"
DEST = HERE / "source_data"

PATIENT_FILES = {"fig5D_patient_profiles.csv", "fig5D_patient_shapley.csv",
                 "fig5D_patient_horizon_risks.csv", "fig5D_patient_timelines.csv"}


def restricted_files():
    return set(PATIENT_FILES)

_ABSOLUTE = re.compile(r"^(/|~|[A-Za-z]:[\\\\/]|\\\\\\\\)")


def strip_paths(value):
    if isinstance(value, dict):
        return {key: strip_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    if isinstance(value, str) and _ABSOLUTE.match(value):
        return Path(value.replace("\\", "/")).name
    return value


def check_csv(path):
    frame = pd.read_csv(path)
    if any(str(c).strip().lower() == "eid" for c in frame.columns):
        sys.exit(f"STOP: {path} has an eid column; it must not be published.")
    notes = []
    for column in frame.columns:
        name = str(column).lower()
        if name.endswith("_n") or name in {"n", "n_pos", "count"}:
            small = int(pd.to_numeric(frame[column], errors="coerce").between(1, 9).sum())
            if small:
                notes.append(f"{path.name}  {column}: {small} value(s) between 1 and 9")
    return notes


def main():
    manifest_path = EXPORT / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"No completed export at {EXPORT} (manifest.json missing).")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    restricted = restricted_files()
    files = [name for name in manifest["files"] if name not in restricted]
    missing = [name for name in files if not (EXPORT / name).is_file()]
    if missing:
        sys.exit("Files listed in manifest.json are missing:\\n  " + "\\n  ".join(missing))

    DEST.mkdir(parents=True, exist_ok=True)
    notes = []
    for name in files:
        shutil.copy2(EXPORT / name, DEST / name)
        if name.endswith(".csv"):
            notes += check_csv(DEST / name)
        print("copied", name)
    (DEST / "manifest.json").write_text(json.dumps(strip_paths(manifest), indent=2), encoding="utf-8")
    print("copied manifest.json (machine paths removed)")
    for name in sorted(restricted & set(manifest["files"])):
        print("NOT copied (participant-level):", name)
    if notes:
        print("\\nSmall counts to review before committing:")
        print("\\n".join("  " + line for line in notes))
    else:
        print("No counts between 1 and 9.")


if __name__ == "__main__":
    main()
