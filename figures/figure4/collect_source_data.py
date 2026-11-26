"""Copy the Fig. 4 export from `runs/explainability/figure4/` into `figures/figure4/source_data/`.

    python figures/figure4/collect_source_data.py [--include-sampled-points]

The export (manifest.json and the CSVs it lists) is written by the Fig. 4 data generator in
`explainability/` (v5, export schema crescent.fig4.explainability.v4) into the repository's `runs/`
folder (git-ignored; see figures/README.md). Set CRESCENT_RUNS_DIR to read it from another location.
It includes the panel E tables: `fig4E_dose_response.csv` (cohort-averaged predicted risk across CRP
and Townsend values, per condition) and `fig4E_input_distribution.csv` (binned test-cohort values,
bins with fewer than 10 participants already suppressed by the generator).

`fig4D_beeswarm_points.csv` holds one Shapley value and input percentile per sampled
participant-condition pair (no identifiers, but participant-level). It is left out unless
--include-sampled-points is given, which should follow UK Biobank's agreement. plot.py needs it
for the individual-contribution panel.

No machine information leaves the export: manifest.json keeps only file hashes and settings, with run
folders and any other absolute paths removed. The file hashes that plot.py checks are unchanged.
The script stops if a table has an `eid` column, or if any copied file still contains a machine path,
the current user or host name, or a term listed in CRESCENT_PRIVATE_TERMS (comma-separated, e.g. a
project or storage name). It lists any count between 1 and 9 for review before anything is committed.
"""
import getpass
import json
import os
import re
import shutil
import socket
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUNS = Path(os.environ.get("CRESCENT_RUNS_DIR", REPO / "runs")).expanduser().resolve()
EXPORT = RUNS / "explainability" / "figure4"
DEST = HERE / "source_data"

SAMPLED_POINTS = "fig4D_beeswarm_points.csv"
# Manifest entries that only describe where and how the export was produced on the analysis machine.
MACHINE_KEYS = {"run_dir"}


def restricted_files():
    return set() if "--include-sampled-points" in sys.argv[1:] else {SAMPLED_POINTS}


_ABSOLUTE = re.compile(r"^(/|~|[A-Za-z]:[\\/]|\\\\)")
# An absolute path anywhere inside a string, e.g. "Run folder: /gpfs/..." or "C:\\Users\\...".
_EMBEDDED = re.compile(r"(?<![\w.])(?:/(?:[\w.-]+/)+[\w.-]*|~/[\w./-]+|[A-Za-z]:\\[^\s\"']+)")


def strip_paths(value):
    if isinstance(value, dict):
        return {key: strip_paths(item) for key, item in value.items() if key not in MACHINE_KEYS}
    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    if isinstance(value, str):
        if _ABSOLUTE.match(value):
            return Path(value.replace("\\", "/")).name
        return _EMBEDDED.sub(lambda m: Path(m.group(0).replace("\\", "/")).name, value)
    return value


def private_terms():
    terms = {getpass.getuser(), socket.gethostname().split(".")[0]}
    terms |= {t.strip() for t in os.environ.get("CRESCENT_PRIVATE_TERMS", "").split(",")}
    return {t for t in terms if len(t) >= 3}


def leak_scan(paths):
    """Return problems: machine paths or private terms in any copied text file."""
    terms = private_terms()
    problems = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in sorted(set(_EMBEDDED.findall(text)))[:5]:
            problems.append(f"{path.name}: path-like text {match!r}")
        lowered = text.lower()
        for term in sorted(terms):
            if term.lower() in lowered:
                problems.append(f"{path.name}: contains {term!r}")
    return problems


def check_csv(path):
    frame = pd.read_csv(path)
    if any(str(c).strip().lower() == "eid" for c in frame.columns):
        sys.exit(f"STOP: {path.name} has an eid column; it must not be published.")
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
        sys.exit("No completed export in runs/explainability/figure4 (manifest.json missing).")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "crescent.fig4.explainability.v4":
        sys.exit("The export is not schema v4; rerun the v5 Fig. 4 data generator (panel E).")
    restricted = restricted_files()
    files = [name for name in manifest["files"] if name not in restricted]
    missing = [name for name in files if not (EXPORT / name).is_file()]
    if missing:
        sys.exit("Files listed in manifest.json are missing:\n  " + "\n  ".join(missing))

    DEST.mkdir(parents=True, exist_ok=True)
    notes = []
    for name in files:
        shutil.copy2(EXPORT / name, DEST / name)
        if name.endswith(".csv"):
            notes += check_csv(DEST / name)
        print("copied", name)
    (DEST / "manifest.json").write_text(json.dumps(strip_paths(manifest), indent=2), encoding="utf-8")
    print("copied manifest.json (run folders and machine paths removed)")
    for name in sorted(restricted & set(manifest["files"])):
        print("NOT copied (participant-level):", name)

    problems = leak_scan([DEST / name for name in files] + [DEST / "manifest.json"])
    if problems:
        sys.exit("STOP: machine or location details found in source_data/; fix before committing:\n  "
                 + "\n  ".join(problems))
    print("No machine paths, user or host names found.")
    if notes:
        print("\nSmall counts to review before committing:")
        print("\n".join("  " + line for line in notes))
    else:
        print("No counts between 1 and 9.")


if __name__ == "__main__":
    main()
