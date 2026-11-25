"""Fig. 3: CRESCENT discrimination and calibration on the held-out test cohort.

Redraws manuscript Fig. 3 from the aggregate tables in `source_data/` and writes PNG,
PDF and SVG to `output/`. No UK Biobank access is needed.

    python figures/figure3/plot.py

Set FIG2_DPI (default 600) or FIG2_SAVE_TIFF=1 to change the raster output.

Code from the notebook `Figure2_CRESCENT_publication` (the notebook calls this
"Figure 2"; it is Fig. 3 in the manuscript). Cells 5 and 7 are copied unchanged apart
from three marked path lines. `source_data/` is filled on the HPC by
`build_source_data.py`.
"""
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without a display; files are written to output/

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import *  # noqa: E402,F401,F403  (notebook cell 1 settings)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# =============================================================================
# Paths (replace the HPC paths of notebook cell 1): source_data/ stands in for the run folder
# =============================================================================

CRESCENT_RUN_DIR = Path(
    os.environ.get("FIG3_SOURCE_DATA", HERE / "source_data")
).expanduser().resolve()
CRESCENT_METRICS_PATH = (
    CRESCENT_RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
)
CRESCENT_FIG2_DIR = CRESCENT_RUN_DIR / "figure2_revised_inputs_2yr"
CRESCENT_FIG2_OUTPUT_DIR = HERE / "output"
CRESCENT_FIG2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG2_COMPARATOR_METRICS_PATH = (
    CRESCENT_RUN_DIR
    / f"comparator_{FIG2_COMPARATOR_MODEL}_test_per_condition_metrics_by_horizon.csv"
)


# =============================================================================
# ICD-chapter helpers from notebook cell 3 (unchanged). The plotting cell re-validates
# the colour key with them; in the notebook they came from running cell 3 first.
# condition_base is rebuilt from the published colour key, which keeps the model order.
# =============================================================================

condition_base = (
    pd.read_csv(CRESCENT_FIG2_DIR / "figure2_revised_condition_color_key.csv")
    [["condition_id", "condition_name"]]
    .sort_values("condition_id")
    .reset_index(drop=True)
)
condition_base["condition_name"] = condition_base["condition_name"].astype(str).str.strip()
K = int(len(condition_base))

import re

_roman_chapters = (
    "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII",
    "XIX", "XX", "XXI", "XXII",
)


def _chapter_number(value):
    """Recognise ICD chapter numerals, not arbitrary free-text labels."""
    match = re.fullmatch(
        r"(?:Chapter\s+)?([IVXLCDM]+)(?:\s*[:.]\s*.*)?",
        str(value).strip(), flags=re.I,
    )
    if not match:
        return None
    roman = match.group(1).upper()
    return (_roman_chapters.index(roman) + 1) if roman in _roman_chapters else None


def _validated_color_key(candidate, description):
    missing_columns = sorted(
        {"condition_name", "ICD_Chapter", "Chapter_Name"} - set(candidate)
    )
    if missing_columns:
        raise ValueError(f"{description}: missing columns {missing_columns}")
    candidate = candidate[
        ["condition_name", "ICD_Chapter", "Chapter_Name"]
    ].copy()
    candidate["condition_name"] = candidate["condition_name"].astype(str).str.strip()
    if candidate["condition_name"].duplicated().any():
        duplicates = candidate.loc[
            candidate["condition_name"].duplicated(), "condition_name"
        ].tolist()
        raise ValueError(f"{description}: duplicate condition names {duplicates[:8]}")
    merged = condition_base.merge(
        candidate, how="left", on="condition_name", validate="one_to_one"
    )
    chapter_numbers = merged["ICD_Chapter"].map(_chapter_number)
    chapter_names = merged["Chapter_Name"].fillna("").astype(str).str.strip()
    valid_name = ~chapter_names.str.lower().isin(
        ["", "unmapped", "nan", "none", "unknown"]
    )
    if chapter_numbers.isna().any() or (~valid_name).any():
        bad = merged.loc[
            chapter_numbers.isna() | (~valid_name), "condition_name"
        ].tolist()
        raise ValueError(
            f"{description}: {len(bad)} of {K} conditions have missing/invalid "
            f"ICD chapters: {bad[:12]}"
        )
    merged["Chapter_Name"] = chapter_names
    merged["ICD_Chapter"] = chapter_numbers.map(
        lambda number: f"Chapter {_roman_chapters[int(number) - 1]}"
    )
    collisions = merged.assign(_n=chapter_numbers).groupby("_n")[
        "Chapter_Name"
    ].nunique()
    if (collisions > 1).any():
        raise ValueError(
            f"{description}: one ICD chapter has multiple Chapter_Name labels; "
            "use one exact name per chapter."
        )
    repeated_names = merged.groupby("Chapter_Name")["ICD_Chapter"].nunique()
    if (repeated_names > 1).any():
        raise ValueError(
            f"{description}: a Chapter_Name is assigned to multiple chapters."
        )
    return merged


# =============================================================================
# Notebook cell 5
# =============================================================================

def _normalise_panel_h_metrics(frame, model_name):
    frame = frame.copy()
    columns = set(frame.columns)

    legacy_required = {
        "section",
        "condition",
        "horizon_months",
        "auroc",
        "n_pos",
    }
    if legacy_required.issubset(columns) and "condition_name" not in columns:
        if "risk_type" in frame.columns:
            frame = frame[
                frame["risk_type"].astype(str).str.strip().eq(PRIMARY_RISK_TYPE)
            ].copy()
        frame = frame[
            frame["section"].astype(str).str.strip().str.lower().eq("within")
        ].copy()
        out = frame[
            ["condition", "horizon_months", "auroc", "n_pos"]
        ].copy()
    else:
        required = {
            "condition_name",
            "horizon_months",
            "risk_type",
            "event_n",
        }
        missing = sorted(required - columns)
        if missing:
            raise ValueError(
                f"{model_name}: Panel H metrics are missing {missing}."
            )
        frame = frame[
            frame["risk_type"].astype(str).str.strip().eq(PRIMARY_RISK_TYPE)
        ].copy()
        auc_column = next(
            (
                column
                for column in ("auc_ipcw_cr", "auroc")
                if column in frame.columns
            ),
            None,
        )
        if auc_column is None:
            raise ValueError(
                f"{model_name}: no competing-risk AUROC column was found."
            )
        out = frame[
            ["condition_name", "horizon_months", auc_column, "event_n"]
        ].rename(
            columns={
                "condition_name": "condition",
                auc_column: "auroc",
                "event_n": "n_pos",
            }
        )

    out["condition"] = out["condition"].astype(str).str.strip()
    for column in ("horizon_months", "auroc", "n_pos"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out[
        out["horizon_months"].eq(int(FIG2_PRIMARY_HORIZON_MONTHS))
        & out["condition"].ne("")
        & out["auroc"].notna()
    ].copy()
    if out.empty:
        raise RuntimeError(
            f"{model_name}: no finite death-adjusted 2-year AUROC rows."
        )

    disagreement = (
        out.groupby("condition")["auroc"].agg(lambda values: values.max() - values.min())
    )
    if (disagreement > 1e-12).any():
        conditions = disagreement[disagreement > 1e-12].index.tolist()
        raise RuntimeError(
            f"{model_name}: duplicate 2-year AUROC rows disagree for "
            f"{conditions[:10]}."
        )

    out = (
        out.groupby("condition", as_index=False)
        .agg(auroc=("auroc", "first"), n_pos=("n_pos", "max"))
    )
    out["Model"] = model_name
    return out


if not FIG2_COMPARATOR_METRICS_PATH.is_file():
    raise FileNotFoundError(
        "Panel H requires the configured Delphi-2M metrics table:\n  "
        + str(FIG2_COMPARATOR_METRICS_PATH)
        + "\nSet FIG2_COMPARATOR_METRICS_PATH before running if it is elsewhere."
    )

panel_h_target = _normalise_panel_h_metrics(
    pd.read_csv(CRESCENT_METRICS_PATH),
    FIG2_TARGET_MODEL,
)
panel_h_comparator = _normalise_panel_h_metrics(
    pd.read_csv(FIG2_COMPARATOR_METRICS_PATH),
    FIG2_COMPARATOR_MODEL,
)
panel_h_metrics = pd.concat(
    [panel_h_target, panel_h_comparator],
    ignore_index=True,
)

panel_h_auroc = panel_h_metrics.pivot(
    index="condition",
    columns="Model",
    values="auroc",
)
panel_h_counts = panel_h_metrics.pivot(
    index="condition",
    columns="Model",
    values="n_pos",
)
paired_conditions = panel_h_auroc.dropna(
    subset=[FIG2_TARGET_MODEL, FIG2_COMPARATOR_MODEL]
).index
if len(paired_conditions) == 0:
    raise RuntimeError(
        "Panel H has no paired 2-year conditions for CRESCENT and Delphi-2M."
    )

paired_counts = panel_h_counts.reindex(paired_conditions)
count_difference = (
    paired_counts[FIG2_TARGET_MODEL] - paired_counts[FIG2_COMPARATOR_MODEL]
).abs()
count_mismatch = count_difference[
    count_difference.notna() & count_difference.gt(0)
]
if len(count_mismatch):
    raise RuntimeError(
        "Panel H model tables have different locked-test event counts for "
        f"{len(count_mismatch)} paired conditions. Examples: "
        + ", ".join(count_mismatch.index[:10].astype(str))
    )

panel_h_comparison = (
    panel_h_auroc.loc[paired_conditions, [FIG2_COMPARATOR_MODEL, FIG2_TARGET_MODEL]]
    .rename(
        columns={
            FIG2_COMPARATOR_MODEL: "comparator_auroc",
            FIG2_TARGET_MODEL: "target_auroc",
        }
    )
    .reset_index()
)
panel_h_comparison["event_n"] = paired_counts[
    FIG2_TARGET_MODEL
].to_numpy()
panel_h_audit_path = CRESCENT_FIG2_OUTPUT_DIR / "figure2_panelH_model_comparison_audit.csv"  # repository: write to output/
panel_h_comparison.to_csv(panel_h_audit_path, index=False)

print("Panel H comparison loaded")
print("  comparator:       ", FIG2_COMPARATOR_MODEL)
print("  paired conditions:", len(panel_h_comparison))
print("  audit:            ", panel_h_audit_path)


# =============================================================================
# Notebook cell 7
# =============================================================================

# =============================================================================
# FIGURE 2 — PLOTTING CELL (drawn at final printed size, 183 mm wide)
#
# Layout of this cell:
#   1. Style block      every size, weight, colour and author switch, once
#   2. Validation block source/model checks and table loading (unchanged logic)
#   3. Helpers          axes, ticks, keys, layout audit
#   4. Panel functions  one per panel, A to I, plus the chapter key
#   5. Layout block     builds the figure
#   6. Export block     PDF, SVG, PNG (and optional TIFF) at exact size
#
# Presentation only: no table, filter or statistic differs from the input-build
# cell. The calibration column `ici` is displayed under the name set in
# CAL_MEAN_LABEL; the column name itself is a lookup key and is not renamed.
# =============================================================================

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import to_rgba
from matplotlib.ticker import (
    FixedLocator,
    FuncFormatter,
    LogLocator,
    MultipleLocator,
    NullFormatter,
    NullLocator,
    SymmetricalLogLocator,
)


# =============================================================================
# 1. STYLE BLOCK
# =============================================================================

# ---- Author switches. Each one changes how a reader interprets the figure. --
PANEL_LETTER_CASE = "lower"           # "lower" (a, b, c) or "upper" (A, B, C)
A_HOLLOW_BELOW_EVENTS = None          # None = every observed marker filled.
                                      # Integer n = hollow marker when an age
                                      # bin has fewer than n events.
A_SIZE_BY_PARTICIPANTS = False        # True = observed-marker area follows the
                                      # number of eligible participants.
DE_Y_SCALE = "log"                    # panels D, E: "log" or "linear"
G_AXIS_SCALE = "linear"                  # panel G: "log" or "linear"
ROC_AXIS_WORDING = "clinical"         # "clinical" = sensitivity / 1 − specificity
                                      # "rates"    = true / false positive rate
CAL_ERROR_UNIT = "percentage points"  # or "%"
CAL_MEAN_LABEL = "GCE"                # display name of the `ici` column
CAL_E90_LABEL = "E90"                 # display name of the `e90` column

# ---- Run configuration ------------------------------------------------------
TARGET_MODEL = FIG2_TARGET_MODEL
COMPARATOR_MODEL = FIG2_COMPARATOR_MODEL
PRIMARY_HORIZON = int(FIG2_PRIMARY_HORIZON_MONTHS)
HORIZONS = [int(x) for x in FIG2_HORIZONS_MONTHS]

FIG2_RUN_DIR = Path(CRESCENT_RUN_DIR).expanduser().resolve()
FIG2_DIR = Path(CRESCENT_FIG2_DIR).expanduser().resolve()
FIG2_OUTPUT_DIR = Path(CRESCENT_FIG2_OUTPUT_DIR).expanduser().resolve()
CALIBRATION_RUN_DIR = FIG2_RUN_DIR
FIG2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 2026                       # only for sampling the optional dense layer
DPI = int(os.getenv("FIG2_DPI", "600"))
if DPI < 72:
    raise ValueError("FIG2_DPI must be at least 72.")
SAVE_TIFF = os.getenv("FIG2_SAVE_TIFF", "0").strip() == "1"
PREVIEW_DPI = 200                     # on-screen sharpness only, not figure size

# ---- Page geometry (millimetres) --------------------------------------------
FIG_W_MM = 183.0
FIG_MAX_H_MM = 225.0
MARGIN_LEFT_MM = 12.5                 # common left edge of the first axes
MARGIN_RIGHT_MM = 2.0                 # common right edge of the last axes
MARGIN_TOP_MM = 1.5
MARGIN_BOTTOM_MM = 1.5
LETTER_X_MM = 1.0                     # common left margin of panel letters
LETTER_RISE_MM = 1.6                  # letter baseline above a panel's top edge
HEADER_MM = 5.5                       # strip above each row (letter, keys)
XLABEL_ZONE_MM = 7.5                  # tick labels plus axis title below axes

A_LEGEND_MM = 4.0
A_TITLE_MM = 6.3                      # two lines of title text
A_AXES_H_MM = 22.5
A_COL_GAP_MM = 8.0
A_ROW_GAP_MM = 2.0
B_AXES_H_MM = 31.0
B_TICK_ZONE_MM = 13.5                 # rotated chapter names
CDE_AXES_H_MM = 31.0
CDE_COL_GAP_MM = 14.0
F_LEGEND_COL_MM = 17.5                # key for F and G, between the two panels
G_GAP_MM = 15.0                       # room for G's two-line y-axis title
H_GAP_MM = 10.5
I_GAP_MM = 14.0                       # room for I's two-line y-axis title
KEY_GAP_MM = 2.5
KEY_ROW_MM = 3.1
KEY_ROWS = 3
KEY_COLS = 5
CHAPTER_KEY_FORMAT = "Ch {numeral}: {name}"   # e.g. "Ch IX: Circulatory diseases"

# ---- Typography (points) ----------------------------------------------------
FONT_PREFERENCE = ("Arial", "Helvetica", "Liberation Sans")
FONT_LAST_RESORT = "DejaVu Sans"
FS_LETTER = 8.0
FS_LABEL = 7.0                        # axis titles, small-multiple titles
FS_SMALL = 6.0                        # tick labels, legends, annotations
FS_MIN = 5.0
MINUS = "−"
EN_DASH = "–"

# ---- Colour -----------------------------------------------------------------
INK = "#1A1A1A"                       # text
INK_SECONDARY = "#4D4D4D"             # axes, secondary text, neutral key marks
REFERENCE_GREY = "#8C8C8C"            # diagonals, chance level
GRID_GREY = "#E6E6E6"                 # gridlines
BACKGROUND = "white"
MARK_EDGE = "white"
DEATH_COLOR = "#222222"
PLACEHOLDER_CHAPTER_COLOR = "#B0B0B0" # chapters XV to XXII (absent from data)
SEX_ORDER = ("Male", "Female")
SEX_COLOR = {"Male": "#31446B", "Female": "#D4683C"}
SEX_MARKER = {"Male": "o", "Female": "^"}
SEX_OFFSET = {"Male": -0.17, "Female": 0.17}
BOX_FILL_ALPHA = 0.20
BAND_ALPHA = 0.35                     # faint alternate-chapter band in panel B

# ---- Lines and marks (points) -----------------------------------------------
LW_AXIS = 0.5
LW_DATA = 1.0
LW_REFERENCE = 0.5
LW_GRID = 0.3
LW_BOX = 0.6
LW_MEDIAN = 1.0
LW_ROC = 0.9
LW_CALIBRATION = 0.7
LW_STEM = 0.4
LW_MARK_EDGE = 0.3
LW_HOLLOW = 0.6
TICK_LEN = 2.5
TICK_LEN_MINOR = 1.4
MS_POINT = 3.5                        # scatter marker, diameter
MS_OBSERVED = 3.4                     # panel A observed markers
MS_DOT = 2.0                          # per-condition dots in C, D, E
MS_CURVE = 2.6                        # markers on the F and G lines
MS_KEY = 3.2                          # chapter-key and neutral key markers
DOT_ALPHA = 0.35
STEM_ALPHA = 0.40
DENSE_ALPHA = 0.12
DENSE_MS = 1.4
BUBBLE_MIN_PT = 2.5                   # panel B bubble diameter range
BUBBLE_MAX_PT = 9.0
BOX_WIDTH = 0.27
DOT_SPREAD = 0.20                     # sideways spread of dots, x data units
KEY_LINE_MM = 5.0                     # length of a line sample in a key
KEY_PAD_MM = 1.0                      # sample-to-label distance
KEY_ITEM_GAP_MM = 3.5                 # distance between key entries
B_LABEL_ROTATION = 35
SYMLOG_LINTHRESH = 0.01               # D, E: linear below this (percentage points)
SYMLOG_LINSCALE = 0.6
G_ZERO_FRACTION_LIMIT = 1.0 / 7.0     # above this, panel G falls back to linear
GOLDEN = 0.6180339887498949

# Within one ICD chapter the selected conditions share a colour, so they are
# separated by line style and marker (never by tint).
WITHIN_CHAPTER_STYLE = (("-", "o"), ("--", "s"), (":", "D"), ("-.", "v"))

if ROC_AXIS_WORDING == "clinical":
    ROC_X_LABEL, ROC_Y_LABEL = f"1 {MINUS} specificity", "Sensitivity"
elif ROC_AXIS_WORDING == "rates":
    ROC_X_LABEL, ROC_Y_LABEL = "False positive rate", "True positive rate"
else:
    raise ValueError("ROC_AXIS_WORDING must be 'clinical' or 'rates'.")
for _name, _value in (("DE_Y_SCALE", DE_Y_SCALE), ("G_AXIS_SCALE", G_AXIS_SCALE)):
    if _value not in ("log", "linear"):
        raise ValueError(f"{_name} must be 'log' or 'linear'.")
if PANEL_LETTER_CASE not in ("lower", "upper"):
    raise ValueError("PANEL_LETTER_CASE must be 'lower' or 'upper'.")

_installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
FONT_USED = next(
    (name for name in FONT_PREFERENCE if name in _installed_fonts), None
)
if FONT_USED is None:
    FONT_USED = FONT_LAST_RESORT
    warnings.warn(
        "None of " + ", ".join(FONT_PREFERENCE) + " is installed. The figure "
        f"is drawn in {FONT_USED}; install Arial before final submission."
    )
print("Figure 2 font:", FONT_USED)

plt.rcParams.update(
    {
        "font.family": [FONT_USED],
        "font.size": FS_SMALL,
        "axes.titlesize": FS_LABEL,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_SMALL,
        "ytick.labelsize": FS_SMALL,
        "legend.fontsize": FS_SMALL,
        "legend.title_fontsize": FS_SMALL,
        "legend.frameon": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_SECONDARY,
        "axes.linewidth": LW_AXIS,
        "axes.labelpad": 2.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": True,
        "axes.facecolor": "none",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": TICK_LEN,
        "ytick.major.size": TICK_LEN,
        "xtick.minor.size": TICK_LEN_MINOR,
        "ytick.minor.size": TICK_LEN_MINOR,
        "xtick.major.width": LW_AXIS,
        "ytick.major.width": LW_AXIS,
        "xtick.minor.width": LW_AXIS,
        "ytick.minor.width": LW_AXIS,
        "xtick.major.pad": 1.5,
        "ytick.major.pad": 1.5,
        "lines.linewidth": LW_DATA,
        "figure.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


# =============================================================================
# 2. VALIDATION BLOCK
# =============================================================================

# =============================================================================
# One locked CRESCENT source for every model-dependent Figure-2 panel
# =============================================================================

TARGET_METRICS_PATH = Path(CRESCENT_METRICS_PATH).expanduser().resolve()
COMPARATOR_METRICS_PATH = Path(
    FIG2_COMPARATOR_METRICS_PATH
).expanduser().resolve()
if TARGET_METRICS_PATH != Path(CRESCENT_METRICS_PATH).expanduser().resolve():
    raise RuntimeError(
        "The Figure-2 target metrics do not resolve to CRESCENT_METRICS_PATH: "
        f"{TARGET_METRICS_PATH}"
    )

if TARGET_METRICS_PATH.parent != FIG2_RUN_DIR:
    raise RuntimeError(
        "CRESCENT metrics and Figure-2 inputs do not share one run directory."
    )

if not COMPARATOR_METRICS_PATH.is_file():
    raise FileNotFoundError(
        f"{COMPARATOR_MODEL} metrics are missing: {COMPARATOR_METRICS_PATH}"
    )

if not FIG2_DIR.is_dir():
    raise FileNotFoundError(
        "The rebuilt CRESCENT Figure-2 input directory does not exist. "
        "Run Cell 4A first:\n  " + str(FIG2_DIR)
    )

source_manifest_path = FIG2_DIR / "figure2_crescent_source_manifest.json"
if not source_manifest_path.exists():
    raise FileNotFoundError(
        "CRESCENT Figure-2 source manifest is missing. Run Cell 4A first:\n  "
        + str(source_manifest_path)
    )

with open(source_manifest_path) as f:
    source_manifest = json.load(f)

if str(source_manifest.get("model", "")).strip() != TARGET_MODEL:
    raise RuntimeError(
        "Figure-2 source manifest model mismatch: "
        f"{source_manifest.get('model')!r} versus {TARGET_MODEL!r}."
    )

if str(source_manifest.get("comparator_model", "")).strip() != COMPARATOR_MODEL:
    raise RuntimeError(
        "Figure-2 source manifest comparator mismatch: "
        f"{source_manifest.get('comparator_model')!r} versus "
        f"{COMPARATOR_MODEL!r}."
    )

manifest_comparator = (  # repository: manifest paths may be relative to the run folder
    FIG2_RUN_DIR / source_manifest.get("comparator_metrics_path", "")
).expanduser().resolve()
if manifest_comparator != COMPARATOR_METRICS_PATH:
    raise RuntimeError(
        "Figure-2 source manifest points to different comparator metrics: "
        f"{manifest_comparator}"
    )

manifest_run = (FIG2_RUN_DIR / source_manifest.get("run_dir", "")).expanduser().resolve()  # repository: relative paths allowed
if manifest_run != FIG2_RUN_DIR:
    raise RuntimeError(
        "Figure-2 source manifest points to a different run: "
        f"{manifest_run}"
    )

print("FIGURE-2 CRESCENT SOURCE AUDIT")
print("  Target model:             ", TARGET_MODEL)
print("  Target metrics:           ", TARGET_METRICS_PATH)
print("  Comparator metrics:       ", COMPARATOR_METRICS_PATH)
print("  Locked run:               ", FIG2_RUN_DIR)
print("  Rebuilt Figure-2 inputs:  ", FIG2_DIR)
print("  Figure-2 output folder:   ", FIG2_OUTPUT_DIR)
print("  Source manifest:          ", source_manifest_path)


# =============================================================================
# Load all generated tables
# =============================================================================

FILES = {
    "A": "figure2_revised_panelA_age_sex_rates_2yr.csv",
    "B": "figure2_revised_panelB_frequency_sex_2yr.csv",
    "C": "figure2_revised_panelC_chapter_sex_auroc_2yr.csv",
    "Cpair": "figure2_revised_panelC_chapter_sex_pair_deltas_2yr.csv",
    "E": "figure2_revised_panelE_horizon_sex_auroc.csv",
    "F": "figure2_revised_panelF_roc_curves_2yr.csv",
    "H": "figure2_revised_panelH_calibration_deciles_2yr.csv",
    "Hsummary": "figure2_revised_panelH_calibration_summary_2yr.csv",
    "colors": "figure2_revised_condition_color_key.csv",
    "selected": "figure2_revised_selected_endpoints_panelA.csv",
}

missing = [
    str(FIG2_DIR / filename)
    for filename in FILES.values()
    if not (FIG2_DIR / filename).exists()
]

if missing:
    raise FileNotFoundError(
        "Missing Figure-2 files:\n" + "\n".join(missing)
    )

T = {
    key: pd.read_csv(FIG2_DIR / filename)
    for key, filename in FILES.items()
}

A = T["A"]
B = T["B"]
C = T["C"]
Cpair = T["Cpair"]
E = T["E"]
F = T["F"]
H = T["H"]
Hsummary = T["Hsummary"]
color_key = T["colors"]
selected = T["selected"]


# =============================================================================
# Load all-horizon calibration results from the locked CRESCENT run
# =============================================================================

all_cal_path = (
    FIG2_DIR / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
).resolve()

if not all_cal_path.is_file():
    raise FileNotFoundError(
        f"The canonical all-horizon calibration file is missing for "
        f"{TARGET_MODEL!r}:\n  {all_cal_path}\nRun the input-build cell first."
    )

try:
    all_cal_path.relative_to(CALIBRATION_RUN_DIR)
except ValueError as exc:
    raise RuntimeError(
        "Calibration data resolved outside the locked CRESCENT run: "
        f"{all_cal_path}"
    ) from exc

ALL_CAL = pd.read_csv(all_cal_path)

# All calibration files must be generated by the new CRESCENT export cell.
calibration_metadata_path = FIG2_RUN_DIR / "figure2_calibration_export_metadata.json"
if calibration_metadata_path.exists():
    with open(calibration_metadata_path) as f:
        calibration_metadata = json.load(f)
    if str(calibration_metadata.get("model", "")).strip() != TARGET_MODEL:
        raise RuntimeError(
            "Calibration export metadata model mismatch: "
            f"{calibration_metadata.get('model')!r}."
        )
    if str(calibration_metadata.get("risk_type", "")).strip() != "death_adjusted":
        raise RuntimeError("Figure-2 calibration export is not death_adjusted.")
else:
    warnings.warn(
        "figure2_calibration_export_metadata.json is absent; file-path and "
        "schema checks will still be enforced."
    )

if "horizon_months" in ALL_CAL.columns:
    available_calibration_horizons = sorted(
        pd.to_numeric(ALL_CAL["horizon_months"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
else:
    available_calibration_horizons = []

missing_calibration_horizons = [
    horizon for horizon in HORIZONS
    if horizon not in available_calibration_horizons
]

print("  Calibration source:       ", all_cal_path)
print("  Calibration horizons:     ", available_calibration_horizons)

if missing_calibration_horizons:
    raise RuntimeError(
        f"{TARGET_MODEL!r} calibration file is missing required horizons: "
        + ", ".join(str(h) for h in missing_calibration_horizons)
    )


# =============================================================================
# Numeric cleanup
# =============================================================================

NUMERIC = {
    "A": [
        "condition_id",
        "horizon_months",
        "age_bin_mid",
        "eligible_n",
        "known_n",
        "event_n",
        "mean_predicted_risk",
        "observed_aj_cif",
    ],
    "B": [
        "condition_id",
        "horizon_months",
        "event_n",
        "known_n",
        "control_n",
        "auroc",
        "auprc",
        "frequency_for_plot",
        "event_n_2yr_all",
    ],
    "C": [
        "condition_id",
        "horizon_months",
        "event_n",
        "known_n",
        "control_n",
        "auroc",
        "auprc",
        "frequency_for_plot",
    ],
    "Cpair": [
        "condition_id",
        "Male",
        "Female",
        "auroc_female_minus_male",
        "frequency_for_plot",
    ],
    "E": [
        "condition_id",
        "horizon_months",
        "event_n",
        "known_n",
        "control_n",
        "auroc",
        "auprc",
        "frequency_for_plot",
    ],
    "F": [
        "condition_id",
        "horizon_months",
        "fpr",
        "tpr",
        "threshold",
        "auroc",
        "auprc",
    ],
    "H": [
        "condition_id",
        "horizon_months",
        "calibration_bin",
        "n",
        "known_n",
        "event_n",
        "predicted_mean",
        "observed_aj_cif",
    ],
    "Hsummary": [
        "condition_id",
        "horizon_months",
        "ici",
        "e50",
        "e90",
        "total_event_n",
    ],
    "ALL_CAL": [
        "condition_id",
        "horizon_months",
        "ici",
        "e50",
        "e90",
        "total_event_n",
    ],
}

for name, frame in [
    ("A", A),
    ("B", B),
    ("C", C),
    ("Cpair", Cpair),
    ("E", E),
    ("F", F),
    ("H", H),
    ("Hsummary", Hsummary),
    ("ALL_CAL", ALL_CAL),
]:
    for column in NUMERIC[name]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    if "condition_name" in frame.columns:
        frame["condition_name"] = (
            frame["condition_name"].astype(str).str.strip()
        )

if "sex" not in ALL_CAL.columns:
    if "calibration_stratum" in ALL_CAL.columns:
        ALL_CAL["sex"] = ALL_CAL["calibration_stratum"]
    else:
        ALL_CAL["sex"] = "All"

ALL_CAL["sex"] = ALL_CAL["sex"].astype(str)


# =============================================================================
# ICD chapter and condition colours
# =============================================================================

# Use the EXACT validated mapping created by Cell 4A. Never replace a
# missing chapter with gray and never let an unknown condition plot silently.
try:
    color_key = _validated_color_key(
        color_key, str(FIG2_DIR / FILES["colors"])
    )
except ValueError as exc:
    raise ValueError(
        "Figure 2's colour key is stale. Run the input-build section of "
        "THIS revised notebook, which embeds Con_chapters__V3.csv and "
        "recreates figure2_revised_condition_color_key.csv."
    ) from exc

_written_key = T["colors"].set_index("condition_name")
if "hex_color" not in _written_key.columns:
    raise ValueError(
        "The Figure-2 colour key lacks hex_color. Run the input-build "
        "section of this revised notebook again."
    )
_expected_hex = color_key["ICD_Chapter"].map(
    lambda chapter: ICD_CHAPTER_COLORS[_chapter_number(chapter)]
)
_actual_hex = color_key["condition_name"].map(
    _written_key["hex_color"].astype(str).str.upper()
)
if not np.array_equal(_actual_hex.to_numpy(), _expected_hex.to_numpy()):
    raise ValueError(
        "The cached chapter colours differ from this notebook's palette. "
        "Re-run the input-build section before plotting."
    )

# Stable, print-friendly dark/mid-tone palette, keyed by ICD chapter number.
# A chapter keeps its hue when another chapter is absent from the data.
if len(ICD_CHAPTER_COLORS) != 22:
    raise AssertionError("Expected all 22 ICD chapters in the palette.")


def roman_value(text):
    match = re.search(
        r"Chapter\s+([IVXLCDM]+)",
        str(text),
        flags=re.I,
    )

    if not match:
        return 999

    values = {
        "I": 1,
        "V": 5,
        "X": 10,
        "L": 50,
        "C": 100,
        "D": 500,
        "M": 1000,
    }

    total = 0
    previous = 0

    for character in reversed(match.group(1).upper()):
        value = values[character]
        total += -value if value < previous else value
        previous = max(previous, value)

    return total


def short_chapter(text):
    match = re.search(
        r"Chapter\s+([IVXLCDM]+)",
        str(text),
        flags=re.I,
    )

    if match:
        return f"Ch. {match.group(1).upper()}"

    return str(text).split(":")[0]


chapter_meta = (
    color_key[
        ["Chapter_Name", "ICD_Chapter"]
    ]
    .drop_duplicates()
    .copy()
)

chapter_meta["sort"] = (
    chapter_meta["ICD_Chapter"].map(roman_value)
)

chapter_meta = chapter_meta.sort_values(
    ["sort", "Chapter_Name"]
)

chapter_order = [
    chapter
    for chapter in chapter_meta["Chapter_Name"]
    if chapter != "Death"
]

chapter_color = {
    row.Chapter_Name: ICD_CHAPTER_COLORS[roman_value(row.ICD_Chapter)]
    for row in chapter_meta.itertuples(index=False)
    if row.Chapter_Name != "Death"
}
if len(chapter_color) != len(chapter_order):
    raise RuntimeError("ICD chapter palette did not cover all figure chapters.")
chapter_color["Death"] = DEATH_COLOR

condition_chapter = (
    color_key
    .set_index("condition_name")["Chapter_Name"]
    .to_dict()
)

condition_icd = (
    color_key
    .set_index("condition_name")["ICD_Chapter"]
    .to_dict()
)

condition_color = {
    condition: chapter_color[chapter]
    for condition, chapter in condition_chapter.items()
}
condition_color["Death"] = DEATH_COLOR

_plotted_conditions = set()
for _frame in (A, B, C, Cpair, E, F, H, Hsummary, ALL_CAL):
    if "condition_name" in _frame.columns:
        _plotted_conditions.update(
            _frame["condition_name"].dropna().astype(str).str.strip()
        )
_plotted_conditions.update(
    panel_h_comparison["condition"].dropna().astype(str).str.strip()
)
_missing_colours = sorted(_plotted_conditions - set(condition_color))
if _missing_colours:
    raise ValueError(
        "Figure 2 contains conditions absent from the chapter mapping: "
        f"{_missing_colours[:15]}"
    )
print("Verified Figure-2 ICD palette:", len(chapter_order), "chapters,",
      len(condition_chapter), "conditions")

for frame in [
    A,
    B,
    C,
    Cpair,
    E,
    F,
    H,
    Hsummary,
    ALL_CAL,
]:
    if "condition_name" not in frame.columns:
        continue

    mapped_chapter = frame["condition_name"].map(
        condition_chapter
    )

    mapped_icd = frame["condition_name"].map(
        condition_icd
    )

    frame["Chapter_Name"] = mapped_chapter
    frame["ICD_Chapter"] = mapped_icd


# =============================================================================
# Selected endpoints
# =============================================================================

selected = selected.sort_values(
    (
        "selection_rank"
        if "selection_rank" in selected.columns
        else "condition_id"
    )
)

endpoint_order = (
    selected["condition_name"]
    .astype(str)
    .tolist()[:10]
)
_unknown_selected = sorted(set(endpoint_order) - set(condition_color))
if _unknown_selected:
    raise ValueError("Selected endpoints missing chapter mapping: "
                     f"{_unknown_selected}")

# Match the abbreviations defined in the manuscript Figure 2 caption.
ENDPOINT_ABBREVIATION = {
    "Cerebrovascular disease": "CeVD",
    "Chronic kidney disease": "CKD",
    "Chronic lower respiratory diseases": "CRD",
    "Dementia": "DEM",
    "Dermatitis and eczema": "ECZ",
    "Diabetes": "DBT",
    "Hypertensive diseases": "HTN",
    "Ischaemic heart diseases": "IHD",
    "Inflammatory polyarthropathies": "IAR",
    "Mood disorders": "MOD",
}
_missing_abbreviations = sorted(set(endpoint_order) - set(ENDPOINT_ABBREVIATION))
if _missing_abbreviations:
    raise ValueError("Add manuscript abbreviations for selected endpoints: "
                     f"{_missing_abbreviations}")


# =============================================================================
# Optional dense Panel-A predictions
# =============================================================================

dense = pd.DataFrame()

dense_candidates = [
    (
        FIG2_DIR
        / "figure2_panelA_token_level_predictions_2yr.parquet"
    ),
    (
        FIG2_DIR
        / "figure2_panelA_token_level_predictions_2yr.csv.gz"
    ),
]

for dense_path in dense_candidates:
    if not dense_path.exists():
        continue

    try:
        if dense_path.suffix == ".parquet":
            dense = pd.read_parquet(dense_path)
        else:
            dense = pd.read_csv(
                dense_path,
                compression="gzip",
            )

        break

    except Exception as error:
        warnings.warn(
            f"Could not read {dense_path}: {error}"
        )

if not dense.empty:
    dense["index_age_years"] = pd.to_numeric(
        dense["index_age_years"],
        errors="coerce",
    )

    dense["predicted_risk"] = pd.to_numeric(
        dense["predicted_risk"],
        errors="coerce",
    )

    dense = dense[
        dense["condition_name"]
        .astype(str)
        .isin(endpoint_order)
        & dense["sex"]
        .astype(str)
        .isin(["Male", "Female"])
        & dense["index_age_years"].notna()
        & dense["predicted_risk"].notna()
    ].copy()

    sampled_dense = []

    for _, group in dense.groupby(
        ["condition_name", "sex"],
        sort=False,
    ):
        sampled_dense.append(
            group.sample(
                n=min(len(group), 500),
                random_state=RNG_SEED,
            )
        )

    dense = (
        pd.concat(sampled_dense, ignore_index=True)
        if sampled_dense
        else pd.DataFrame()
    )

# Chapters XV to XXII carry one neutral placeholder colour because they do not
# occur in this dataset. Stop if a plotted condition ever maps to one of them.
_placeholder_conditions = sorted(
    condition
    for condition in _plotted_conditions
    if condition in condition_icd and roman_value(condition_icd[condition]) > 14
)
if _placeholder_conditions:
    raise ValueError(
        "These plotted conditions map to ICD chapters XV to XXII, which only "
        "have a placeholder colour. Assign a checked colour in "
        f"ICD_CHAPTER_COLORS first: {_placeholder_conditions[:15]}"
    )

# One colour, line style and marker per selected condition, shared by F and G.
_used_per_chapter = {}
endpoint_style = {}
for _endpoint in endpoint_order:
    _chapter = condition_chapter[_endpoint]
    _position = _used_per_chapter.get(_chapter, 0)
    _linestyle, _marker = WITHIN_CHAPTER_STYLE[
        _position % len(WITHIN_CHAPTER_STYLE)
    ]
    endpoint_style[_endpoint] = {
        "color": condition_color[_endpoint],
        "linestyle": _linestyle,
        "marker": _marker,
    }
    _used_per_chapter[_chapter] = _position + 1


# =============================================================================
# 3. HELPERS
# =============================================================================

PLOTTED_COUNTS = {}     # panel -> number of plotted marks (printed at the end)
FIGURE_NUMBERS = []     # every number printed on the figure, for checking
AUDIT_POINTS = {}       # axes -> list of (n, 2) arrays in data coordinates
AUDIT_BOXES = []        # (axes, artist, description)


def mm_to_pt(value_mm):
    return value_mm * 72.0 / 25.4


def add_axes_mm(fig, left, top, width, height):
    """Axes placed in millimetres; `top` is measured from the top of the page."""
    return fig.add_axes(
        [
            left / FIG_W_MM,
            1.0 - (top + height) / FIG_H_MM,
            width / FIG_W_MM,
            height / FIG_H_MM,
        ]
    )


def axes_size_pt(ax):
    box = ax.get_position()
    fig = ax.figure
    return (
        box.width * fig.get_figwidth() * 72.0,
        box.height * fig.get_figheight() * 72.0,
    )


def panel_letter(fig, letter, axes_left_mm, top_edge_mm):
    """Letter on the common margin, a fixed rise above the panel's top edge."""
    if PANEL_LETTER_CASE == "lower":
        letter = letter.lower()
    fig.text(
        (axes_left_mm - MARGIN_LEFT_MM + LETTER_X_MM) / FIG_W_MM,
        1.0 - (top_edge_mm - LETTER_RISE_MM) / FIG_H_MM,
        letter,
        fontsize=FS_LETTER,
        fontweight="bold",
        color=INK,
        ha="left",
        va="baseline",
    )


def plain_number(value):
    """1000 -> '1,000'; 0.01 -> '0.01'; never powers-of-ten notation."""
    return format(float(value), ",.10g").replace("-", MINUS)


def signed_number(value, decimals):
    return f"{value:+.{decimals}f}".replace("-", MINUS)


def linear_ticks(lower, upper, nbins=4):
    """Round tick positions and the one decimal count that suits all of them."""
    span = float(upper - lower)
    if not np.isfinite(span) or span <= 0:
        return np.array([lower]), 0
    exponent = int(np.floor(np.log10(span)))
    best = None
    for e in (exponent - 1, exponent, exponent + 1):
        for mantissa in (1, 2, 5):
            step = mantissa * 10.0 ** e
            first = int(np.ceil(lower / step - 1e-9))
            last = int(np.floor(upper / step + 1e-9))
            count = last - first + 1
            # 3 to 5 ticks, as close to `nbins` as possible; ties -> fewer.
            score = (not 3 <= count <= 5, abs(count - nbins), -step)
            if best is None or score < best[0]:
                best = (score, step, first, last)
    _, step, first, last = best
    ticks = np.arange(first, last + 1) * step
    decimals = max(0, int(np.ceil(-np.log10(step) - 1e-9)))
    return np.round(ticks, decimals + 2), decimals


def set_fixed_ticks(axis, ticks, decimals=None):
    axis.set_major_locator(FixedLocator(list(ticks)))
    if decimals is None:
        axis.set_major_formatter(FuncFormatter(lambda v, _: plain_number(v)))
    else:
        axis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:.{decimals}f}".replace("-", MINUS))
        )
    axis.set_minor_locator(NullLocator())


def set_log_ticks(axis, ticks):
    axis.set_major_locator(FixedLocator(list(ticks)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: plain_number(v)))
    axis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=50))
    axis.set_minor_formatter(NullFormatter())


def horizontal_grid(ax):
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which="major", color=GRID_GREY, linewidth=LW_GRID)


def reference_line(ax, x, y):
    ax.plot(x, y, color=REFERENCE_GREY, linewidth=LW_REFERENCE, zorder=1)


def register_points(ax, x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    keep = np.isfinite(x) & np.isfinite(y)
    AUDIT_POINTS.setdefault(ax, []).append(np.column_stack([x[keep], y[keep]]))


def settle_layout(fig):
    """Resolve artist positions without writing a file."""
    getattr(fig, "draw_without_rendering", fig.canvas.draw)()


def points_under(ax, artist, pad_pt=2.0):
    """Number of registered data points of `ax` covered by `artist`."""
    fig = ax.figure
    settle_layout(fig)
    box = artist.get_window_extent(fig.canvas.get_renderer())
    pad = pad_pt * fig.dpi / 72.0
    covered = 0
    for points in AUDIT_POINTS.get(ax, []):
        if not len(points):
            continue
        shown = ax.transData.transform(points)
        covered += int(
            np.sum(
                (shown[:, 0] >= box.x0 - pad)
                & (shown[:, 0] <= box.x1 + pad)
                & (shown[:, 1] >= box.y0 - pad)
                & (shown[:, 1] <= box.y1 + pad)
            )
        )
    return covered


def place_legend(ax, description, handles, locations, **legend_kwargs):
    """Try the listed positions in order and keep the first that covers no
    data point. If none is clear the legend goes above the axes."""
    for location in locations:
        legend = ax.legend(handles=handles, loc=location, **legend_kwargs)
        if points_under(ax, legend) == 0:
            AUDIT_BOXES.append((ax, legend, description))
            return legend
        legend.remove()
    legend_kwargs.pop("ncol", None)
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        ncol=len(handles),
        **legend_kwargs,
    )
    print(f"  note: {description} moved above the axes to stay clear of data.")
    return legend


def neutral_marker_handle(marker, label):
    return Line2D(
        [0], [0], linestyle="none", marker=marker, markersize=MS_KEY,
        markerfacecolor=INK_SECONDARY, markeredgecolor="none", label=label,
    )


def text_width_mm(fig, string, size):
    probe = fig.text(0, 0, string, fontsize=size)
    box = probe.get_window_extent(fig.canvas.get_renderer())
    probe.remove()
    return box.width / fig.dpi * 25.4


def draw_key_row(overlay, x_mm, y_mm, items, align="left"):
    """One row of key entries on the page overlay (millimetre coordinates).

    Each item is a dict with `label` and optionally `marker`, `ms`, `color`,
    `hollow`, `line` (a line style) and `lw`. An item with neither marker nor
    line is plain text, used as a key title. Returns the row width in mm.
    """
    fig = overlay.figure
    measured = []
    for item in items:
        has_line = item.get("line") is not None
        has_marker = item.get("marker") is not None
        if has_line:
            sample = KEY_LINE_MM
        elif has_marker:
            sample = item.get("ms", MS_KEY) * 25.4 / 72.0
        else:
            sample = 0.0
        label_width = text_width_mm(fig, item["label"], FS_SMALL)
        pad = KEY_PAD_MM if sample else 0.0
        measured.append((sample, pad, label_width))
    total = sum(s + p + w for s, p, w in measured) + KEY_ITEM_GAP_MM * (
        len(items) - 1
    )
    cursor = x_mm if align == "left" else x_mm - total
    for item, (sample, pad, label_width) in zip(items, measured):
        color = item.get("color", INK_SECONDARY)
        if item.get("line") is not None:
            overlay.plot(
                [cursor, cursor + sample], [y_mm, y_mm], color=color,
                linestyle=item["line"], linewidth=item.get("lw", LW_DATA),
                solid_capstyle="butt",
            )
        if item.get("marker") is not None:
            hollow = item.get("hollow", False)
            overlay.plot(
                [cursor + sample / 2.0], [y_mm], linestyle="none",
                marker=item["marker"], markersize=item.get("ms", MS_KEY),
                markerfacecolor=BACKGROUND if hollow else color,
                markeredgecolor=color if hollow else MARK_EDGE,
                markeredgewidth=LW_HOLLOW if hollow else LW_MARK_EDGE,
            )
        overlay.text(
            cursor + sample + pad, y_mm, item["label"], fontsize=FS_SMALL,
            color=item.get("text_color", INK), ha="left", va="center_baseline",
        )
        cursor += sample + pad + label_width + KEY_ITEM_GAP_MM
    return total


def two_line_title(fig, string, size, max_width_mm):
    """Break a title onto at most two lines at the most balanced space.
    The text is never truncated."""
    if text_width_mm(fig, string, size) <= max_width_mm:
        return string
    words = string.split(" ")
    best = string
    best_width = np.inf
    for cut in range(1, len(words)):
        first, second = " ".join(words[:cut]), " ".join(words[cut:])
        width = max(
            text_width_mm(fig, first, size), text_width_mm(fig, second, size)
        )
        if width < best_width:
            best, best_width = first + "\n" + second, width
    return best


def chapter_short_label(chapter_name):
    """'Blood and immune disorders' -> 'Blood & immune'."""
    label = re.sub(r"\s+(diseases|disorders)$", "", str(chapter_name).strip())
    return label.replace(" and ", " & ")


def swarm_offsets(y_pt, radius_pt, left_room_pt, right_room_pt, step_pt=0.5):
    """Deterministic sideways nudges (points) so circles do not overlap.
    Larger marks are placed first; each later mark moves the least distance,
    alternately right and left, that clears the marks already placed. A mark
    never leaves its own sub-column (`left_room_pt`, `right_room_pt`)."""
    count = len(y_pt)
    offsets = np.zeros(count)
    order = np.lexsort((np.arange(count), -np.asarray(radius_pt)))
    all_shifts = [0.0]
    for step in range(1, int(max(left_room_pt, right_room_pt) / step_pt) + 1):
        all_shifts.extend([step * step_pt, -step * step_pt])
    placed = []
    for index in order:
        best_shift, best_clearance = 0.0, -np.inf
        shifts = [
            shift for shift in all_shifts
            if shift == 0.0
            or -(left_room_pt - radius_pt[index]) <= shift
            <= right_room_pt - radius_pt[index]
        ]
        for shift in shifts:
            clearance = min(
                (
                    np.hypot(shift - offsets[j], y_pt[index] - y_pt[j])
                    - (radius_pt[index] + radius_pt[j])
                    for j in placed
                ),
                default=np.inf,
            )
            if clearance >= 0:
                best_shift = shift
                break
            if clearance > best_clearance:
                best_shift, best_clearance = shift, clearance
        offsets[index] = best_shift
        placed.append(index)
    return offsets


def spread_offsets(count):
    """Fixed, evenly filling sideways positions for the dots behind a box."""
    return ((np.arange(count) * GOLDEN) % 1.0 - 0.5) * DOT_SPREAD


def round_reference_values(low, high):
    """Three round values spanning [low, high] for a size key."""
    exponents = range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
    powers = [10.0 ** e for e in exponents if low <= 10.0 ** e <= high]
    if len(powers) >= 3:
        return powers[-3:]
    candidates = sorted(
        {m * 10.0 ** e for e in exponents for m in (1, 2, 5)
         if low <= m * 10.0 ** e <= high}
    )
    if len(candidates) <= 3:
        return candidates
    middle = np.sqrt(candidates[0] * candidates[-1])
    centre = min(candidates[1:-1], key=lambda v: abs(np.log(v / middle)))
    return [candidates[0], centre, candidates[-1]]


# =============================================================================
# 4. PANEL FUNCTIONS
# =============================================================================

# ---- Panel A: predicted versus observed 2-year risk by age and sex ----------

def draw_panel_a(fig, axes_grid, overlay, legend_y_mm, slot_width_mm):
    table = A[A["horizon_months"] == PRIMARY_HORIZON].copy()
    table = table[table["condition_name"].isin(endpoint_order)]
    half_bin = float(FIG2_AGE_BIN_WIDTH_YEARS) / 2.0

    ages = [table["age_bin_mid"].dropna().to_numpy(float)]
    if not dense.empty:
        ages.append(dense["index_age_years"].dropna().to_numpy(float))
    ages = np.concatenate(ages)
    x_limits = (
        (max(0.0, ages.min() - half_bin), ages.max() + half_bin)
        if ages.size else None
    )
    largest_bin = float(table["eligible_n"].max()) if len(table) else 1.0
    n_rows = len(axes_grid)
    marks = 0

    for index, ax in enumerate(np.ravel(axes_grid)):
        if index >= len(endpoint_order):
            ax.axis("off")
            continue
        endpoint = endpoint_order[index]
        row = index // len(axes_grid[0])
        column = index % len(axes_grid[0])
        endpoint_data = table[table["condition_name"] == endpoint]
        dense_endpoint = (
            dense[dense["condition_name"].astype(str).eq(endpoint)]
            if not dense.empty else pd.DataFrame()
        )
        drawn_values = []

        for sex in SEX_ORDER:
            colour = SEX_COLOR[sex]
            subset = endpoint_data[endpoint_data["sex"] == sex].sort_values(
                "age_bin_mid"
            )
            if len(subset):
                x = subset["age_bin_mid"].to_numpy(float)
                predicted = 100 * subset["mean_predicted_risk"].to_numpy(float)
                observed = 100 * subset["observed_aj_cif"].to_numpy(float)
                ax.plot(x, predicted, color=colour, linewidth=LW_DATA, zorder=3)

                area = np.full(len(subset), MS_OBSERVED ** 2)
                if A_SIZE_BY_PARTICIPANTS:
                    area = area * (
                        0.35 + 1.65 * subset["eligible_n"].to_numpy(float)
                        / largest_bin
                    )
                filled = np.ones(len(subset), dtype=bool)
                if A_HOLLOW_BELOW_EVENTS is not None:
                    filled = subset["event_n"].to_numpy(float) >= float(
                        A_HOLLOW_BELOW_EVENTS
                    )
                for mask, is_filled in ((filled, True), (~filled, False)):
                    if not mask.any():
                        continue
                    ax.scatter(
                        x[mask], observed[mask], s=area[mask],
                        marker=SEX_MARKER[sex],
                        facecolor=colour if is_filled else BACKGROUND,
                        edgecolor=MARK_EDGE if is_filled else colour,
                        linewidth=LW_MARK_EDGE if is_filled else LW_HOLLOW,
                        zorder=4, clip_on=False,
                    )
                marks += int(np.isfinite(observed).sum())
                drawn_values.extend(predicted[np.isfinite(predicted)])
                drawn_values.extend(observed[np.isfinite(observed)])

            if not dense_endpoint.empty:
                dense_sex = dense_endpoint[dense_endpoint["sex"].astype(str).eq(sex)]
                x_dense = pd.to_numeric(dense_sex["index_age_years"], errors="coerce")
                y_dense = 100 * pd.to_numeric(
                    dense_sex["predicted_risk"], errors="coerce"
                )
                valid = x_dense.notna() & y_dense.notna()
                ax.scatter(
                    x_dense[valid], y_dense[valid], s=DENSE_MS ** 2,
                    color=colour, marker=SEX_MARKER[sex], alpha=DENSE_ALPHA,
                    edgecolor="none", rasterized=True, zorder=2,
                )
                drawn_values.extend(y_dense[valid])

        # Independent y-axes: the ten risks differ more than tenfold. Zero is
        # always shown, with a sliver below it so a point at 0% is fully visible.
        highest = float(np.max(drawn_values)) if len(drawn_values) else 1.0
        top = highest * 1.08 if highest > 0 else 1.0
        ticks, decimals = linear_ticks(0.0, top, nbins=4)
        ax.set_ylim(-0.045 * top, top)
        set_fixed_ticks(ax.yaxis, ticks, decimals)

        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(NullLocator())
        if row == n_rows - 1:
            ax.set_xlabel("Age (years)")
        else:
            ax.tick_params(axis="x", labelbottom=False)
        if column == 0:
            ax.set_ylabel(f"{FIG2_PRIMARY_HORIZON_MONTHS // 12}-year risk (%)")

        title = f"{endpoint} ({ENDPOINT_ABBREVIATION[endpoint]})"
        ax.text(
            0.5, 1.0 + mm_to_pt(0.9) / axes_size_pt(ax)[1],
            two_line_title(fig, title, FS_LABEL, slot_width_mm),
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS_LABEL, color=INK, linespacing=1.1,
        )

    PLOTTED_COUNTS["A observed markers"] = marks
    draw_key_row(
        overlay,
        MARGIN_LEFT_MM,
        legend_y_mm,
        [
            {"label": sex, "marker": SEX_MARKER[sex], "ms": MS_OBSERVED,
             "color": SEX_COLOR[sex], "line": "-"}
            for sex in SEX_ORDER
        ]
        + [
            {"label": "Line: mean predicted risk", "line": "-"},
            {"label": "Marker: observed cumulative incidence "
                      f"(Aalen{EN_DASH}Johansen)", "marker": "o",
             "ms": MS_OBSERVED},
        ]
        + (
            [{"label": f"Hollow: fewer than {A_HOLLOW_BELOW_EVENTS} events",
              "marker": "o", "ms": MS_OBSERVED, "hollow": True}]
            if A_HOLLOW_BELOW_EVENTS is not None else []
        ),
    )


# ---- Panel B: AUROC by ICD chapter and sex ----------------------------------

def draw_panel_b(fig, ax, overlay, header_y_mm, right_edge_mm):
    table = C[
        (C["horizon_months"] == PRIMARY_HORIZON)
        & C["sex"].isin(SEX_ORDER)
        & C["auroc"].notna()
        & C["frequency_for_plot"].notna()
    ].copy()
    if table.empty:
        ax.text(0.5, 0.5, "No chapter AUROC data", ha="center", va="center",
                transform=ax.transAxes)
        PLOTTED_COUNTS["B markers"] = 0
        return

    chapters = table[["Chapter_Name", "ICD_Chapter"]].drop_duplicates().copy()
    chapters["sort"] = chapters["ICD_Chapter"].map(roman_value)
    chapters = chapters.sort_values(["sort", "Chapter_Name"])
    chapter_names = chapters["Chapter_Name"].tolist()
    chapter_x = {name: position for position, name in enumerate(chapter_names)}

    frequency = table["frequency_for_plot"].clip(lower=1).to_numpy(float)
    root_low, root_high = np.sqrt(frequency.min()), np.sqrt(frequency.max())

    def bubble_diameter(events):
        """Diameter in points; the same formula draws the data and the key."""
        share = (np.sqrt(np.clip(events, 1, None)) - root_low) / max(
            root_high - root_low, 1e-9
        )
        return BUBBLE_MIN_PT + (BUBBLE_MAX_PT - BUBBLE_MIN_PT) * share

    table["diameter"] = bubble_diameter(frequency)

    # Lower limit rounded down to a 0.1 step so every marker is inside.
    y_low = np.floor(table["auroc"].min() * 10 + 1e-9) / 10
    y_high = 1.0
    pad = 0.035 * (y_high - y_low)
    ax.set_ylim(y_low - pad, y_high + pad)
    ax.set_xlim(-0.5, len(chapter_names) - 0.5)
    width_pt, height_pt = axes_size_pt(ax)
    x_unit_pt = width_pt / len(chapter_names)
    y_unit_pt = height_pt / (y_high - y_low + 2 * pad)

    for position in range(1, len(chapter_names), 2):
        ax.axvspan(position - 0.5, position + 0.5, color=GRID_GREY,
                   alpha=BAND_ALPHA, linewidth=0, zorder=0)
    horizontal_grid(ax)
    if y_low <= 0.5 <= y_high:
        reference_line(ax, [-0.5, len(chapter_names) - 0.5], [0.5, 0.5])

    # Males in a left sub-column, females in a right sub-column, nudged
    # sideways only as far as needed. Identical on every run.
    table = table.sort_values(["Chapter_Name", "sex", "condition_name"])
    table["x"] = np.nan
    for (chapter, sex), group in table.groupby(["Chapter_Name", "sex"], sort=False):
        inner = abs(SEX_OFFSET[sex]) * x_unit_pt          # to the chapter centre
        outer = (0.5 - abs(SEX_OFFSET[sex])) * x_unit_pt  # to the chapter edge
        shift = swarm_offsets(
            group["auroc"].to_numpy(float) * y_unit_pt,
            group["diameter"].to_numpy(float) / 2.0,
            left_room_pt=outer if SEX_OFFSET[sex] < 0 else inner,
            right_room_pt=inner if SEX_OFFSET[sex] < 0 else outer,
        )
        table.loc[group.index, "x"] = (
            chapter_x[chapter] + SEX_OFFSET[sex] + shift / x_unit_pt
        )

    for sex in SEX_ORDER:
        subset = table[table["sex"] == sex].sort_values("diameter", ascending=False)
        ax.scatter(
            subset["x"], subset["auroc"], s=subset["diameter"] ** 2,
            marker=SEX_MARKER[sex], c=list(subset["condition_name"].map(condition_color)),
            edgecolor=MARK_EDGE, linewidth=LW_MARK_EDGE, zorder=3, clip_on=False,
        )
    register_points(ax, table["x"], table["auroc"])
    PLOTTED_COUNTS["B markers"] = int(len(table))

    ax.set_xticks(range(len(chapter_names)))
    ax.set_xticklabels(
        [chapter_short_label(name) for name in chapter_names],
        rotation=B_LABEL_ROTATION, ha="right", rotation_mode="anchor",
    )
    ax.tick_params(axis="x", length=0, pad=2.5)
    set_fixed_ticks(ax.yaxis, np.round(np.arange(y_low, y_high + 1e-9, 0.1), 1), 1)
    ax.set_ylabel("AUROC")

    # Keys and the annotation sit in the strip above the axes, never on data.
    key_values = round_reference_values(frequency.min(), frequency.max())
    used = draw_key_row(
        overlay, MARGIN_LEFT_MM, header_y_mm,
        [{"label": sex, "marker": SEX_MARKER[sex]} for sex in SEX_ORDER],
    )
    draw_key_row(
        overlay, MARGIN_LEFT_MM + used + 2.5 * KEY_ITEM_GAP_MM, header_y_mm,
        [{"label": f"Events within {PRIMARY_HORIZON // 12} years:"}]
        + [
            {"label": plain_number(value), "marker": "o",
             "ms": float(bubble_diameter(value))}
            for value in key_values
        ],
    )
    delta = Cpair["auroc_female_minus_male"].dropna()
    if len(delta):
        median_delta = float(delta.median())
        FIGURE_NUMBERS.append(
            ("B median AUROC difference (female − male)", median_delta)
        )
        draw_key_row(
            overlay, right_edge_mm, header_y_mm,
            [{"label": f"Median AUROC difference (female {MINUS} male): "
                       + signed_number(median_delta, 3),
              "text_color": INK_SECONDARY}],
            align="right",
        )


# ---- Panels C, D, E: distributions across horizons, by sex ------------------

def distribution_table(data, metric, letter):
    """One condition-level estimate per horizon and sex (unchanged filter)."""
    required = {"condition_name", "horizon_months", "sex", metric}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Panel {letter} lacks columns: {sorted(missing)}")
    table = data[
        data["sex"].isin(SEX_ORDER) & data["horizon_months"].isin(HORIZONS)
    ].copy()
    table[metric] = pd.to_numeric(table[metric], errors="coerce")
    table[metric] = table[metric].replace([np.inf, -np.inf], np.nan)
    table = table.dropna(subset=[metric])
    keys = ["condition_name", "sex", "horizon_months"]
    duplicates = table.duplicated(keys, keep=False)
    if duplicates.any():
        examples = table.loc[duplicates, keys].head(5).to_dict("records")
        raise ValueError(f"Panel {letter} has duplicated condition rows: {examples}")
    return table.sort_values(keys)


def draw_distribution(ax, table, metric, letter, scale=1.0):
    """Tukey boxes (whiskers 1.5 × IQR) over every condition-level dot."""
    plotted = 0
    counts = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        for sex in SEX_ORDER:
            subset = table[
                (table["horizon_months"] == horizon) & (table["sex"] == sex)
            ]
            values = scale * subset[metric].to_numpy(dtype=float)
            counts[(horizon, sex)] = int(values.size)
            if values.size == 0:
                continue
            plotted += 1
            centre = horizon_index + SEX_OFFSET[sex]
            colour = SEX_COLOR[sex]
            dots_x = centre + spread_offsets(values.size)
            ax.plot(
                dots_x, values, linestyle="none", marker=SEX_MARKER[sex],
                markersize=MS_DOT, markerfacecolor=colour,
                markeredgecolor="none", alpha=DOT_ALPHA, zorder=2, clip_on=False,
            )
            register_points(ax, dots_x, values)
            ax.boxplot(
                [values], positions=[centre], widths=BOX_WIDTH, whis=1.5,
                patch_artist=True, showfliers=False, showmeans=False,
                manage_ticks=False,
                boxprops={"facecolor": to_rgba(colour, BOX_FILL_ALPHA),
                          "edgecolor": colour, "linewidth": LW_BOX},
                medianprops={"color": colour, "linewidth": LW_MEDIAN},
                whiskerprops={"color": colour, "linewidth": LW_BOX},
                capprops={"color": colour, "linewidth": LW_BOX},
                zorder=3,
            )
    if plotted == 0:
        raise ValueError(f"Panel {letter} has no finite condition estimates")

    ax.set_xlim(-0.5, len(HORIZONS) - 0.5)
    ax.set_xticks(range(len(HORIZONS)))
    ax.set_xticklabels([plain_number(h / 12) for h in HORIZONS])
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_xlabel("Prediction horizon (years)")
    horizontal_grid(ax)
    PLOTTED_COUNTS[f"{letter} condition dots"] = int(sum(counts.values()))
    print(
        f"  Panel {letter} conditions per box (n): "
        + "; ".join(
            f"{plain_number(h / 12)} y "
            + "/".join(f"{sex[0]} {counts[(h, sex)]}" for sex in SEX_ORDER)
            for h in HORIZONS
        )
    )


def draw_panel_c(ax, table):
    draw_distribution(ax, table, "auroc", "C")
    lowest = float(table["auroc"].min())
    y_low = min(0.4, np.floor(lowest * 10 + 1e-9) / 10)
    pad = 0.03 * (1.0 - y_low)
    ax.set_ylim(y_low - pad, 1.0 + pad)
    set_fixed_ticks(ax.yaxis, np.round(np.arange(y_low, 1.0 + 1e-9, 0.2), 1), 1)
    reference_line(ax, [-0.5, len(HORIZONS) - 0.5], [0.5, 0.5])
    ax.set_ylabel("AUROC")
    place_legend(
        ax, "panel C sex key",
        [
            Line2D([0], [0], linestyle="none", marker=SEX_MARKER[sex],
                   markersize=MS_KEY, markerfacecolor=SEX_COLOR[sex],
                   markeredgecolor="none", label=sex)
            for sex in SEX_ORDER
        ],
        ["upper right", "lower left", "lower right"],
        handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25,
        borderpad=0.2, handlelength=1.0,
    )


def calibration_error_axis(values):
    """Shared y-axis for D and E. Returns (scale name, set-up function)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    highest = float(values.max())
    positive = values[values > 0]
    n_zero = int((values <= 0).sum())

    if DE_Y_SCALE == "linear" or positive.size == 0:
        top = highest * 1.08 if highest > 0 else 1.0
        ticks, decimals = linear_ticks(0.0, top, nbins=4)

        def apply(ax):
            ax.set_ylim(-0.03 * top, top)
            set_fixed_ticks(ax.yaxis, ticks, decimals)

        return "linear", apply, n_zero

    top = highest * 1.6
    high_exponent = int(np.floor(np.log10(top)))
    if n_zero:
        # Zero cannot sit on a log axis: linear below SYMLOG_LINTHRESH,
        # logarithmic above it.
        low_exponent = int(np.round(np.log10(SYMLOG_LINTHRESH)))
        ticks = [0.0] + [10.0 ** e for e in range(low_exponent, high_exponent + 1)]

        def apply(ax):
            ax.set_yscale("symlog", linthresh=SYMLOG_LINTHRESH,
                          linscale=SYMLOG_LINSCALE)
            ax.set_ylim(-0.12 * SYMLOG_LINTHRESH, top)
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(
                FuncFormatter(lambda v, _: plain_number(v))
            )
            ax.yaxis.set_minor_locator(
                SymmetricalLogLocator(base=10, linthresh=SYMLOG_LINTHRESH,
                                      subs=np.arange(2, 10))
            )
            ax.yaxis.set_minor_formatter(NullFormatter())

        return "symlog", apply, n_zero

    bottom = float(positive.min()) / 1.6
    low_exponent = int(np.ceil(np.log10(bottom)))
    exponents = list(range(low_exponent, high_exponent + 1))
    if len(exponents) > 6:
        exponents = exponents[::2]
    ticks = [10.0 ** e for e in exponents]

    def apply(ax):
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)
        set_log_ticks(ax.yaxis, ticks)

    return "log", apply, n_zero


def draw_panels_d_e(ax_d, ax_e, table_d, table_e):
    draw_distribution(ax_d, table_d, "ici", "D", scale=100)
    draw_distribution(ax_e, table_e, "e90", "E", scale=100)
    scale_name, apply, n_zero = calibration_error_axis(
        np.r_[100 * table_d["ici"].to_numpy(float),
              100 * table_e["e90"].to_numpy(float)]
    )
    for ax in (ax_d, ax_e):
        apply(ax)
    ax_d.set_ylabel(f"{CAL_MEAN_LABEL} ({CAL_ERROR_UNIT})")
    ax_e.set_ylabel(f"{CAL_E90_LABEL} ({CAL_ERROR_UNIT})")
    print(
        f"  Panels D and E share one y-axis: {scale_name} scale"
        + (f" (linear below {SYMLOG_LINTHRESH}, logarithmic above)"
           if scale_name == "symlog" else "")
        + f"; values equal to zero: {n_zero}."
    )


# ---- Panel F: ROC curves for the selected conditions ------------------------

def selected_condition_handles():
    """Key for F and G: sorted from highest to lowest AUROC."""
    table = F[F["horizon_months"] == PRIMARY_HORIZON]
    entries = []
    for endpoint in endpoint_order:
        auc = table.loc[table["condition_name"] == endpoint, "auroc"].dropna()
        value = float(auc.iloc[0]) if len(auc) else np.nan
        entries.append((endpoint, value))
    entries.sort(key=lambda item: (np.isnan(item[1]), -np.nan_to_num(item[1])))
    handles = []
    for endpoint, value in entries:
        style = endpoint_style[endpoint]
        label = ENDPOINT_ABBREVIATION[endpoint]
        if np.isfinite(value):
            label += f" {value:.2f}"
            FIGURE_NUMBERS.append((f"F AUROC {ENDPOINT_ABBREVIATION[endpoint]}", value))
        handles.append(
            Line2D([0], [0], color=style["color"], linestyle=style["linestyle"],
                   linewidth=LW_ROC, marker=style["marker"], markersize=MS_CURVE,
                   markeredgecolor=MARK_EDGE, markeredgewidth=LW_MARK_EDGE,
                   label=label)
        )
    return handles


def draw_panel_f(ax):
    table = F[
        (F["horizon_months"] == PRIMARY_HORIZON)
        & F["fpr"].notna()
        & F["tpr"].notna()
    ]
    reference_line(ax, [0, 1], [0, 1])
    curves = 0
    for position, endpoint in enumerate(endpoint_order):
        subset = table[table["condition_name"] == endpoint].sort_values("fpr")
        if not len(subset):
            continue
        curves += 1
        style = endpoint_style[endpoint]
        # Raw, unsmoothed curve; markers only identify the line.
        ax.plot(
            subset["fpr"], subset["tpr"], color=style["color"],
            linestyle=style["linestyle"], linewidth=LW_ROC,
            marker=style["marker"], markersize=MS_CURVE,
            markeredgecolor=MARK_EDGE, markeredgewidth=LW_MARK_EDGE,
            markevery=(0.05 + 0.021 * position, 0.26), zorder=3,
        )
    PLOTTED_COUNTS["F ROC curves"] = curves
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        set_fixed_ticks(axis, [0, 0.5, 1.0], 1)
    ax.set_xlabel(ROC_X_LABEL)
    ax.set_ylabel(ROC_Y_LABEL)
    legend = ax.legend(
        handles=selected_condition_handles(), title="AUROC",
        loc="upper left", bbox_to_anchor=(1.0, 1.0),
        borderaxespad=mm_to_pt(1.0) / FS_SMALL, borderpad=0.0,
        handlelength=2.7, handletextpad=0.5, labelspacing=0.22,
    )
    legend._legend_box.align = "left"


# ---- Panel G: calibration by decile for the selected conditions -------------

def draw_panel_g(ax):
    table = H[
        (H["horizon_months"] == PRIMARY_HORIZON)
        & H["calibration_stratum"].astype(str).eq("All")
        & H["predicted_mean"].notna()
        & H["observed_aj_cif"].notna()
    ].copy()
    table = table[table["condition_name"].isin(endpoint_order)]
    if table.empty:
        ax.text(0.5, 0.5, "No calibration data", ha="center", va="center",
                transform=ax.transAxes)
        PLOTTED_COUNTS["G decile markers"] = 0
        return
    table["x"] = 100 * table["predicted_mean"]
    table["y"] = 100 * table["observed_aj_cif"]
    off_scale = (table["x"] <= 0) | (table["y"] <= 0)
    use_log = G_AXIS_SCALE == "log" and off_scale.mean() <= G_ZERO_FRACTION_LIMIT
    if G_AXIS_SCALE == "log" and not use_log:
        print(
            f"  Panel G: {int(off_scale.sum())} of {len(table)} deciles are at "
            "zero, more than the allowed share; linear axes are used instead."
        )

    if use_log:
        positive = np.r_[table.loc[table["x"] > 0, "x"], table.loc[table["y"] > 0, "y"]]
        grid = np.array(sorted(m * 10.0 ** e for e in range(-8, 4) for m in (1, 3)))
        low = grid[grid <= positive.min() * 0.85].max()
        high = grid[grid >= positive.max() * 1.15].min()
        ticks = grid[(grid >= low) & (grid <= high)]
        if len(ticks) > 5:
            ticks = np.array(
                [t for t in ticks if np.isclose(np.log10(t), np.round(np.log10(t)))]
            )
        if len(ticks) > 4:
            ticks = ticks[::-1][::2][::-1]   # every second decade, from the top
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            set_log_ticks(axis, ticks)
    else:
        low = 0.0
        high = float(np.nanmax(np.r_[table["x"], table["y"]])) * 1.08
        ticks, decimals = linear_ticks(0.0, high, nbins=4)
        for axis in (ax.xaxis, ax.yaxis):
            set_fixed_ticks(axis, ticks, decimals)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    reference_line(ax, [low, high], [low, high])

    hollow_total = 0
    for endpoint in endpoint_order:
        subset = table[table["condition_name"] == endpoint].sort_values("x")
        if not len(subset):
            continue
        style = endpoint_style[endpoint]
        on_scale = (
            (subset["x"] > 0) & (subset["y"] > 0) if use_log
            else pd.Series(True, index=subset.index)
        )
        shown = subset[on_scale]
        ax.plot(
            shown["x"], shown["y"], color=style["color"],
            linestyle=style["linestyle"], linewidth=LW_CALIBRATION,
            marker=style["marker"], markersize=MS_CURVE,
            markeredgecolor=MARK_EDGE, markeredgewidth=LW_MARK_EDGE,
            zorder=3, clip_on=False,
        )
        edge = subset[~on_scale]
        if len(edge):
            # Deciles at zero are kept: hollow marker on the axis edge.
            ax.plot(
                edge["x"].where(edge["x"] > 0, low),
                edge["y"].where(edge["y"] > 0, low),
                linestyle="none", marker=style["marker"], markersize=MS_CURVE,
                markerfacecolor=BACKGROUND, markeredgecolor=style["color"],
                markeredgewidth=LW_HOLLOW, zorder=4, clip_on=False,
            )
            hollow_total += len(edge)
    PLOTTED_COUNTS["G decile markers"] = int(len(table))
    PLOTTED_COUNTS["G of which hollow at the axis edge (zero)"] = int(hollow_total)
    ax.set_xlabel(f"Predicted {PRIMARY_HORIZON // 12}-year risk (%)")
    ax.set_ylabel(
        f"Observed {PRIMARY_HORIZON // 12}-year\ncumulative incidence (%)"
    )
    print(
        f"  Panel G axes: {'logarithmic' if use_log else 'linear'}; "
        f"deciles drawn hollow on the axis edge (zero): {hollow_total} of {len(table)}."
    )


# ---- Panel H: target model versus comparator --------------------------------

def draw_panel_h(ax):
    table = panel_h_comparison.rename(
        columns={
            "condition": "condition_name",
            "target_auroc": "target",
            "comparator_auroc": "comparator",
        }
    ).dropna(subset=["target", "comparator"])
    if table.empty:
        raise RuntimeError(
            "Panel H has no finite paired CRESCENT/Delphi-2M AUROC values."
        )
    lowest = float(min(table["comparator"].min(), table["target"].min()))
    highest = float(max(table["comparator"].max(), table["target"].max()))
    low = max(0.0, np.floor((lowest - 0.005) * 10) / 10)
    high = min(1.0, np.ceil((highest + 0.005) * 10) / 10)
    if high <= low:
        raise RuntimeError("Panel H AUROC axis limits are degenerate.")
    step = 0.1 if (high - low) > 0.25 else 0.05
    ticks = np.round(np.arange(low, high + 1e-9, step), 2)
    decimals = 1 if step == 0.1 else 2
    pad = 0.03 * (high - low)
    reference_line(ax, [low, high], [low, high])
    ax.scatter(
        table["comparator"], table["target"], s=MS_POINT ** 2,
        c=list(table["condition_name"].map(condition_color)),
        edgecolor=MARK_EDGE, linewidth=LW_MARK_EDGE, zorder=3, clip_on=False,
    )
    register_points(ax, table["comparator"], table["target"])
    PLOTTED_COUNTS["H conditions"] = int(len(table))
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    for axis in (ax.xaxis, ax.yaxis):
        set_fixed_ticks(axis, ticks, decimals)
    ax.set_xlabel(f"{COMPARATOR_MODEL} AUROC")
    ax.set_ylabel(f"{TARGET_MODEL} AUROC")

    total = int(len(table))
    target_higher = int((table["target"] > table["comparator"]).sum())
    comparator_higher = int((table["target"] < table["comparator"]).sum())
    FIGURE_NUMBERS.append((f"H {TARGET_MODEL} higher (of {total})", target_higher))
    FIGURE_NUMBERS.append(
        (f"H {COMPARATOR_MODEL} higher (of {total})", comparator_higher)
    )
    if target_higher + comparator_higher != total:
        print(f"  Panel H: {total - target_higher - comparator_higher} exact ties.")
    for x, y, ha, va, name, count in (
        (0.04, 0.985, "left", "top", TARGET_MODEL, target_higher),
        (0.985, 0.03, "right", "bottom", COMPARATOR_MODEL, comparator_higher),
    ):
        note = ax.text(
            x, y, f"{name} higher:\n{count} of {total}", transform=ax.transAxes,
            ha=ha, va=va, fontsize=FS_SMALL, color=INK_SECONDARY, linespacing=1.15,
        )
        AUDIT_BOXES.append((ax, note, f"panel H note ({name})"))


# ---- Panel I: calibration error versus number of events ---------------------

def draw_panel_i(ax):
    table = Hsummary.copy()
    if "endpoint_type" in table.columns:
        table = table[table["endpoint_type"].astype(str).eq("condition")]
    if "risk_type" in table.columns:
        preferred = table[table["risk_type"].astype(str).eq("death_adjusted")]
        if len(preferred):
            table = preferred
    table = table[table["calibration_stratum"].astype(str).eq("All")]
    frequency = B[
        (B["horizon_months"] == PRIMARY_HORIZON) & B["sex"].astype(str).eq("All")
    ][["condition_name", "event_n_2yr_all"]].drop_duplicates("condition_name")
    table = table.merge(frequency, on="condition_name", how="left")
    table = table[table["event_n_2yr_all"].fillna(0) > 0].copy()
    if table.empty:
        ax.text(0.5, 0.5, "No calibration-error data", ha="center", va="center",
                transform=ax.transAxes)
        PLOTTED_COUNTS["I conditions"] = 0
        return

    events = table["event_n_2yr_all"].to_numpy(float)
    mean_error = 100 * table["ici"].to_numpy(float)
    e90_error = 100 * table["e90"].to_numpy(float)
    colours = list(table["condition_name"].map(condition_color))
    both = np.isfinite(mean_error) & np.isfinite(e90_error)
    for x, low, high, colour in zip(
        events[both], mean_error[both], e90_error[both],
        np.asarray(colours, dtype=object)[both],
    ):
        ax.plot([x, x], [low, high], color=colour, linewidth=LW_STEM,
                alpha=STEM_ALPHA, zorder=2)
    for values, marker in ((mean_error, "o"), (e90_error, "^")):
        keep = np.isfinite(values)
        ax.scatter(
            events[keep], values[keep], s=MS_POINT ** 2, marker=marker,
            c=list(np.asarray(colours, dtype=object)[keep]),
            edgecolor=MARK_EDGE, linewidth=LW_MARK_EDGE, zorder=3, clip_on=False,
        )
        register_points(ax, events[keep], values[keep])
    PLOTTED_COUNTS["I conditions"] = int(len(table))
    PLOTTED_COUNTS[f"I {CAL_MEAN_LABEL} markers"] = int(np.isfinite(mean_error).sum())
    PLOTTED_COUNTS[f"I {CAL_E90_LABEL} markers"] = int(np.isfinite(e90_error).sum())

    ax.set_xscale("log")
    low_exponent = int(np.floor(np.log10(events.min())))
    high_exponent = int(np.floor(np.log10(events.max())))
    ax.set_xlim(10.0 ** low_exponent / 1.35, events.max() * 1.35)
    set_log_ticks(
        ax.xaxis, [10.0 ** e for e in range(low_exponent, high_exponent + 1)]
    )
    highest = float(np.nanmax(np.r_[mean_error, e90_error]))
    top = highest * 1.12 if highest > 0 else 1.0
    ticks, decimals = linear_ticks(0.0, top, nbins=4)
    ax.set_ylim(-0.035 * top, top)
    set_fixed_ticks(ax.yaxis, ticks, decimals)
    ax.set_xlabel(f"Events within {PRIMARY_HORIZON // 12} years")
    ax.set_ylabel(f"Calibration error\n({CAL_ERROR_UNIT})")
    place_legend(
        ax, f"panel I {CAL_MEAN_LABEL}/{CAL_E90_LABEL} key",
        [neutral_marker_handle("o", CAL_MEAN_LABEL),
         neutral_marker_handle("^", CAL_E90_LABEL)],
        ["upper left", "center left"],
        handletextpad=0.2, borderaxespad=0.2, labelspacing=0.25,
        borderpad=0.2, handlelength=1.0,
    )


# ---- Chapter key: one for the whole figure ----------------------------------

def draw_chapter_key(overlay, top_mm, left_mm, right_mm):
    """KEY_ROWS rows, read left to right in chapter order, spanning the
    width. Each label carries its ICD chapter numeral."""
    fig = overlay.figure
    names = list(chapter_order)
    icd_of = chapter_meta.set_index("Chapter_Name")["ICD_Chapter"].to_dict()

    def key_label(name):
        numeral = re.search(r"Chapter\s+([IVXLCDM]+)", str(icd_of[name]), flags=re.I)
        if not numeral:
            return name
        return CHAPTER_KEY_FORMAT.format(numeral=numeral.group(1).upper(), name=name)

    per_row = int(np.ceil(len(names) / KEY_ROWS)) if names else 0
    per_row = max(per_row, min(KEY_COLS, len(names)))
    marker_mm = MS_KEY * 25.4 / 72.0
    column_width = []
    for column in range(per_row):
        labels = names[column::per_row]
        column_width.append(
            marker_mm + KEY_PAD_MM
            + max(text_width_mm(fig, key_label(label), FS_SMALL) for label in labels)
        )
    free = (right_mm - left_mm) - sum(column_width)
    if free < 0:
        warnings.warn(
            f"The chapter key is {-free:.1f} mm wider than the figure; "
            "shorten Chapter_Name labels or reduce KEY_COLS."
        )
    gap = max(free, 0.0) / max(per_row - 1, 1)
    column_left = left_mm + np.r_[0.0, np.cumsum(np.array(column_width[:-1]) + gap)]
    for index, name in enumerate(names):
        row, column = divmod(index, per_row)
        y = top_mm + (row + 0.5) * KEY_ROW_MM
        x = column_left[column]
        overlay.plot(
            [x + marker_mm / 2.0], [y], linestyle="none", marker="o",
            markersize=MS_KEY, markerfacecolor=chapter_color[name],
            markeredgecolor="none",
        )
        overlay.text(
            x + marker_mm + KEY_PAD_MM, y, key_label(name), fontsize=FS_SMALL,
            color=INK,
            ha="left", va="center_baseline",
        )


# ---- Automatic layout audit --------------------------------------------------

def layout_audit(fig):
    """Report text sizes, text collisions and keys that cover data points."""
    settle_layout(fig)
    renderer = fig.canvas.get_renderer()
    undrawn = set()   # tick labels of switched-off axes or outside the view
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not ax.axison or not (low <= tick.get_loc() <= high):
                    undrawn.update((tick.label1, tick.label2))
    texts = [
        t for t in fig.findobj(Text)
        if t.get_visible() and t.get_text().strip() and t not in undrawn
    ]
    boxes = []
    for t in texts:
        try:
            box = t.get_window_extent(renderer)
        except Exception:
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((t, box))
    sizes = sorted({round(float(t.get_fontsize()), 2) for t, _ in boxes})
    print("LAYOUT AUDIT")
    print("  font:", FONT_USED, "| text sizes used (pt):", sizes)
    if sizes and (sizes[0] < FS_MIN or sizes[-1] > FS_LETTER):
        print("  FAIL: a text size is outside", FS_MIN, "to", FS_LETTER, "pt")

    page = fig.bbox
    outside = [
        t.get_text() for t, b in boxes
        if b.x0 < page.x0 - 0.5 or b.x1 > page.x1 + 0.5
        or b.y0 < page.y0 - 0.5 or b.y1 > page.y1 + 0.5
    ]
    print("  text cut off by the page edge:", outside if outside else "none")

    shrink = 0.6 * fig.dpi / 72.0
    collisions = []
    upright = [(t, b) for t, b in boxes if (t.get_rotation() % 180) == 0]
    for i in range(len(upright)):
        ti, bi = upright[i]
        for j in range(i + 1, len(upright)):
            tj, bj = upright[j]
            if (
                bi.x0 + shrink < bj.x1 and bj.x0 + shrink < bi.x1
                and bi.y0 + shrink < bj.y1 and bj.y0 + shrink < bi.y1
            ):
                collisions.append((ti.get_text(), tj.get_text()))
    print(
        "  overlapping text pairs:",
        collisions[:12] if collisions else "none",
    )
    covered = [
        (description, points_under(ax, artist))
        for ax, artist, description in AUDIT_BOXES
    ]
    covered = [(d, n) for d, n in covered if n]
    print("  keys or notes covering data points:", covered if covered else "none")
    return {"outside": outside, "collisions": collisions, "covered": covered}


# =============================================================================
# 5. LAYOUT BLOCK
# =============================================================================

CONTENT_W_MM = FIG_W_MM - MARGIN_LEFT_MM - MARGIN_RIGHT_MM
RIGHT_EDGE_MM = FIG_W_MM - MARGIN_RIGHT_MM
A_COLS, A_ROWS = 5, 2
A_AXES_W_MM = (CONTENT_W_MM - (A_COLS - 1) * A_COL_GAP_MM) / A_COLS
CDE_AXES_W_MM = (CONTENT_W_MM - 2 * CDE_COL_GAP_MM) / 3
FGHI_SIDE_MM = (
    CONTENT_W_MM - F_LEGEND_COL_MM - G_GAP_MM - H_GAP_MM - I_GAP_MM
) / 4

# Vertical positions, measured in millimetres from the top of the page.
A_LEGEND_Y_MM = MARGIN_TOP_MM + A_LEGEND_MM / 2
A_TOP_EDGE_MM = MARGIN_TOP_MM + A_LEGEND_MM
A_AXES_TOP_MM = [
    A_TOP_EDGE_MM + A_TITLE_MM
    + row * (A_AXES_H_MM + A_ROW_GAP_MM + A_TITLE_MM)
    for row in range(A_ROWS)
]
B_AXES_TOP_MM = A_AXES_TOP_MM[-1] + A_AXES_H_MM + XLABEL_ZONE_MM + HEADER_MM
CDE_AXES_TOP_MM = B_AXES_TOP_MM + B_AXES_H_MM + B_TICK_ZONE_MM + HEADER_MM
FGHI_AXES_TOP_MM = CDE_AXES_TOP_MM + CDE_AXES_H_MM + XLABEL_ZONE_MM + HEADER_MM
KEY_TOP_MM = FGHI_AXES_TOP_MM + FGHI_SIDE_MM + XLABEL_ZONE_MM + KEY_GAP_MM
FIG_H_MM = KEY_TOP_MM + KEY_ROWS * KEY_ROW_MM + MARGIN_BOTTOM_MM
if FIG_H_MM > FIG_MAX_H_MM:
    raise ValueError(
        f"Figure height {FIG_H_MM:.1f} mm exceeds the {FIG_MAX_H_MM:.0f} mm limit."
    )

fig = plt.figure(
    figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=PREVIEW_DPI,
    facecolor=BACKGROUND,
)

# Page overlay in millimetres (origin top left) for keys and the chapter key.
overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
overlay.set_xlim(0, FIG_W_MM)
overlay.set_ylim(FIG_H_MM, 0)
overlay.axis("off")

axes_a = [
    [
        add_axes_mm(
            fig, MARGIN_LEFT_MM + column * (A_AXES_W_MM + A_COL_GAP_MM),
            A_AXES_TOP_MM[row], A_AXES_W_MM, A_AXES_H_MM,
        )
        for column in range(A_COLS)
    ]
    for row in range(A_ROWS)
]
axis_b = add_axes_mm(fig, MARGIN_LEFT_MM, B_AXES_TOP_MM, CONTENT_W_MM, B_AXES_H_MM)
CDE_LEFT_MM = [
    MARGIN_LEFT_MM + i * (CDE_AXES_W_MM + CDE_COL_GAP_MM) for i in range(3)
]
axis_c, axis_d, axis_e = [
    add_axes_mm(fig, left, CDE_AXES_TOP_MM, CDE_AXES_W_MM, CDE_AXES_H_MM)
    for left in CDE_LEFT_MM
]
FGHI_LEFT_MM = [MARGIN_LEFT_MM]
for _gap in (F_LEGEND_COL_MM + G_GAP_MM, H_GAP_MM, I_GAP_MM):
    FGHI_LEFT_MM.append(FGHI_LEFT_MM[-1] + FGHI_SIDE_MM + _gap)
axis_f, axis_g, axis_h, axis_i = [
    add_axes_mm(fig, left, FGHI_AXES_TOP_MM, FGHI_SIDE_MM, FGHI_SIDE_MM)
    for left in FGHI_LEFT_MM
]

panel_letter(fig, "A", MARGIN_LEFT_MM, A_TOP_EDGE_MM)
panel_letter(fig, "B", MARGIN_LEFT_MM, B_AXES_TOP_MM)
for _letter, _left in zip("CDE", CDE_LEFT_MM):
    panel_letter(fig, _letter, _left, CDE_AXES_TOP_MM)
for _letter, _left in zip("FGHI", FGHI_LEFT_MM):
    panel_letter(fig, _letter, _left, FGHI_AXES_TOP_MM)

# Tables for C, D, E (filters unchanged from the previous version).
c_data = E.copy()
if "endpoint_type" in c_data.columns:
    c_data = c_data[c_data["endpoint_type"].astype(str).eq("condition")]
calibration_data = ALL_CAL.copy()
if "endpoint_type" in calibration_data.columns:
    calibration_data = calibration_data[
        calibration_data["endpoint_type"].astype(str).eq("condition")
    ]
if "risk_type" in calibration_data.columns:
    preferred = calibration_data[
        calibration_data["risk_type"].astype(str).eq("death_adjusted")
    ]
    if len(preferred):
        calibration_data = preferred

print("FIGURE-2 DRAWING LOG")
draw_panel_a(
    fig, axes_a, overlay, A_LEGEND_Y_MM,
    A_AXES_W_MM + 2 * min(MARGIN_RIGHT_MM, A_COL_GAP_MM / 2) - 1.0,
)
draw_panel_b(
    fig, axis_b, overlay, B_AXES_TOP_MM - HEADER_MM / 2 + 0.6, RIGHT_EDGE_MM
)
draw_panel_c(axis_c, distribution_table(c_data, "auroc", "C"))
draw_panels_d_e(
    axis_d, axis_e,
    distribution_table(calibration_data, "ici", "D"),
    distribution_table(calibration_data, "e90", "E"),
)
draw_panel_f(axis_f)
draw_panel_g(axis_g)
draw_panel_h(axis_h)
draw_panel_i(axis_i)
draw_chapter_key(overlay, KEY_TOP_MM, MARGIN_LEFT_MM - 8.0, RIGHT_EDGE_MM)

print("  Marks drawn per panel:")
for _name, _count in PLOTTED_COUNTS.items():
    print(f"    {_name}: {_count:,}")
print("  Numbers printed on the figure (computed from the loaded tables):")
for _name, _value in FIGURE_NUMBERS:
    print(f"    {_name}: {_value:.6g}")
LAYOUT_AUDIT = layout_audit(fig)


# =============================================================================
# Final source/model sanity check
# =============================================================================

if TARGET_METRICS_PATH != Path(CRESCENT_METRICS_PATH).expanduser().resolve():
    raise RuntimeError("Final CRESCENT metrics-path check failed.")
if COMPARATOR_METRICS_PATH != Path(
    FIG2_COMPARATOR_METRICS_PATH
).expanduser().resolve():
    raise RuntimeError("Final comparator metrics-path check failed.")
if FIG2_DIR.parent != FIG2_RUN_DIR:
    raise RuntimeError("Final Figure-2 input-directory check failed.")
source_manifest_path.relative_to(FIG2_RUN_DIR)
all_cal_path.relative_to(FIG2_RUN_DIR)

for key, filename in FILES.items():
    (FIG2_DIR / filename).resolve().relative_to(FIG2_RUN_DIR)

print("FINAL FIGURE MODEL:               ", TARGET_MODEL)
print("FINAL LOCKED CRESCENT RUN:        ", FIG2_RUN_DIR)
print("FINAL FIGURE-2 INPUT DIRECTORY:   ", FIG2_DIR)
print("FINAL CALIBRATION FILE:           ", all_cal_path)
print("FINAL FIGURE-2 OUTPUT DIRECTORY:  ", FIG2_OUTPUT_DIR)

# =============================================================================
# 6. EXPORT BLOCK
# =============================================================================
# Saved at the exact page size: no tight bounding box, which would change the
# final width. Text stays editable (TrueType in the PDF, text in the SVG).

output_stem = (
    FIG2_OUTPUT_DIR
    / f"figure2_{TARGET_MODEL}_locked_test_comparison"
)
output_png = output_stem.with_name(output_stem.name + f"_{DPI}dpi.png")
output_pdf = output_stem.with_suffix(".pdf")
output_svg = output_stem.with_suffix(".svg")



def save_raster(path, **pil_options):
    """RGB (no alpha channel), white background, resolution stored in the file."""
    import io
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, facecolor=BACKGROUND)
    buffer.seek(0)
    Image.open(buffer).convert("RGB").save(path, dpi=(DPI, DPI), **pil_options)


save_raster(output_png)
fig.savefig(output_pdf, dpi=DPI, facecolor=BACKGROUND)
fig.savefig(output_svg, dpi=DPI, facecolor=BACKGROUND)
if SAVE_TIFF:
    output_tiff = output_stem.with_name(output_stem.name + f"_{DPI}dpi.tiff")
    save_raster(output_tiff, compression="tiff_lzw")

plt.show()

print(f"Page size: {FIG_W_MM:.1f} x {FIG_H_MM:.1f} mm")
print("Saved:", output_png)
print("Saved:", output_pdf)
print("Saved:", output_svg)
if SAVE_TIFF:
    print("Saved:", output_tiff)
