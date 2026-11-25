"""Fig. 5: from population patterns to individual patients.

Redraws manuscript Fig. 5 and its extended-data figures from the export in `source_data/` and
writes PNG, PDF, SVG, plotted source data and draft legends to `output/`. No model is loaded and
no UK Biobank access is needed.

    python figures/figure5/plot.py

Panels e and f show two synthetic example patients (profiles with status "synthetic": no real
participant identifiers or observed outcomes, so the outcome lane is omitted). If the patient
tables are not in `source_data/`, panels a-d and the extended-data figures are still drawn.

Code from the notebook `Fig5_plotting_v2_two_synthetic`, cells copied unchanged with cell numbers kept, apart
from three marked lines: default folders (cell 2), the manifest check for the absent patient
tables (cell 4), and drawing without panels e, f when those tables are absent (cell 11).
`source_data/` is filled by `collect_source_data.py`.
"""

import matplotlib

matplotlib.use("Agg")  # draw without a display; files are written to output/


# =============================================================================
# Notebook cell 2
# =============================================================================

from pathlib import Path
import os

# Repository paths: the export is read from source_data/ and figures are written to output/.
HERE = Path(__file__).resolve().parent
EXPORT_DIR = Path(os.getenv("FIG5_EXPORT_DIR", str(HERE / "source_data"))).expanduser().resolve()
FIGURE_DIR = Path(os.getenv("FIG5_FIGURE_DIR", str(HERE / "output"))).expanduser().resolve()
HORIZONS = (60, 24, 120)                 # first = main figure; panels e, f always use the patients' horizon

FIGURE_WIDTH_IN = 7.2                    # double-column width (~183 mm)
FONT_SIZE = 7
SEX_LABELS = {"code 0": "Female", "code 1": "Male", "all": "All"}   # CHECK against static_covariate_vocabulary.json
GROUP_LABELS = {"Diagnosis history": "Diagnoses", "Landmark age": "Age", "BMI history": "BMI",
                "Smoking history": "Smoking", "Alcohol history": "Alcohol", "CRP history": "CRP",
                "Ethnicity": "Ethnicity", "Townsend deprivation": "Deprivation"}

# Panel a (full matrix, original layout)
A_FULL_METRIC = "mean_diff_pp"           # "mean_diff_pp" (original) or "risk_ratio"
A_FULL_SEXES = ("code 0", "code 1")      # heatmaps side by side
A_FULL_COLOUR_QUANTILE = 0.99            # colour limit = this quantile of |values|, shared by both heatmaps
A_FULL_SHOW_CI_DOTS = False              # dot where the 95% interval excludes no effect
ICD_CHAPTER_COLOURS = ["#1F4E9C", "#4FA3D1", "#0F8B8D", "#35BFAE", "#2E8B4F", "#8CBF3F", "#5F6B21", "#D9C22E",
                       "#E8912A", "#D4462B", "#6B4B2E", "#A32B4A", "#D96BA0", "#7B3F9E", "#4B4FA8"]   # original palette
# Extended data: 15-condition heatmap
A_METRIC = "risk_ratio"                  # "risk_ratio", "model_hazard_ratio" or "mean_diff_pp"
A_SEX = "all"                            # "all", "code 0" or "code 1" (both sexes are in extended data)
A_ANNOTATE_ABOVE = 1.5                   # print the value in cells with ratio >= this (or <= 1/this); None = never
# Panel b: observed vs predicted risk ratios for the shown conditions
B_MIN_EVENTS = 10                        # pairs with fewer observed outcome events among people with the prior diagnosis are not drawn
B_LABELS = 5                             # pairs labelled (largest predicted ratios), shown as dark points with labels beside them
# Panel c
C_HIGHLIGHT = 4                          # outcomes highlighted in colour (largest age effect); max 4
# Panels e, f: diagnoses labelled on each timeline (most recent first)
PATIENT_MAX_DX_LABELS = 6
PATIENT_LOOKBACK_YEARS = 15              # records older than this (years before the landmark) are not drawn
FOREST_PAIRS = 20                        # extended data: strongest pairs shown in the forest plot


# =============================================================================
# Notebook cell 4
# =============================================================================

import hashlib
import json
import re
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

manifest = json.loads((EXPORT_DIR / "manifest.json").read_text())
if manifest.get("schema") != "crescent.fig5.v2":
    raise ValueError(f"Expected a Fig5_data_generator_v2 export; found schema {manifest.get('schema')!r}.")
# Repository: the participant-level patient tables (panels e, f) are not published, so they may be
# absent; every file that is present must still match the export manifest.
PATIENT_FILES = {"fig5D_patient_profiles.csv", "fig5D_patient_shapley.csv",
                 "fig5D_patient_horizon_risks.csv", "fig5D_patient_timelines.csv"}
for name, expected in manifest["files"].items():
    if name in PATIENT_FILES and not (EXPORT_DIR / name).exists():
        continue
    if hashlib.sha256((EXPORT_DIR / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"{name} changed after export (hash mismatch); regenerate the export.")

def table(name):
    path = EXPORT_DIR / name
    return pd.read_csv(path, keep_default_na=True) if path.exists() and path.stat().st_size > 1 else None

CAT = table("condition_catalogue.csv")
A, COX, B, C = (table(f) for f in ("fig5A_from_to.csv", "fig5A_cox_pairs.csv", "fig5B_age_ale.csv", "fig5C_lookback.csv"))
AFULL = table("fig5A_from_to_all_conditions.csv")
CAL = table("fig5B_observed_vs_predicted.csv")
PROFILES, SHAP, RISKS, TIMELINES = (table(f) for f in ("fig5D_patient_profiles.csv", "fig5D_patient_shapley.csv",
                                                        "fig5D_patient_horizon_risks.csv", "fig5D_patient_timelines.csv"))
for frame in (A, AFULL, B, C, CAL):
    for column in ("target_id", "condition_id", "source_id"):
        if frame is not None and column in frame:
            frame[column] = frame[column].astype(str)
if COX is not None:
    COX["source_id"], COX["target_id"] = COX.source_id.astype(str), COX.target_id.astype(str)
CAT["condition_id"] = CAT.condition_id.astype(str)
CAT["chapter_roman"] = CAT.chapter_roman.fillna("").astype(str)
def _roman(text):
    match = re.match(r"\s*(?:chapter|ch\.?)?\s*([IVXLC]+)\b", str(text), flags=re.IGNORECASE)
    return match[1].upper() if match and str(text).strip().lower() != "death" else ""
missing_roman = CAT.chapter_roman.str.strip().eq("") & CAT.condition_id.ne("death")
CAT.loc[missing_roman, "chapter_roman"] = CAT.loc[missing_roman, "icd_chapter"].map(_roman)
ROMAN_VALUE = {r: i + 1 for i, r in enumerate("I II III IV V VI VII VIII IX X XI XII XIII XIV XV XVI XVII XVIII XIX XX XXI XXII".split())}
CAT["chapter_rank"] = [ROMAN_VALUE.get(r, 99) if k != "death" else 999 for k, r in zip(CAT.condition_id, CAT.chapter_roman)]
if (CAT.chapter_roman.eq("") & CAT.condition_id.ne("death")).any():
    warnings.warn("Some ICD chapters have no Roman numeral; check the ICD_Chapter column of the chapter CSV.")
ABBR = dict(zip(CAT.condition_id, CAT.abbreviation))
CHAPTER = dict(zip(CAT.condition_id, CAT.chapter_roman))
CHAPTER_NAME = dict(zip(CAT.condition_id, CAT.chapter_name))
# Display order: by ICD-10 chapter, then abbreviation; death last.
if "selected" not in CAT:
    CAT["selected"] = True
CAT["selected"] = CAT.selected.astype(str).str.lower().eq("true")
if "chapter_csv_order" not in CAT:
    CAT["chapter_csv_order"] = CAT.order
# Full matrix (panel a): ordered by chapter, then by the chapter CSV's row order; death last.
FULL_TARGETS = CAT.sort_values(["chapter_rank", "chapter_csv_order"]).condition_id.tolist()
FULL_TARGETS = [k for k in FULL_TARGETS if k != "death"] + ["death"]
FULL_SOURCES = [k for k in FULL_TARGETS if k != "death" and (AFULL is None or k in set(AFULL.source_id))]
TARGET_ORDER = CAT[CAT.selected].sort_values(["chapter_rank", "abbreviation"]).condition_id.tolist()
TARGET_ORDER = [k for k in TARGET_ORDER if k != "death"] + (["death"] if "death" in TARGET_ORDER else [])
SOURCE_ORDER = [k for k in TARGET_ORDER if k != "death"]
DEV = bool(manifest["development_subset"])
REF_AGE = float(manifest["reference_age_years"])
PATIENT_TARGET = manifest["patients"]["target"]
print(f"Export {manifest['created_utc']}: {manifest['n_test_participants']:,} test participants, "
      f"{len(SOURCE_ORDER)} conditions, bootstrap {manifest['bootstrap']['n']}" + ("  [DEVELOPMENT SUBSET]" if DEV else ""))
if DEV:
    warnings.warn("This export is a development subset. Regenerate on the full test set for the paper.")


# =============================================================================
# Notebook cell 6
# =============================================================================

# Nature: Arial/Helvetica, 5-7 pt text, 8 pt bold lowercase panel letters, 180 mm double-column width.
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE - 1, "ytick.labelsize": FONT_SIZE - 1, "legend.fontsize": FONT_SIZE - 1,
    "axes.linewidth": 0.5, "axes.edgecolor": "#2b2b2b", "axes.labelcolor": "#0b0b0b", "axes.labelpad": 2.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.0, "ytick.major.size": 2.0,
    "xtick.minor.size": 1.2, "xtick.minor.width": 0.4, "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.color": "#2b2b2b", "ytick.color": "#2b2b2b",
    "legend.frameon": False, "legend.handletextpad": 0.3, "legend.borderaxespad": 0.2,
    "lines.solid_capstyle": "round", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 150,
})
INK, INK2, MUTED, RULE, FAINT = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#ecebe6"

# One colour per input-group family, used in every panel (validated: CVD and normal-vision separation pass
# for all pairs; aqua is below 3:1 contrast, so every coloured mark is also directly labelled).
FAMILY_COLOURS = {"Clinical history": "#2a78d6", "Lifestyle": "#eb6834", "Biomarker": "#1baf7a", "Demographic & social": "#52514e"}
GROUP_FAMILY = {"Diagnosis history": "Clinical history", "Landmark age": "Demographic & social", "BMI history": "Lifestyle", "Smoking history": "Lifestyle",
                "Alcohol history": "Lifestyle", "CRP history": "Biomarker", "Ethnicity": "Demographic & social",
                "Townsend deprivation": "Demographic & social"}
group_colour = lambda g: FAMILY_COLOURS[GROUP_FAMILY.get(g, "Demographic & social")]
BIOMARKER_SHADES = ["#085041", "#1baf7a", "#5dcaa5"]          # CRP curves: one hue, three steps
DIVERGING = LinearSegmentedColormap.from_list(
    "ratio", ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f1a3a2", "#d03b3b", "#7a1f1f"])
DIVERGING.set_bad("white")


def despine(ax, left=True, bottom=True):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


RATIO_TIERS = [np.array([0.1, 0.25, 0.5, 1, 2, 4, 10]), np.array([0.25, 0.5, 0.67, 0.8, 1, 1.25, 1.5, 2, 4]),
               np.array([0.8, 0.9, 1, 1.1, 1.25]), np.array([0.9, 0.95, 1, 1.05, 1.1]),
               np.array([0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03]), np.array([0.99, 0.995, 1, 1.005, 1.01])]


def nice_ratio_ticks(low, high, max_ticks=5):
    for tier in RATIO_TIERS:
        ticks = tier[(tier >= low) & (tier <= high)]
        if len(ticks) >= 3:
            break
    if len(ticks) < 3:
        ticks = np.unique([float(f"{x:.3g}") for x in np.exp(np.linspace(np.log(low), np.log(high), 3))])
    while len(ticks) > max_ticks:
        keep = (np.arange(len(ticks)) - (int(np.flatnonzero(ticks == 1)[0]) if 1 in ticks else 0)) % 2 == 0
        ticks = ticks[keep]
    return ticks


def ratio_axis(ax, axis, low, high, max_ticks=5):
    ticks = nice_ratio_ticks(low, high, max_ticks)
    (ax.set_xticks if axis == "x" else ax.set_yticks)(ticks, [f"{t:g}" for t in ticks])
    (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(mpl.ticker.NullLocator())


def place_labels(ax, items, size=None, marker_radius=4.0, extend_right=0.0, right_only=False):
    """Direct labels that avoid each other and every labelled point: try right, left, above and below at
    growing distances; a hairline leader joins any label placed away from its point.
    items: (text, x, y, fixed offset in points or None)."""
    size = size or FONT_SIZE - 1
    to_points = 72.0 / ax.figure.dpi
    points = [(t, *(v * to_points for v in ax.transData.transform((x, y))), x, y, f)
              for t, x, y, f in items if np.isfinite(x) and np.isfinite(y)]
    obstacles = [(px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius) for _, px, py, *_ in points]
    frame = [v * to_points for v in ax.get_window_extent().extents]
    frame[2] += extend_right                                   # labels must stay inside the axes (plus any right margin)
    inside = lambda b: b[0] >= frame[0] and b[2] <= frame[2] and b[1] >= frame[1] and b[3] <= frame[3]
    overlaps = lambda a, b: a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]
    for text, px, py, x, y, fixed in sorted(points, key=lambda p: (-p[2], p[1])):
        width, height = len(text) * size * 0.52, size * 1.05
        candidates = [fixed] if fixed else [
            (dx, dy) for r in (1.0, 2.2, 3.4, 4.6)
            for dx, dy in ((marker_radius + 1.5, 0), (-(marker_radius + 1.5) - width, 0),
                           (-width / 2, marker_radius + height * r * 0.75), (-width / 2, -(marker_radius + height * r * 0.75)),
                           (marker_radius + 1.5 + (r - 1) * 6, height * r), (marker_radius + 1.5 + (r - 1) * 6, -height * r),
                           (-(marker_radius + 1.5) - width - (r - 1) * 6, height * r),
                           (-(marker_radius + 1.5) - width - (r - 1) * 6, -height * r))]
        if right_only and not fixed:
            candidates = [(marker_radius + 2, k * height * sign) for k in range(8) for sign in ((1, -1) if k else (1,))]
        for dx, dy in candidates:
            box = (px + dx, py + dy - height / 2, px + dx + width, py + dy + height / 2)
            if fixed or (inside(box) and not any(overlaps(box, b) for b in obstacles)):
                break
        obstacles.append(box)
        far = abs(dy) > height * 0.9 or abs(dx) > marker_radius + width + 3
        ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=size, color=INK,
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.4, shrinkA=0, shrinkB=marker_radius,
                                    relpos=(0 if dx > 0 else 1, 0.5)) if far else None)


SOURCE, PENDING_LABELS = {}, []


# =============================================================================
# Notebook cell 7
# =============================================================================

def save(fig, stem):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(FIGURE_DIR / f"{stem}.{ext}", dpi=450 if ext == "png" else None)


def stamp(fig):
    if DEV:
        fig.text(0.5, 0.998, "DEVELOPMENT SUBSET — not for publication", ha="center", va="top",
                 color="#d03b3b", fontsize=FONT_SIZE, fontweight="bold")


def letters(fig, layout):
    """Lowercase bold letters: one x per column (outermost edge of that column's panels), one baseline per row.
    layout: list of rows; each row a list of (letter, [axes]) from left to right."""
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    box = lambda axes: [inv.transform(a.get_tightbbox(renderer)) for a in axes]
    n_cols = max(len(r) for r in layout)
    column_x = [min(b[0][0] for row in layout if len(row) > c for b in box(row[c][1])) for c in range(n_cols)]
    for row in layout:
        top = max(b[1][1] for _, axes in row for b in box(axes))
        for c, (letter, _) in enumerate(row):
            fig.text(column_x[c], top + 0.003, letter, fontsize=FONT_SIZE + 1, fontweight="bold", ha="left", va="bottom")




def row_letters(fig, layout):
    """Bold lowercase letters at the left edge of each panel, on one baseline per row."""
    renderer = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    box = lambda axes: [inv.transform(a.get_tightbbox(renderer)) for a in axes]
    for row in layout:
        top = max(b[1][1] for _, axes in row for b in box(axes))
        for letter, axes in row:
            x = max(0.004, min(b[0][0] for b in box(axes)))
            fig.text(x, top + 0.003, letter, fontsize=FONT_SIZE + 1, fontweight="bold", ha="left", va="bottom")


def chapter_runs(order):
    """Consecutive runs of the same ICD chapter: (start, end, roman numeral)."""
    runs, start = [], 0
    for i in range(1, len(order) + 1):
        if i == len(order) or CHAPTER[order[i]] != CHAPTER[order[start]]:
            runs.append((start, i - 1, CHAPTER[order[start]] or "†"))
            start = i
    return runs


def ratio_limit(values, floor=1.1, cap=4.0):
    logs = np.abs(np.log(values[np.isfinite(values) & (values > 0)]))
    return float(np.clip(np.nanpercentile(logs, 98) if logs.size else np.log(floor), np.log(floor), np.log(cap)))


def soften(rgba, keep):
    """Blend colours toward white where keep is False (interval includes no effect)."""
    out = rgba.copy()
    out[~keep, :3] = out[~keep, :3] * 0.38 + 0.62
    return out


def ratio_colourbar(fig, cax, limit, label, orientation="horizontal"):
    norm = TwoSlopeNorm(0, -limit, limit)
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, orientation=orientation)
    ticks = nice_ratio_ticks(np.exp(-limit), np.exp(limit), 5)
    bar.set_ticks(np.log(ticks), labels=[f"{t:g}" for t in ticks]); bar.outline.set_visible(False)
    bar.ax.tick_params(length=1.5, labelsize=FONT_SIZE - 1.5)
    bar.set_label(label, fontsize=FONT_SIZE - 1, labelpad=1)
    return bar


# =============================================================================
# Notebook cell 9
# =============================================================================

METRIC_LABEL = {"risk_ratio": "Risk ratio", "model_hazard_ratio": "Model hazard ratio",
                "mean_diff_pp": "Risk difference (pp)"}


def from_to_matrix(h, sex, metric):
    d = A.query("horizon_months == @h and sex == @sex")
    piv = lambda col: d.pivot(index="source_id", columns="target_id", values=col).reindex(index=SOURCE_ORDER, columns=TARGET_ORDER)
    low_col, high_col = {"risk_ratio": ("ratio_ci_low", "ratio_ci_high"), "model_hazard_ratio": ("hr_ci_low", "hr_ci_high"),
                         "mean_diff_pp": ("diff_ci_low", "diff_ci_high")}[metric]
    value, low, high, show = piv(metric), piv(low_col), piv(high_col), piv("displayable").fillna(False).astype(bool)
    null = 0.0 if metric == "mean_diff_pp" else 1.0
    stable = ((low > null) | (high < null)).to_numpy()
    return d, value.to_numpy(float), stable, show.to_numpy()


def heatmap(fig, ax, h, sex=A_SEX, metric=A_METRIC, limit=None, row_labels=True, annotate=True):
    d, value, stable, show = from_to_matrix(h, sex, metric)
    is_ratio = metric != "mean_diff_pp"
    shown = np.where(show, value, np.nan)
    if limit is None:
        limit = ratio_limit(shown) if is_ratio else float(np.nanpercentile(np.abs(shown), 98))
    norm = TwoSlopeNorm(0, -limit, limit) if is_ratio else TwoSlopeNorm(0, -limit, limit)
    scaled = np.log(shown) if is_ratio else shown
    rgba = DIVERGING(norm(np.clip(np.nan_to_num(scaled), -limit, limit)))
    rgba = soften(rgba, stable)
    rgba[~show] = mpl.colors.to_rgba(FAINT)
    ax.imshow(rgba, aspect="equal", interpolation="nearest")
    n_rows, n_cols = value.shape
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True); ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6); ax.tick_params(which="both", length=0)
    for start, end, _ in chapter_runs(TARGET_ORDER)[1:]:
        ax.axvline(start - 0.5, color="white", linewidth=1.8)
    for start, end, _ in chapter_runs(SOURCE_ORDER)[1:]:
        ax.axhline(start - 0.5, color="white", linewidth=1.8)
    ax.set_xticks(range(n_cols), [ABBR[k] for k in TARGET_ORDER], rotation=90)
    ax.xaxis.tick_top(); ax.xaxis.set_label_position("top")
    ax.set_yticks(range(n_rows), [ABBR[k] for k in SOURCE_ORDER] if row_labels else [])
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Chapter strips (right of the rows, under the columns): roman numerals, alternating greys.
    for runs, along_x in ((chapter_runs(TARGET_ORDER), True), (chapter_runs(SOURCE_ORDER), False)):
        for i, (start, end, roman) in enumerate(runs):
            shade = "#d9d8d2" if i % 2 == 0 else "#b4b3ab"
            if along_x:
                ax.add_patch(mpl.patches.Rectangle((start - 0.45, n_rows - 0.5 + 0.25), end - start + 0.9, 0.62,
                                                   color=shade, clip_on=False, linewidth=0))
                ax.text((start + end) / 2, n_rows - 0.5 + 0.56, roman, ha="center", va="center",
                        fontsize=FONT_SIZE - 2, color=INK, clip_on=False)
            else:
                ax.add_patch(mpl.patches.Rectangle((n_cols - 0.5 + 0.25, start - 0.45), 0.62, end - start + 0.9,
                                                   color=shade, clip_on=False, linewidth=0))
                ax.text(n_cols - 0.5 + 0.56, (start + end) / 2, roman, ha="center", va="center", rotation=90,
                        fontsize=FONT_SIZE - 2, color=INK, clip_on=False)
    if annotate and A_ANNOTATE_ABOVE and is_ratio:
        for i in range(n_rows):
            for j in range(n_cols):
                v = shown[i, j]
                if np.isfinite(v) and stable[i, j] and (v >= A_ANNOTATE_ABOVE or v <= 1 / A_ANNOTATE_ABOVE):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=FONT_SIZE - 2.5,
                            color="white" if abs(np.log(v)) > 0.6 * limit else INK)
    ax.set_xlim(-0.5, n_cols - 0.5); ax.set_ylim(n_rows - 0.5, -0.5)
    return d, limit

CHAPTER_FULL = dict(zip(CAT.condition_id, CAT.icd_chapter))


def chapter_colours(chapters):
    ordered = list(dict.fromkeys(chapters))
    disease = [c for c in ordered if c != "Death"]
    colours = {c: ICD_CHAPTER_COLOURS[i % len(ICD_CHAPTER_COLOURS)] for i, c in enumerate(disease)}
    colours["Death"] = "#000000"
    return {c: colours[c] for c in ordered}


def full_matrix(h, sex, metric):
    d = AFULL.query("horizon_months == @h and sex == @sex")
    piv = lambda col: d.pivot(index="source_id", columns="target_id", values=col).reindex(index=FULL_SOURCES, columns=FULL_TARGETS)
    low_col, high_col = ("diff_ci_low", "diff_ci_high") if metric == "mean_diff_pp" else ("ratio_ci_low", "ratio_ci_high")
    null = 0.0 if metric == "mean_diff_pp" else 1.0
    show = piv("displayable").fillna(False).astype(bool).to_numpy()
    value = np.where(show, piv(metric).to_numpy(float), np.nan)
    stable = ((piv(low_col) > null) | (piv(high_col) < null)).to_numpy() & show
    return d, value, stable


def chapter_strip(ax, ids, colours, vertical):
    codes = {c: i for i, c in enumerate(colours)}
    values = np.array([codes[CHAPTER_FULL[k]] for k in ids])
    cmap = mpl.colors.ListedColormap(list(colours.values()))
    ax.imshow(values[:, None] if vertical else values[None, :], aspect="auto", cmap=cmap, vmin=-0.5,
              vmax=len(colours) - 0.5, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def chapter_label(k, with_name):
    if k == "death" or not CHAPTER.get(k):
        return "Death"
    name = CHAPTER_NAME.get(k, "")
    return f"Ch {CHAPTER[k]}: {name}" if with_name and name else f"Ch {CHAPTER[k]}"


def chapter_ticks(ax, ids, axis, with_name=False):
    runs, start = [], 0
    for i in range(1, len(ids) + 1):
        if i == len(ids) or CHAPTER_FULL[ids[i]] != CHAPTER_FULL[ids[start]]:
            runs.append((start, i - 1, CHAPTER_FULL[ids[start]])); start = i
    mids = [(a + b) / 2 for a, b, _ in runs]
    labels = [chapter_label(ids[a], with_name) for a, b, _ in runs]
    if axis == "x":
        ax.set_xticks(mids, labels, fontsize=FONT_SIZE - 2.5, rotation=90)
    else:
        ax.set_yticks(mids, labels, fontsize=FONT_SIZE - 2.5)
    return [a - 0.5 for a, _, _ in runs[1:]]


def panel_a_full(fig, spec, h):
    """Original from -> to layout: one heatmap per sex, rows and columns ordered and coloured by ICD chapter."""
    metric = A_FULL_METRIC
    sexes = [s for s in A_FULL_SEXES if s in set(AFULL.sex)]
    box = spec.get_position(fig)                       # own grid: room on the left for the chapter names
    grid = fig.add_gridspec(2, 2 * len(sexes) + 2, height_ratios=[0.035, 1],
                            width_ratios=[0.022] + [1, 0.05] * (len(sexes) - 1) + [1, 0.022, 0.07], hspace=0.05, wspace=0.03,
                            left=0.215, right=0.985, top=box.y1, bottom=box.y0)
    colours = chapter_colours([CHAPTER_FULL[k] for k in FULL_TARGETS])
    matrices = [full_matrix(h, s, metric) for s in sexes]
    values = np.concatenate([m[1][np.isfinite(m[1])] for m in matrices])
    if metric == "mean_diff_pp":
        limit = float(np.quantile(np.abs(values), A_FULL_COLOUR_QUANTILE)) if values.size else 1.0
        scale = lambda v: v
    else:
        limit = float(np.quantile(np.abs(np.log(values)), A_FULL_COLOUR_QUANTILE)) if values.size else np.log(1.1)
        scale = np.log
    norm = TwoSlopeNorm(0, -limit, limit)
    row_strip = fig.add_subplot(grid[1, 0])
    chapter_strip(row_strip, FULL_SOURCES, colours, vertical=True)
    boundaries_y = chapter_ticks(row_strip, FULL_SOURCES, "y", with_name=True)
    row_strip.tick_params(axis="y", length=0, pad=1)
    row_strip.set_ylabel("Prior diagnosis removed (ICD-10 chapter)", labelpad=2)
    heat_axes = []
    for i, (sex, (d, value, stable)) in enumerate(zip(sexes, matrices)):
        column = 1 + 2 * i
        strip = fig.add_subplot(grid[0, column]); ax = fig.add_subplot(grid[1, column])
        chapter_strip(strip, FULL_TARGETS, colours, vertical=False)
        strip.set_title(SEX_LABELS.get(sex, sex), loc="left", fontweight="bold", fontsize=FONT_SIZE, pad=2)
        cmap = DIVERGING.copy(); cmap.set_bad("white")          # white = not shown (too few people)
        ax.imshow(np.ma.masked_invalid(scale(value)), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        for b in boundaries_y:
            ax.axhline(b, color="white", linewidth=0.8)
        for b in chapter_ticks(ax, FULL_TARGETS, "x"):
            ax.axvline(b, color="white", linewidth=0.8)
        ax.set_yticks([]); ax.tick_params(length=0, pad=1)
        if A_FULL_SHOW_CI_DOTS:
            yy, xx = np.where(stable)
            ax.scatter(xx, yy, s=0.6, color=INK, linewidths=0)
        ax.set_xlabel("Outcome (ICD-10 chapter)", labelpad=1.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        heat_axes += [strip, ax]
    cax = fig.add_subplot(grid[1, -2])
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=DIVERGING), cax=cax, extend="both")
    bar.outline.set_visible(False); bar.ax.tick_params(length=1.5, labelsize=FONT_SIZE - 1.5)
    if metric == "mean_diff_pp":
        bar.set_label("Δ risk, with − without (pp)", fontsize=FONT_SIZE - 1, labelpad=1)
    else:
        ticks = nice_ratio_ticks(np.exp(-limit), np.exp(limit), 5)
        bar.set_ticks(np.log(ticks), labels=[f"{t:g}" for t in ticks]); bar.set_label("Risk ratio", fontsize=FONT_SIZE - 1)
    SOURCE["a"] = pd.concat([m[0] for m in matrices])[["sex", "source_name", "source_icd_chapter", "target_name",
                                                      "target_icd_chapter", "n_with_source_eligible", "mean_diff_pp",
                                                      "diff_ci_low", "diff_ci_high", "risk_ratio", "ratio_ci_low",
                                                      "ratio_ci_high", "displayable"]]
    return [row_strip] + heat_axes + [cax]


def panel_a(fig, ax, cax, h):
    d, limit = heatmap(fig, ax, h)
    ax.set_ylabel("Prior diagnosis", labelpad=1.5)
    ax.set_xlabel("Outcome", labelpad=2)
    if A_METRIC == "mean_diff_pp":
        bar = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(0, -limit, limit), cmap=DIVERGING), cax=cax,
                           orientation="horizontal")
        bar.outline.set_visible(False); bar.ax.tick_params(length=1.5, labelsize=FONT_SIZE - 1.5)
        bar.set_label("Risk difference, recorded − removed (pp)", fontsize=FONT_SIZE - 1, labelpad=1)
    else:
        ratio_colourbar(fig, cax, limit, f"{METRIC_LABEL[A_METRIC]}, prior diagnosis recorded vs removed")
    SOURCE["a_selected"] = d[~d.same_condition][["source_name", "source_abbreviation", "source_icd_chapter", "target_name",
                                        "target_abbreviation", "target_icd_chapter", "n_with_source_eligible", "mean_diff_pp",
                                        "diff_ci_low", "diff_ci_high", "risk_ratio", "ratio_ci_low", "ratio_ci_high",
                                        "model_hazard_ratio", "hr_ci_low", "hr_ci_high", "displayable"]]


def model_vs_cox(h):
    model = A.query("horizon_months == @h and sex == 'all' and displayable")
    if COX is None or not len(COX):
        return None
    return model.merge(COX[["source_id", "target_id", "hazard_ratio", "ci_low", "ci_high", "convergence_warning"]],
                       on=["source_id", "target_id"])


SLOPES = {}


def panel_b(ax, h):
    """Does the model reproduce the associations in the observed outcomes? Each point: one prior diagnosis ->
    outcome among the shown conditions; age- and sex-standardised risk ratio, with vs without the prior diagnosis."""
    if CAL is None or not len(CAL):
        ax.text(0.5, 0.5, "Observed vs predicted not available", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        ax.axis("off"); return
    d = CAL.query("horizon_months == @h and exposed_events >= @B_MIN_EVENTS").dropna(subset=["observed_ratio", "predicted_ratio"])
    d = d[(d.observed_ratio > 0) & (d.predicted_ratio > 0)]
    if len(d) < 3:
        ax.text(0.5, 0.5, f"Too few pairs with >= {B_MIN_EVENTS} events\n(lower B_MIN_EVENTS)", ha="center", va="center",
                transform=ax.transAxes, color=MUTED, fontsize=FONT_SIZE - 1)
        ax.axis("off"); return
    x, y = d.observed_ratio.to_numpy(), d.predicted_ratio.to_numpy()
    lo = min(x.min(), y.min(), 1) / 1.1; hi = max(x.max(), y.max(), 1) * 1.1
    ax.plot([lo, hi], [lo, hi], color=RULE, linewidth=0.6, zorder=0)
    ax.axhline(1, color=FAINT, linewidth=0.5, zorder=0); ax.axvline(1, color=FAINT, linewidth=0.5, zorder=0)
    ax.hlines(y, np.clip(d.observed_ci_low, lo, hi), np.clip(d.observed_ci_high, lo, hi), color="#2a78d6", alpha=0.18, linewidth=0.6, zorder=1)
    size = (4 + 14 * np.sqrt(d.exposed_events / d.exposed_events.max())).to_numpy()
    ax.scatter(x, y, s=size, color="#6aa3e6", alpha=0.85, edgecolors="white", linewidths=0.3, zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ratio_axis(ax, "x", lo, hi); ratio_axis(ax, "y", lo, hi)
    rho = pd.Series(np.log(y)).corr(pd.Series(np.log(x)), method="spearman")
    weights = d.exposed_events.to_numpy(float)
    wx, wy = np.log(x), np.log(y)
    mx, my = np.average(wx, weights=weights), np.average(wy, weights=weights)
    r_w = np.sum(weights * (wx - mx) * (wy - my)) / np.sqrt(np.sum(weights * (wx - mx) ** 2) * np.sum(weights * (wy - my) ** 2))
    slope = np.sum(weights * (wx - mx) * (wy - my)) / np.sum(weights * (wx - mx) ** 2)      # log predicted on log observed
    ax.text(0.97, 0.03, f"Spearman ρ = {rho:.2f}\nevent-weighted r = {r_w:.2f}\nslope = {slope:.2f}\n{len(d)} pairs",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_SIZE - 1.5, color=INK2)
    SLOPES[h] = slope
    ax.set_xlabel(f"Observed risk ratio ({h // 12} y, test set)")
    ax.set_ylabel("Predicted risk ratio (CRESCENT)")
    # Highlight the strongest predicted pairs: darker points, labels set beside them (no leaders),
    # each placed at the first nearby position that overlaps neither another label nor the axes edge.
    import matplotlib.patheffects as pe
    top = d.nlargest(B_LABELS, "predicted_ratio").sort_values("predicted_ratio", ascending=False)
    ax.scatter(top.observed_ratio, top.predicted_ratio, s=size[[d.index.get_loc(i) for i in top.index]] + 4,
               color="#184f95", edgecolors="white", linewidths=0.4, zorder=4)
    renderer = ax.figure.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    # Left of the point first: above the identity line the plot is empty, so labels stay unambiguous.
    candidates = [(-5, 0, "right", "center"), (-5, 4, "right", "bottom"), (-5, -4, "right", "top"),
                  (5, 0, "left", "center"), (0, 5, "center", "bottom"), (0, -5, "center", "top"),
                  (-9, 9, "right", "bottom"), (9, -9, "left", "top")]
    from matplotlib.transforms import Bbox
    pts = ax.transData.transform(np.column_stack([top.observed_ratio, top.predicted_ratio]))
    pad = 4 * ax.figure.dpi / 72
    placed = [Bbox.from_extents(px - pad, py - pad, px + pad, py + pad) for px, py in pts]   # keep labels off the dark points
    for r in top.itertuples():
        label = f"{r.source_abbreviation}→{r.target_abbreviation}"
        for dx, dy, ha, va in candidates:
            text = ax.annotate(label, (r.observed_ratio, r.predicted_ratio), xytext=(dx, dy), textcoords="offset points",
                               ha=ha, va=va, fontsize=FONT_SIZE - 1.5, color="#0b0b0b", zorder=5,
                               path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
            box = text.get_window_extent(renderer).expanded(1.04, 1.1)
            inside = (box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1 and box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1)
            if inside and not any(box.overlaps(other) for other in placed):
                placed.append(box)
                break
            text.remove()
        else:
            placed.append(ax.annotate(label, (r.observed_ratio, r.predicted_ratio), xytext=(5, 0), textcoords="offset points",
                                      ha="left", va="center", fontsize=FONT_SIZE - 1.5, color="#0b0b0b", zorder=5,
                                      path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]).get_window_extent(renderer))
    despine(ax)
    SOURCE["b"] = d


HIGHLIGHT_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#7a4fc4"]


def panel_c(ax, h):
    d = B.query("horizon_months == @h and displayable")
    span = d.groupby("condition_id").ale_pp.agg(lambda v: v.max() - v.min()).sort_values(ascending=False)
    highlight = list(span.index[:C_HIGHLIGHT])
    ax.axhline(0, color=RULE, linewidth=0.6, zorder=0)
    ax.axvline(REF_AGE, color=RULE, linewidth=0.6, linestyle=(0, (2, 2)), zorder=0)
    for k in TARGET_ORDER:
        rows = d[d.condition_id.eq(k)].sort_values("age_midpoint_years")
        if k in highlight:
            continue
        ax.plot(rows.age_midpoint_years, rows.ale_pp, color="#c3c2b7", linewidth=0.7, zorder=1)
    items = []
    for k, colour in zip(highlight, HIGHLIGHT_COLOURS):
        rows = d[d.condition_id.eq(k)].sort_values("age_midpoint_years")
        ax.fill_between(rows.age_midpoint_years, rows.ale_ci_low, rows.ale_ci_high, color=colour, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(rows.age_midpoint_years, rows.ale_pp, color=colour, linewidth=1.2, zorder=3)
        items.append((ABBR[k], rows.age_midpoint_years.iloc[-1], rows.ale_pp.iloc[-1], None))
    x_max = d.age_midpoint_years.max()
    ax.set_xlim(d.age_midpoint_years.min() - 1, x_max + 1)
    ax.set_xlabel("Landmark age (years)")
    ax.set_ylabel(f"Change in {h // 12}-year risk (pp)")
    ax.text(REF_AGE, ax.get_ylim()[0], f" {REF_AGE:.0f} y (reference)", fontsize=FONT_SIZE - 2, color=INK2, va="bottom", ha="left")
    ax.text(0.02, 0.97 if d.ale_pp.iloc[0] < 0 else 0.1, "Grey: other outcomes", transform=ax.transAxes,
            fontsize=FONT_SIZE - 2, color=MUTED, va="top")
    PENDING_LABELS.append((ax, items, dict(right_only=True, extend_right=24)))
    despine(ax)
    SOURCE["c"] = d[["condition_name", "condition_abbreviation", "age_interval", "age_midpoint_years", "ale_pp", "ale_ci_low",
                     "ale_ci_high", "n_in_bin", "reference_age_years"]]


def window_label(w):
    return {"0-1": "<1", ">=10": "≥10"}.get(w, w.replace("-", "–"))


def panel_d(ax, bax, h):
    d = C.query("horizon_months == @h")
    windows = list(dict.fromkeys(d.sort_values("window_low_years").window))
    x = np.arange(len(windows))
    share = d.groupby("window").participants_percent.first().reindex(windows)
    bax.bar(x, share, width=0.55, color="#b4b3ab", linewidth=0)
    for xi, v in zip(x, share):
        bax.text(xi, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=FONT_SIZE - 2, color=INK2)
    bax.set_ylim(0, share.max() * 1.35); bax.set_yticks([]); bax.set_xticks([])
    bax.set_title("Participants with a diagnosis in the window", fontsize=FONT_SIZE - 2, color=MUTED, loc="left", pad=1.5)
    despine(bax, left=False, bottom=False)
    conditions = d[d.condition_id.ne("death")]
    ax.axhline(1, color=RULE, linewidth=0.6, zorder=0)
    for i, w in enumerate(windows):
        rows = conditions[conditions.window.eq(w)]
        jitter = np.random.default_rng(i).uniform(-0.17, 0.17, len(rows))
        ax.scatter(i + jitter, rows.risk_ratio, s=6, color="#2a78d6", alpha=0.45, linewidths=0, zorder=2)
    gm = conditions.groupby("window").risk_ratio.apply(lambda v: float(np.exp(np.log(v).mean()))).reindex(windows)
    ax.plot(x, gm, color="#184f95", linewidth=1.1, zorder=3, marker="o", markersize=3.2, label="Conditions (geometric mean)")
    death = d[d.condition_id.eq("death")].set_index("window").reindex(windows)
    if len(death.dropna(how="all")):
        ax.errorbar(x + 0.24, death.risk_ratio, yerr=[death.risk_ratio - death.ratio_ci_low, death.ratio_ci_high - death.risk_ratio],
                    fmt="D", markersize=2.6, color=INK2, elinewidth=0.6, capsize=0, zorder=3, label="Death")
    ax.set_yscale("log")
    values = pd.concat([conditions.risk_ratio, death.risk_ratio]).dropna()
    lo, hi = min(values.min(), 1) / 1.03, values.max() * 1.05
    ax.set_ylim(lo, hi); ratio_axis(ax, "y", lo, hi)
    ax.set_xticks(x, [window_label(w) for w in windows]); ax.set_xlim(-0.5, len(windows) - 0.5); bax.set_xlim(-0.5, len(windows) - 0.5)
    ax.set_xlabel("Years before landmark")
    ax.set_ylabel(f"Risk ratio ({h // 12} y)")
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1.5, handletextpad=0.2, borderaxespad=0.1)
    despine(ax)
    SOURCE["d"] = d[["window", "condition_name", "condition_abbreviation", "n_with_diagnosis_in_window", "participants_percent",
                     "mean_diff_pp", "ci_low", "ci_high", "risk_ratio", "ratio_ci_low", "ratio_ci_high"]]


DOMAIN_ROWS = ["Diagnosis", "BMI", "Smoking", "Alcohol", "CRP"]
DOMAIN_COLOUR = {"Diagnosis": FAMILY_COLOURS["Clinical history"], "BMI": FAMILY_COLOURS["Lifestyle"],
                 "Smoking": FAMILY_COLOURS["Lifestyle"], "Alcohol": FAMILY_COLOURS["Lifestyle"], "CRP": FAMILY_COLOURS["Biomarker"]}


def tidy_token(text, width=22):
    text = str(text)
    for prefix in ("DX_", "BMI_", "SMOKING_", "SMOKE_", "ALCOHOL_", "ALC_"):
        if text.upper().startswith(prefix):
            text = text[len(prefix):]
    text = re.sub(r"(\d)_(\d)", r"\1–\2", text).replace("_", " ").replace(" PER WEEK", "/WEEK").replace(" per week", "/week").replace(" /", "/")
    text = text[0].upper() + text[1:].lower() if text.isupper() else text
    return text if len(text) <= width else text[:width - 1] + "…"


def patient_panel(fig, tax, rax, sax, row):
    """Patient profile on a real-age axis: diagnoses as labelled stems, lifestyle records and CRP values,
    with the landmark, entry delay and prediction window marked; beside it, predicted risk across horizons
    and exact 8-group Shapley contributions. Omits observed outcome lane for synthetic profiles."""
    profile = PROFILES[PROFILES.panel_row.eq(row)].iloc[0]
    events = TIMELINES[TIMELINES.panel_row.eq(row)].copy()
    risks = RISKS[RISKS.panel_row.eq(row)].sort_values("horizon_years_after_lead")
    is_synthetic = str(profile.status).lower() == "synthetic"

    landmark = float(profile.age_years)
    entry_age = float(profile.lead_end_age_years)
    window_years = manifest["patients"]["horizon_months"] / 12.0
    events["age"] = events.event_age_years
    events = events[events.lookback_years <= PATIENT_LOOKBACK_YEARS]

    x_start = np.floor(min(events.age.min() if len(events) else landmark - 5, landmark - 3)) - 0.5
    x_end = entry_age + window_years + 1.0                       # longer horizons are in the risk plot beside it
    if not is_synthetic and profile.status == "event" and np.isfinite(profile.target_event_age_years):
        x_end = max(x_end, float(profile.target_event_age_years) + 1.0)

    # Lanes, top to bottom: omit observed outcome lane for synthetic profiles.
    lanes = ([] if is_synthetic else [("outcome", f"{PATIENT_TARGET}", 0.6)]) + [("dx", "Diagnoses", 1.6)]
    lanes += [(d, d if d != "CRP" else "CRP (mg/L)", 0.55)
              for d in ("BMI", "Smoking", "Alcohol", "CRP") if events.domain.eq(d).any()]

    top, centres = 0.0, {}
    for key, label, height in lanes:
        centres[key] = (top, top + height)
        top += height
    lane_mid = {k: (a + b) / 2.0 for k, (a, b) in centres.items()}

    # Context: entry delay, prediction window, landmark.
    tax.axvspan(landmark, entry_age, color="#ecebe6", linewidth=0, zorder=0)
    tax.axvspan(entry_age, entry_age + window_years, color="#eaf2fc", linewidth=0, zorder=0)
    tax.axvline(landmark, color=INK, linewidth=0.7, zorder=2)
    tax.annotate(f"Landmark {landmark:.1f} y", (landmark, 0), xytext=(-2, 2), textcoords="offset points",
                 ha="right", va="bottom", fontsize=FONT_SIZE - 2, color=INK, annotation_clip=False)
    tax.annotate(f"{window_years:.0f}-year window", (entry_age + window_years / 2.0, 0), xytext=(0, 2),
                 textcoords="offset points", ha="center", va="bottom", fontsize=FONT_SIZE - 2,
                 color="#184f95", annotation_clip=False)
    for key, (a, b) in list(centres.items())[1:]:
        tax.axhline(a, color="#e4e3de", linewidth=0.4, zorder=0)

    events["short"] = [abbr if isinstance(abbr, str) and abbr else tidy_token(label, 14)
                       for abbr, label in zip(events.get("abbreviation", pd.Series([None] * len(events), index=events.index)),
                                              events.display_label)]

    # Outcome (real participants only).
    if not is_synthetic:
        y = lane_mid["outcome"]
        if profile.status == "event" and np.isfinite(profile.target_event_age_years):
            tax.scatter([profile.target_event_age_years], [y], marker="*", s=42, color="#d03b3b", edgecolors="white",
                        linewidths=0.3, zorder=4)
            tax.annotate(f"{PATIENT_TARGET} at {profile.target_event_age_years:.1f} y",
                         (profile.target_event_age_years, y), xytext=(5, 0), textcoords="offset points",
                         ha="left", va="center", fontsize=FONT_SIZE - 2, color="#d03b3b")
        else:
            tax.annotate("none in window", (entry_age + window_years / 2.0, y), ha="center", va="center",
                         fontsize=FONT_SIZE - 2.5, color=MUTED)
        if np.isfinite(profile.follow_up_end_age_years) and profile.follow_up_end_age_years < x_end:
            tax.plot([profile.follow_up_end_age_years] * 2, list(centres["outcome"]), color=MUTED, linewidth=0.6)
            tax.annotate("end of follow-up", (profile.follow_up_end_age_years, centres["outcome"][0]),
                         xytext=(-2, -1), textcoords="offset points", ha="right", va="top",
                         fontsize=FONT_SIZE - 3, color=MUTED)

    # Diagnoses: stems with labels at up to three heights, chosen so labels never overlap.
    a, b = centres["dx"]
    dx = events[events.domain.eq("Diagnosis")].sort_values("age")
    colour = DOMAIN_COLOUR["Diagnosis"]
    tax.scatter(dx.age, np.full(len(dx), b - 0.15), s=10, color=colour, edgecolors="white", linewidths=0.4, zorder=4)
    labelled = dx.drop_duplicates("short", keep="last").tail(PATIENT_MAX_DX_LABELS)
    fig.canvas.draw()
    units_per_pt = (x_end - x_start) / (tax.get_window_extent().width * 72 / fig.dpi)
    char_w = (FONT_SIZE - 2.3) * 0.68 * units_per_pt
    levels = [b - 0.55, b - 0.95, b - 1.35, b - 1.75][:3]
    right_edge = [-np.inf] * len(levels)
    gap = 1.5 * char_w
    for age_val, text in zip(labelled.age, labelled.short):
        width = len(text) * char_w
        level = int(np.argmin([max(0.0, edge + gap - age_val) for edge in right_edge]))
        x_text = max(age_val, right_edge[level] + gap)
        right_edge[level] = x_text + width
        y_text = levels[level]
        tax.plot([age_val, age_val, x_text], [b - 0.15, y_text + 0.12, y_text + 0.12],
                 color=colour, linewidth=0.45, alpha=0.55, zorder=3, solid_joinstyle="miter")
        tax.annotate(text, (x_text, y_text), xytext=(0.5, -0.5), textcoords="offset points", ha="left", va="center",
                     fontsize=FONT_SIZE - 2.3, color=colour, fontweight="bold", zorder=6,
                     bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none", alpha=0.92))
    hidden = dx.short.nunique() - len(labelled)
    if hidden > 0:
        tax.annotate(f"+{hidden} more", (x_start, a), xytext=(2, -2), textcoords="offset points", ha="left", va="top",
                     fontsize=FONT_SIZE - 3, color=MUTED)

    # Lifestyle records and CRP: value labels beside each record.
    for domain in ("BMI", "Smoking", "Alcohol", "CRP"):
        if domain not in centres:
            continue
        y = lane_mid[domain]
        rows = events[events.domain.eq(domain)].sort_values("age")
        tax.scatter(rows.age, np.full(len(rows), y), s=10, marker="s" if domain == "CRP" else "o",
                    color=DOMAIN_COLOUR[domain], edgecolors="white", linewidths=0.4, zorder=4)
        for age_val, label, value in zip(rows.age, rows.display_label, rows.raw_value):
            text = f"{value:.1f}" if domain == "CRP" and np.isfinite(value) else tidy_token(label, 14)
            tax.annotate(text, (age_val, y), xytext=(-3.5, 0), textcoords="offset points", ha="right", va="center",
                         fontsize=FONT_SIZE - 2.5, color=INK2)

    tax.set_yticks([lane_mid[k] for k, _, _ in lanes], [label for _, label, _ in lanes])
    for tick, (key, _, _) in zip(tax.get_yticklabels(), lanes):
        tick.set_color("#d03b3b" if key == "outcome" else ("#184f95" if key == "risk" else INK))
    tax.tick_params(axis="y", length=0, labelsize=FONT_SIZE - 1.5)
    tax.set_ylim(top, 0); tax.set_xlim(x_start, x_end)
    tax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(5)); tax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))
    tax.set_xlabel("Age (years)", labelpad=1)

    if "condition_name" in labelled and labelled.condition_name.notna().any():
        key = "; ".join(f"{a}, {str(n).lower() if not str(n).isupper() else n}" for a, n in
                        dict(zip(labelled.short, labelled.condition_name)).items() if isinstance(n, str))
        width_chars = max(20, int(tax.get_window_extent().width * 72 / fig.dpi / ((FONT_SIZE - 2.5) * 0.47)))
        tax.annotate("\n".join(textwrap.wrap(key, width_chars)), (0, 0), xycoords="axes fraction", xytext=(0, -20),
                     textcoords="offset points", ha="left", va="top", fontsize=FONT_SIZE - 2.5, color=MUTED)
    despine(tax, left=False)

    # Header: synthetic vs real participant.
    sex = SEX_LABELS.get(f"code {int(profile.recorded_sex_code)}", f"sex code {int(profile.recorded_sex_code)}")
    if is_synthetic:
        idx_num = str(row).lstrip("D")
        header = f"Synthetic patient {idx_num} ({sex}, {landmark:.1f} y)"
    else:
        outcome = (f"developed {PATIENT_TARGET} at {profile.target_event_age_years:.0f} y" if profile.status == "event"
                   else f"no {PATIENT_TARGET} within {window_years:.0f} y")
        header = f"{sex}, {landmark:.0f} y — {outcome}"
    tax.annotate(header, (0, 1), xycoords="axes fraction", xytext=(0, 11),
                 textcoords="offset points", ha="left", va="bottom", fontsize=FONT_SIZE - 0.5, color=INK, fontweight="bold")
    events["t"] = events.age - landmark

    # --- Attribution column: predicted risk by horizon, then Shapley contributions ---
    years = risks.horizon_years_after_lead.to_numpy(float)
    risk_pct = 100.0 * risks.predicted_target_risk.to_numpy(float)
    rax.plot(years, risk_pct, color="#184f95", linewidth=1.0, marker="o", markersize=3.2, markerfacecolor="white",
             markeredgewidth=0.8, zorder=3)
    rax.fill_between(years, 0, risk_pct, color="#2a78d6", alpha=0.1, linewidth=0)
    for yr, v, primary in zip(years, risk_pct, risks.is_primary_horizon):
        rax.annotate(f"{v:.1f}%", (yr, v), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                     fontsize=FONT_SIZE - 2.5, color="#184f95", fontweight="bold" if primary else "normal")
    rax.set_xticks(years, [f"{int(v)} y" for v in years]); rax.set_xlim(years.min() - 1, years.max() + 1)
    rax.set_ylim(0, max(risk_pct.max(), 0.1) * 1.45); rax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    rax.set_ylabel("Risk (%)", fontsize=FONT_SIZE - 1.5)
    rax.set_title(f"Predicted {PATIENT_TARGET} risk after entry", loc="left", fontsize=FONT_SIZE - 1, fontweight="bold", pad=2)
    despine(rax)

    s = SHAP[SHAP.panel_row.eq(row)].copy()
    s = s.reindex(s.shapley_pp.abs().sort_values(ascending=False).index)
    y = np.arange(len(s))
    sax.barh(y, s.shapley_pp, height=0.62, color=[group_colour(g) for g in s.group_name], linewidth=0, zorder=2)
    sax.axvline(0, color=INK2, linewidth=0.6, zorder=3)
    span = max(s.shapley_pp.abs().max(), 1e-6)
    for yi, v in zip(y, s.shapley_pp):
        sax.text(v + np.sign(v or 1) * span * 0.04, yi, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
                 fontsize=FONT_SIZE - 2.5, color=INK2)
    sax.set_yticks(y, [GROUP_LABELS.get(g, g) for g in s.group_name]); sax.set_ylim(len(s) - 0.5, -0.6)
    sax.tick_params(axis="y", length=0, labelsize=FONT_SIZE - 1.5)
    sax.set_xlim(min(s.shapley_pp.min(), 0) - span * 0.4, max(s.shapley_pp.max(), 0) + span * 0.4)
    sax.set_title(f"{window_years:.0f}-year risk: reference {100 * profile.reference_risk:.1f}% → {100 * profile.observed_risk:.1f}%",
                  loc="left", fontsize=FONT_SIZE - 1, fontweight="bold", pad=2)
    sax.set_xlabel("Contribution (pp)", fontsize=FONT_SIZE - 1.5)
    despine(sax, left=False)

    SOURCE[f"patient_{row}"] = s.assign(**{k: profile[k] for k in ("age_years", "recorded_sex_code", "status", "observed_risk", "reference_risk")})
    SOURCE[f"patient_{row}_timeline"] = events


# =============================================================================
# Notebook cell 11
# =============================================================================

def draw_fig5(h):
    SOURCE.clear(); PENDING_LABELS.clear()
    patients = PROFILES.panel_row.tolist()[:2] if PROFILES is not None else []   # repository: e, f need the restricted patient tables
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 9.7))
    outer = fig.add_gridspec(2 + len(patients), 1, height_ratios=[3.0, 1.85] + [1.75] * len(patients), hspace=0.42,
                             left=0.075, right=0.975, top=0.97, bottom=0.04)
    axes_a = panel_a_full(fig, outer[0], h)
    middle = outer[1].subgridspec(2, 3, width_ratios=[1, 1.05, 1.05], height_ratios=[0.22, 1], wspace=0.45, hspace=0.06)
    ax_b = fig.add_subplot(middle[:, 0]); ax_c = fig.add_subplot(middle[:, 1])
    bax_d, ax_d = fig.add_subplot(middle[0, 2]), fig.add_subplot(middle[1, 2])
    patient_axes = []
    for i, row in enumerate(patients):
        grid = outer[2 + i].subgridspec(2, 3, width_ratios=[1, 1, 0.95], height_ratios=[0.6, 1], wspace=0.5, hspace=0.75)
        patient_axes.append((fig.add_subplot(grid[:, :2]), fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 2])))
    panel_b(ax_b, h)
    panel_c(ax_c, h)
    panel_d(ax_d, bax_d, h)
    fig.canvas.draw()
    for row, (tax, rax, sax) in zip(patients, patient_axes):
        patient_panel(fig, tax, rax, sax, row)
    fig.canvas.draw()
    for entry in PENDING_LABELS:
        ax, items, *options = entry
        place_labels(ax, items, **(options[0] if options else {}))
    fig.canvas.draw()
    layout = [[("a", axes_a)], [("b", [ax_b]), ("c", [ax_c]), ("d", [bax_d, ax_d])]]
    layout += [[(letter, [tax])] for letter, (tax, rax, sax) in zip("ef", patient_axes)]
    row_letters(fig, layout)
    stamp(fig)
    save(fig, f"Fig5_{h}m")
    source_dir = FIGURE_DIR / "source_data"; source_dir.mkdir(parents=True, exist_ok=True)
    for key, frame in SOURCE.items():
        frame.to_csv(source_dir / f"Fig5{key}_{h}m.csv", index=False)
    return fig


for h in HORIZONS:
    figure = draw_fig5(h)
    if h == HORIZONS[0]:
        plt.show()
    plt.close(figure)
print("Figures written to", FIGURE_DIR)


# =============================================================================
# Notebook cell 13
# =============================================================================

def ext_selected(h):
    fig = plt.figure(figsize=(4.4, 4.5))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.035], hspace=0.2, left=0.12, right=0.95, top=0.86, bottom=0.08)
    ax, cax = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    panel_a(fig, ax, cax, h)
    ax.apply_aspect(); pa = ax.get_position(); pc = cax.get_position()
    cax.set_position([pa.x0 + pa.width * 0.15, pa.y0 - 0.075, pa.width * 0.7, pc.height])
    stamp(fig); save(fig, f"ExtFig5_from_to_selected_{h}m"); plt.close(fig)
    SOURCE["a_selected"].to_csv(FIGURE_DIR / "source_data" / f"ExtFig5_from_to_selected_{h}m.csv", index=False)


def ext_by_sex(h):
    sexes = [s for s in ("code 0", "code 1") if s in set(A.sex)]
    if not sexes:
        return
    limit = max(ratio_limit(from_to_matrix(h, s, "risk_ratio")[1]) for s in sexes)
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.9))
    grid = fig.add_gridspec(2, len(sexes), height_ratios=[1, 0.04], wspace=0.12, hspace=0.25, left=0.06, right=0.97, top=0.86, bottom=0.08)
    for i, s in enumerate(sexes):
        ax = fig.add_subplot(grid[0, i])
        heatmap(fig, ax, h, sex=s, metric="risk_ratio", limit=limit, row_labels=i == 0)
        ax.set_title(SEX_LABELS.get(s, s), loc="left", fontweight="bold", pad=24)
    ratio_colourbar(fig, fig.add_subplot(grid[1, :]), limit, "Risk ratio, prior diagnosis recorded vs removed")
    pos = fig.axes[-1].get_position(); fig.axes[-1].set_position([pos.x0 + pos.width * 0.3, pos.y0, pos.width * 0.4, pos.height])
    stamp(fig); save(fig, f"ExtFig5_from_to_by_sex_{h}m"); plt.close(fig)


def ext_pp(h):
    fig = plt.figure(figsize=(4.4, 4.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.04], hspace=0.22, left=0.1, right=0.95, top=0.86, bottom=0.08)
    ax = fig.add_subplot(grid[0])
    _, limit = heatmap(fig, ax, h, metric="mean_diff_pp")
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=TwoSlopeNorm(0, -limit, limit), cmap=DIVERGING), cax=fig.add_subplot(grid[1]),
                       orientation="horizontal")
    bar.outline.set_visible(False); bar.set_label("Risk difference, recorded − removed (pp)", fontsize=FONT_SIZE - 1)
    stamp(fig); save(fig, f"ExtFig5_from_to_pp_{h}m"); plt.close(fig)


def ext_forest(h):
    data = model_vs_cox(h)
    d = A.query("horizon_months == @h and sex == 'all' and displayable").nlargest(FOREST_PAIRS, "risk_ratio")
    if data is not None:
        d = d.merge(data[["source_id", "target_id", "hazard_ratio", "ci_low", "ci_high"]], on=["source_id", "target_id"], how="left")
    fig, ax = plt.subplots(figsize=(3.6, 0.17 * len(d) + 0.9))
    y = np.arange(len(d))
    series = [("risk_ratio", "ratio_ci_low", "ratio_ci_high", "CRESCENT risk ratio", "#2a78d6", "o", -0.22),
              ("model_hazard_ratio", "hr_ci_low", "hr_ci_high", "CRESCENT hazard ratio", "#184f95", "s", 0.0)]
    if "hazard_ratio" in d:
        series.append(("hazard_ratio", "ci_low", "ci_high", "Cox hazard ratio", INK2, "D", 0.22))
    for col, lo, hi, label, colour, marker, off in series:
        ax.hlines(y + off, d[lo], d[hi], color=colour, linewidth=0.7)
        ax.scatter(d[col], y + off, s=10, marker=marker, color=colour, edgecolors="white", linewidths=0.3, label=label, zorder=3)
    ax.axvline(1, color=RULE, linewidth=0.6)
    ax.set_xscale("log"); low, high = ax.get_xlim(); ratio_axis(ax, "x", low, high)
    ax.set_yticks(y, [f"{a} → {b}" for a, b in zip(d.source_abbreviation, d.target_abbreviation)])
    ax.set_ylim(len(d) - 0.5, -0.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Ratio ({h // 12}-year horizon)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1 - 1.5 / (len(d) + 4)), ncol=3, fontsize=FONT_SIZE - 1.5)
    despine(ax, left=False); fig.tight_layout(); stamp(fig); save(fig, f"ExtFig5_forest_{h}m"); plt.close(fig)
    d.to_csv(FIGURE_DIR / "source_data" / f"ExtFig5_forest_{h}m.csv", index=False)


def ext_age_grid(h):
    d = B.query("horizon_months == @h and displayable")
    n = len(TARGET_ORDER); cols = 4; rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(FIGURE_WIDTH_IN, 1.25 * rows + 0.3), sharex=True)
    for ax, k in zip(axes.ravel(), TARGET_ORDER):
        r = d[d.condition_id.eq(k)].sort_values("age_midpoint_years")
        ax.fill_between(r.age_midpoint_years, r.ale_ci_low, r.ale_ci_high, color="#2a78d6", alpha=0.18, linewidth=0)
        ax.plot(r.age_midpoint_years, r.ale_pp, color="#2a78d6", linewidth=1.0)
        ax.axhline(0, color=RULE, linewidth=0.5); ax.axvline(REF_AGE, color=RULE, linewidth=0.5, linestyle=(0, (2, 2)))
        ax.set_title(ABBR[k], loc="left", fontsize=FONT_SIZE - 0.5, fontweight="bold", pad=1)
        despine(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.supxlabel("Landmark age (years)", fontsize=FONT_SIZE); fig.supylabel(f"Change in {h // 12}-year risk (pp)", fontsize=FONT_SIZE)
    fig.tight_layout(); stamp(fig); save(fig, f"ExtFig5_age_ale_all_{h}m"); plt.close(fig)


def ext_lookback_heatmap(h):
    d = C.query("horizon_months == @h")
    windows = list(dict.fromkeys(d.sort_values("window_low_years").window))
    piv = lambda col: d.pivot(index="condition_id", columns="window", values=col).reindex(index=TARGET_ORDER, columns=windows)
    value, low, high = piv("risk_ratio").to_numpy(float), piv("ratio_ci_low").to_numpy(float), piv("ratio_ci_high").to_numpy(float)
    limit = ratio_limit(value)
    rgba = soften(DIVERGING(TwoSlopeNorm(0, -limit, limit)(np.clip(np.log(value), -limit, limit))), (low > 1) | (high < 1))
    fig = plt.figure(figsize=(2.6, 0.17 * len(TARGET_ORDER) + 1.1))
    grid = fig.add_gridspec(2, 1, height_ratios=[1, 0.04], hspace=0.3, left=0.2, right=0.95, top=0.95, bottom=0.12)
    ax = fig.add_subplot(grid[0])
    ax.imshow(rgba, aspect="auto", interpolation="nearest")
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            if np.isfinite(value[i, j]):
                ax.text(j, i, f"{value[i, j]:.2f}", ha="center", va="center", fontsize=FONT_SIZE - 2.5,
                        color="white" if abs(np.log(value[i, j])) > 0.6 * limit else INK)
    ax.set_xticks(range(len(windows)), [window_label(w) for w in windows]); ax.set_yticks(range(len(TARGET_ORDER)), [ABBR[k] for k in TARGET_ORDER])
    ax.set_xlabel("Years before landmark"); ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ratio_colourbar(fig, fig.add_subplot(grid[1]), limit, "Risk ratio, recorded vs removed")
    stamp(fig); save(fig, f"ExtFig5_lookback_by_outcome_{h}m"); plt.close(fig)


for h in HORIZONS:
    for function in (ext_selected, ext_by_sex, ext_pp, ext_forest, ext_age_grid, ext_lookback_heatmap):
        function(h)
print("Extended-data figures written.")


# =============================================================================
# Notebook cell 15
# =============================================================================

def legend_text(h):
    years = h // 12
    boot = manifest["bootstrap"]
    d = A.query("horizon_months == @h and sex == 'all' and displayable")
    top = d.nlargest(3, "risk_ratio")
    pairs = "; ".join(f"{r.source_abbreviation}→{r.target_abbreviation} {r.risk_ratio:.2f} ({r.ratio_ci_low:.2f}–{r.ratio_ci_high:.2f})"
                      for r in top.itertuples())
    cal = CAL.query("horizon_months == @h and exposed_events >= @B_MIN_EVENTS and observed_ratio > 0") if CAL is not None else None
    rho = (pd.Series(np.log(cal.predicted_ratio.to_numpy())).corr(pd.Series(np.log(cal.observed_ratio.to_numpy())), method="spearman")
           if cal is not None and len(cal) >= 3 else float("nan"))
    windows = manifest["lookback_windows_years"]
    text = f"""
Fig. 5 | From population patterns to individual patients. a, For every prior diagnosis (rows) and outcome (columns, plus death), the mean change in predicted {years}-year risk when the diagnosis is removed from the history, among test participants who had it (n = {manifest['n_test_participants']:,} test participants), shown separately for women and men; rows and columns are ordered and coloured by ICD-10 chapter; white, too few participants to show. Strongest risk ratios among the highlighted conditions (extended data): {pairs}. b, For each pair of the highlighted conditions, the age- and sex-standardised risk ratio with vs without the prior diagnosis: observed in the test set (x; inverse-probability-of-censoring weighted, horizontal lines 95% intervals) against the model's prediction (y); point size, number of outcome events; pairs with fewer than {B_MIN_EVENTS} events omitted; line of identity shown (Spearman ρ = {rho:.2f}; event-weighted slope of log predicted on log observed {SLOPES.get(h, float('nan')):.2f}). c, Accumulated local effect of landmark age on {years}-year risk, centred at the training-median age ({REF_AGE:.0f} years); the history is shifted with age so event timing relative to the landmark is kept; coloured lines are the outcomes with the largest effect (95% intervals shaded). d, Risk ratio when diagnoses recorded in each lookback window are removed, among participants with a diagnosis in that window (top, share of participants); dots, conditions; line, geometric mean; diamonds, death. e, f, Two illustrative synthetic patients (no real participant identifiers or observed outcomes): synthetic history before the landmark (CRP values in mg/L), the 6-month entry delay (grey) and the {manifest['patients']['horizon_months'] // 12}-year prediction window (blue), predicted {PATIENT_TARGET} risk at 2, 5 and 10 years, and exact Shapley contributions of eight input groups (sex held as observed) to the {manifest['patients']['horizon_months'] // 12}-year risk. Intervals: {boot['n']} participant bootstrap resamples (percentile, {100 * (1 - boot['alpha']):.0f}%).
"""
    return textwrap.fill(" ".join(text.split()), 110)


for h in HORIZONS:
    (FIGURE_DIR / f"Fig5_legend_draft_{h}m.txt").write_text(legend_text(h), encoding="utf-8")
print(legend_text(HORIZONS[0]))
