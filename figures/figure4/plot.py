"""Fig. 4: input contributions to predicted risk (Shapley attributions and lifestyle contrasts).

Redraws manuscript Fig. 4 and its supplementary figure from the export in `source_data/` and
writes PDF, SVG, PNG, captions, source-data workbook and a figure manifest to `output/`.
No model is loaded and no UK Biobank access is needed.

    python figures/figure4/plot.py

Plot settings are environment variables (FIG4_LAYOUT, FIG4_BEESWARM_AXIS, FIG4_VALUE_RAMP, ...),
as listed in the notebook `Fig4_plot_explainability_medians_v4`. Its cells are copied unchanged,
with cell numbers kept, apart from the default input and output folders in cell 2.
`source_data/` is filled by `collect_source_data.py`.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw without a display; files are written to output/

HERE = Path(__file__).resolve().parent


# =============================================================================
# Notebook cell 2
# =============================================================================

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

# Repository paths: the export is read from source_data/ and figures are written to output/.
EXPORT_DIR = Path(os.getenv("FIG4_EXPORT_DIR", str(HERE / "source_data"))).expanduser().resolve()
FIGURE_DIR = Path(os.getenv("FIG4_FIGURE_DIR", str(HERE / "output"))).expanduser().resolve()
BEE_AXIS_SETTING = os.getenv("FIG4_BEESWARM_AXIS", "auto").strip().lower()
BEE_LINEAR_THRESHOLD_SETTING = os.getenv("FIG4_BEESWARM_LINEAR_THRESHOLD_PP", "auto").strip().lower()
BEE_LINEAR_WIDTH_SETTING = os.getenv("FIG4_BEESWARM_LINEAR_WIDTH_PP", "auto").strip().lower()
BEE_MAX_PER_BIN = int(os.getenv("FIG4_BEESWARM_MAX_PER_BIN", "0"))
C_VALUES = os.getenv("FIG4_C_VALUES", "1") != "0"
LAYOUT = os.getenv("FIG4_LAYOUT", "paired").strip().lower()
if LAYOUT not in {"paired", "stacked"}:
    raise ValueError("FIG4_LAYOUT must be 'paired' or 'stacked'.")
# Panel letters by layout. Paired puts overall and individual contributions side by side on shared rows.
LETTERS = {"paired": {"overall": "A", "individual": "B", "direction": "C", "lifestyle": "D"},
           "stacked": {"overall": "A", "direction": "B", "lifestyle": "C", "individual": "D"}}[LAYOUT]
BEE_AUTO_TAIL_RATIO = 8.0
if BEE_AXIS_SETTING not in {"auto", "asinh", "linear", "symlog"}:
    raise ValueError("FIG4_BEESWARM_AXIS must be 'auto', 'asinh', 'linear' or 'symlog'.")
if BEE_LINEAR_WIDTH_SETTING != "auto":
    if not np.isfinite(float(BEE_LINEAR_WIDTH_SETTING)) or float(BEE_LINEAR_WIDTH_SETTING) <= 0:
        raise ValueError("FIG4_BEESWARM_LINEAR_WIDTH_PP must be 'auto' or positive and finite.")
if BEE_MAX_PER_BIN < 0:
    raise ValueError("FIG4_BEESWARM_MAX_PER_BIN must be >= 0 (0 draws every point).")
if BEE_LINEAR_THRESHOLD_SETTING != "auto":
    fixed = float(BEE_LINEAR_THRESHOLD_SETTING)
    if not np.isfinite(fixed) or fixed <= 0:
        raise ValueError("FIG4_BEESWARM_LINEAR_THRESHOLD_PP must be 'auto' or positive and finite.")

manifest_path = EXPORT_DIR / "manifest.json"
if not manifest_path.is_file():
    staging = [p for p in EXPORT_DIR.glob("tmp*") if p.is_dir()] if EXPORT_DIR.is_dir() else []
    others = sorted(str(p.parent) for p in EXPORT_DIR.parent.glob("export_*/manifest.json"))
    if staging:
        hint = ("The generator is still writing this export (a temporary staging folder exists). "
                "Wait for its log line 'Figure 4 data written to' and rerun this cell.")
    else:
        hint = ("Fill source_data/ with figures/figure4/collect_source_data.py, "
                "or set FIG4_EXPORT_DIR to a completed export.")
    raise FileNotFoundError(f"{manifest_path} was not found. {hint}"
                            + (f"\nCompleted exports next to it: {others}" if others else ""))
MANIFEST_SHA256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("schema") != "crescent.fig4.explainability.v3":
    raise ValueError("Use the v3 exports from Fig4_generate_explainability_data_medians.ipynb.")


def load_table(filename, required=True):
    path = EXPORT_DIR / filename
    recorded = manifest["files"].get(filename)
    if recorded is None or not path.is_file():
        if required:
            raise FileNotFoundError(f"Missing source file {path}. Regenerate the Figure 4 export.")
        print(f"Optional input not in this export, skipped: {filename}")
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
        raise ValueError(f"Source file does not match the manifest: {path}")
    return pd.read_csv(path)


A = load_table("fig4A_all_condition_shapley.csv")
B = load_table("fig4B_directional_shapley.csv")
C = load_table("fig4C_lifestyle_contrasts.csv")
D = load_table("fig4D_beeswarm_points.csv")
D_SUMMARY = load_table("fig4D_beeswarm_summary.csv")
SHARE = load_table("fig4A_attribution_share_sensitivity.csv", required=False)
BRIER = load_table("evaluation_ipcw_brier_reference_edit.csv", required=False)
AGE_AUDIT = load_table("audit_shifted_diagnosis_ages.csv", required=False)
AGE_TAILS = load_table("audit_age_attribution_tails.csv", required=False)

DISPLAY_CONDITIONS = manifest["illustrative_condition_ids"]
# Conditions the beeswarm sample was drawn from: every fitted condition in current exports,
# the illustrative ones in exports made before that option existed.
BEE_CONDITION_IDS = manifest["beeswarm"].get("condition_ids", DISPLAY_CONDITIONS)
GROUP_ORDER = manifest["beeswarm"]["rows"]
HORIZONS = sorted(manifest["horizons_months"], key=lambda h: (h != 60, h))
DEV = manifest["development_subset"]
if sorted(GROUP_ORDER) != sorted(manifest["shapley_groups"]):
    raise ValueError("Beeswarm rows and Shapley groups differ in the manifest.")
conditions = B[["condition_id", "abbreviation", "condition_name"]].drop_duplicates().set_index("condition_id")
if not conditions.index.is_unique or not set(DISPLAY_CONDITIONS).issubset(conditions.index):
    raise ValueError("The illustrative condition order does not match the input table.")
CONDITION_LABELS = conditions.abbreviation.fillna(conditions.condition_name).to_dict()

for h in HORIZONS:
    a_groups = set(A.loc[A.horizon_months.eq(h) & A.scope.eq("all conditions"), "group"])
    if a_groups != set(GROUP_ORDER):
        raise ValueError(f"Panel A does not cover every input group at {h} months.")
    points = D.loc[D.horizon_months.eq(h)]
    groups = {group: part.set_index("sample_pair_id").condition_id.sort_index()
              for group, part in points.groupby("group")}
    if points.empty or set(groups) != set(GROUP_ORDER):
        raise ValueError(f"Incomplete pooled beeswarm at {h} months.")
    matched = groups[GROUP_ORDER[0]]
    if not matched.index.is_unique or any(not pairs.equals(matched) for pairs in groups.values()):
        raise ValueError(f"Beeswarm rows do not share the same sample pairs at {h} months.")
    counts = matched.value_counts()
    if counts.max() > int(manifest["beeswarm"]["max_sample_per_condition"]) or not set(counts.index).issubset(BEE_CONDITION_IDS):
        raise ValueError(f"Beeswarm sample exceeds its per-condition cap or uses unexpected conditions at {h} months.")
    summary = D_SUMMARY.loc[D_SUMMARY.horizon_months.eq(h)].set_index("group").reindex(GROUP_ORDER)
    for group in GROUP_ORDER:
        shown = points.loc[points.group.eq(group)]
        if (int(summary.loc[group, "zero_n_sampled"]) != int(shown.contribution_pp.eq(0).sum())
                or int(summary.loc[group, "sampled_pairs_n"]) != len(shown)):
            raise ValueError(f"Beeswarm summary disagrees with sampled points for {group} at {h} months.")


def beeswarm_axis(h):
    """Return the axis scale for panel D and the reason it was chosen."""
    if BEE_AXIS_SETTING != "auto":
        return BEE_AXIS_SETTING, "set by FIG4_BEESWARM_AXIS"
    data = D.loc[D.horizon_months.eq(h), "contribution_pp"].abs()
    summary = D_SUMMARY.loc[D_SUMMARY.horizon_months.eq(h)]
    bulk = float(np.nanmedian(np.abs(summary[["nonzero_sample_p01_pp", "nonzero_sample_p99_pp"]]).max(axis=1)))
    ratio = float(data.max()) / bulk if bulk > 0 else np.inf
    scale = "asinh" if ratio > BEE_AUTO_TAIL_RATIO else "linear"
    return scale, f"auto: largest |contribution| is {ratio:.1f}× the median row 1st–99th percentile"


BEE_AXIS = {h: beeswarm_axis(h) for h in HORIZONS}


def linear_threshold(h):
    """Half-width of the linear centre of the symlog axis.

    The power of ten just below the typical input's median |non-zero contribution| (the median of
    the row medians). Small inputs then spread over the linear centre and first decades instead of
    collapsing onto zero, without stretching negligible values across the axis.
    """
    if BEE_LINEAR_THRESHOLD_SETTING != "auto":
        return float(BEE_LINEAR_THRESHOLD_SETTING)
    data = D.loc[D.horizon_months.eq(h) & D.contribution_pp.ne(0)]
    medians = data.groupby("group").contribution_pp.apply(lambda v: float(np.median(np.abs(v))))
    typical = float(medians.median()) if len(medians) else .1
    return float(np.clip(10 ** np.floor(np.log10(typical)), 1e-3, .1))


BEE_LINTHRESH = {h: linear_threshold(h) for h in HORIZONS}


def asinh_linear_width(h):
    """Linear width of the asinh axis: the typical input's median |non-zero contribution|.

    The median of the row medians, snapped to a 1–2–5 ladder between 0.001 and 1 pp. Values
    smaller than this sit on the near-linear centre; larger ones are compressed smoothly.
    """
    if BEE_LINEAR_WIDTH_SETTING != "auto":
        return float(BEE_LINEAR_WIDTH_SETTING)
    data = D.loc[D.horizon_months.eq(h) & D.contribution_pp.ne(0)]
    medians = data.groupby("group").contribution_pp.apply(lambda v: float(np.median(np.abs(v))))
    typical = float(medians.median()) if len(medians) else .1
    ladder = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .5, 1.])
    return float(ladder[np.argmin(np.abs(np.log(ladder / typical)))])


BEE_ASINH_WIDTH = {h: asinh_linear_width(h) for h in HORIZONS}
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
print("Reading:", EXPORT_DIR, "| manifest", MANIFEST_SHA256[:12], "| created", manifest.get("created_utc"))
print("Writing:", FIGURE_DIR)
print("Panel D scale check (sampled non-zero contributions, pp):")
print(D_SUMMARY[["horizon_months", "group", "nonzero_sample_p01_pp", "nonzero_sample_p99_pp",
                 "zero_fraction_eligible_equal_condition_weight"]].round(3).to_string(index=False))
for h, (scale, reason) in BEE_AXIS.items():
    note = ({"symlog": f"; linear within ±{BEE_LINTHRESH[h]:g} pp",
             "asinh": f"; asinh linear width {BEE_ASINH_WIDTH[h]:g} pp"}.get(scale, "")
            + (f"; thinning ≤{BEE_MAX_PER_BIN} dots per 1-pt bin" if BEE_MAX_PER_BIN else ""))
    print(f"Beeswarm axis at {h} months: {scale} ({reason}{note})")


# =============================================================================
# Notebook cell 4
# =============================================================================

FONT_SIZE = 7
SMALLEST = 5          # floor for any text in the figure
FIGURE_WIDTH_IN = 180 / 25.4
mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - .5, "ytick.labelsize": FONT_SIZE - .5,
    "axes.linewidth": .55, "xtick.major.width": .55, "ytick.major.width": .55,
    "xtick.major.size": 2.3, "ytick.major.size": 2.3, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "figure.facecolor": "white", "savefig.facecolor": "white", "figure.dpi": 140,
    "axes.unicode_minus": True,
})
INK, MUTED, RULE, GRID = "#23232b", "#51515d", "#a9a7ad", "#e9e7ed"
IQR_INK = "#3e454d"   # charcoal for the IQR bar and median ring
FAMILY_COLOURS = {"Clinical history": "#2a78d6", "Lifestyle": "#eb6834",
                  "Biomarker": "#1baf7a", "Sociodemographic": "#52514e"}
FAMILY = {"Diagnosis history": "Clinical history", "BMI history": "Lifestyle",
          "Smoking history": "Lifestyle", "Alcohol history": "Lifestyle", "CRP history": "Biomarker"}
GROUP_LABELS = {"Diagnosis history": "Diagnoses", "Landmark age": "Age", "BMI history": "BMI",
                "Smoking history": "Smoking", "Alcohol history": "Alcohol", "CRP history": "CRP",
                "Townsend deprivation": "Deprivation", "Ethnicity": "Ethnicity"}
GROUP_COLOURS = {group: FAMILY_COLOURS[FAMILY.get(group, "Sociodemographic")] for group in GROUP_ORDER}
NEGATIVE, POSITIVE = "#4285b7", "#c9584f"
RISK_RATIOS = LinearSegmentedColormap.from_list(
    "risk_ratio", ["#0d366b", "#3775b6", "#86b6ef", "#f0efec", "#f1a3a2", "#d03b3b", "#7a1f1f"])
# Percentile colour ramp for the individual-contribution panel (FIG4_VALUE_RAMP).
#   teal  : teal (lower) -> violet -> plum (higher). v3's end colours with a saturated middle, so no
#           value fades towards the light-grey "unavailable" colour; no clash with the orange in A.
#   plasma: indigo (lower) -> magenta -> orange (higher), trimmed before its pale yellow end.
# Both change hue and lightness, so low and high values stay distinct among overlapping dots.
VALUE_RAMPS = {
    "teal": ["#3a9fb0", "#4f86b4", "#6a6bb0", "#7d4c97", "#6a1f42"],
    "plasma": ["#1b068d", "#5502a4", "#8606a6", "#b12a90", "#d14e72", "#eb7556", "#fb9f3a"],
}
VALUE_RAMP_NAME = os.getenv("FIG4_VALUE_RAMP", "teal").strip().lower()
if VALUE_RAMP_NAME not in VALUE_RAMPS:
    raise ValueError(f"FIG4_VALUE_RAMP must be one of {sorted(VALUE_RAMPS)}.")
VALUE_RAMP_STOPS = VALUE_RAMPS[VALUE_RAMP_NAME]
VALUE_RAMP = LinearSegmentedColormap.from_list("recorded_value", VALUE_RAMP_STOPS)
MISSING_VALUE = "#cfcdd4"   # light neutral, drawn beneath the coloured dots

ratios = C.loc[C.condition_id.isin(DISPLAY_CONDITIONS), "risk_ratio"].to_numpy()
log_ratios = np.log(ratios[np.isfinite(ratios) & (ratios > 0)])
ratio_limit = max(np.log(1.05), float(np.abs(log_ratios).max(initial=0)))
RATIO_NORM = TwoSlopeNorm(vmin=-ratio_limit, vcenter=0, vmax=ratio_limit)


def minus(text):
    return text.replace("-", "−")


def decimals_for(values, significant=2, most=3):
    """One decimal count for a set of values, so the smallest still shows `significant` figures."""
    values = np.abs(np.asarray(values, float))
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return 1
    return int(np.clip(significant - 1 - np.floor(np.log10(values.min())), 0, most))


def trim(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(colors=INK)


def panel_heading(fig, y, label, x=.052):
    """Bold panel letter only; what each panel shows is stated by its axes and the caption."""
    fig.text(x, y, label, va="center", color=INK, fontweight="bold", fontsize=FONT_SIZE + 2)


def colourbar(fig, ax, cmap, norm, ticks, labels, label, orientation="horizontal"):
    bar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation=orientation)
    bar.set_ticks(ticks, labels=labels)
    bar.ax.tick_params(length=1.5, labelsize=FONT_SIZE - 1, colors=INK, pad=2)
    bar.set_label(label, fontsize=FONT_SIZE - 1, labelpad=2)
    bar.outline.set_visible(False)
    return bar


def luminance(rgb):
    return float(np.dot([.2126, .7152, .0722], np.asarray(rgb[:3]) ** 2.2))


# =============================================================================
# Notebook cell 6
# =============================================================================

def panel_a(ax, h, headroom=1.3):
    d = A.loc[A.horizon_months.eq(h) & A.scope.eq("all conditions")].set_index("group").reindex(GROUP_ORDER)
    share = (SHARE.loc[SHARE.horizon_months.eq(h)].set_index("group").reindex(GROUP_ORDER)
             if SHARE is not None else None)
    y = np.arange(len(d))
    point = d.mean_abs_shapley_pp.to_numpy()
    low, high = d.ci_low_pp.to_numpy(), d.ci_high_pp.to_numpy()
    ax.barh(y, point, height=.56, color=[GROUP_COLOURS[g] for g in d.index], linewidth=0)
    has_ci = np.isfinite(low) & np.isfinite(high)
    ax.hlines(y[has_ci], low[has_ci], high[has_ci], color=INK, linewidth=.7, zorder=3)
    extent = np.fmax(point, np.where(has_ci, high, point))
    limit = float(np.nanmax(extent, initial=0)) or 1.0
    places = decimals_for(point)
    for row, group, value, end in zip(y, d.index, point, extent):
        if not np.isfinite(value):
            continue
        label = ax.text(end + .02 * limit, row, f"{value:.{places}f}", va="center", color=INK,
                        fontsize=FONT_SIZE - .5)
        if share is not None and np.isfinite(share.loc[group, "mean_attribution_share_pct"]):
            ax.annotate(f"  ({share.loc[group, 'mean_attribution_share_pct']:.0f}%)", xy=(1, .5),
                        xycoords=label, va="center", color=MUTED, fontsize=FONT_SIZE - 1)
    ax.set_yticks(y, [GROUP_LABELS.get(g, g) for g in d.index])
    ax.set_xlim(0, limit * headroom)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylim(len(d) - .5, -.6)
    trim(ax)


def panel_b(fig, axes, h):
    data = B.loc[B.horizon_months.eq(h) & B.condition_id.isin(DISPLAY_CONDITIONS)]
    y = np.arange(len(DISPLAY_CONDITIONS))
    predicted = (data[["condition_id", "mean_predicted_risk_pct"]].drop_duplicates()
                 .set_index("condition_id").mean_predicted_risk_pct.reindex(DISPLAY_CONDITIONS))
    labels = [f"{CONDITION_LABELS[k]} ({risk:.2g}%)" if np.isfinite(risk) else CONDITION_LABELS[k]
              for k, risk in predicted.items()]
    for ax, group in zip(axes, GROUP_ORDER):
        rows = data.loc[data.group.eq(group)].set_index("condition_id").reindex(DISPLAY_CONDITIONS)
        left, right = rows.mean_negative_pp.to_numpy(), rows.mean_increase_pp.to_numpy()
        left_lo, left_hi = rows.negative_ci_low_pp.to_numpy(), rows.negative_ci_high_pp.to_numpy()
        right_lo, right_hi = rows.increase_ci_low_pp.to_numpy(), rows.increase_ci_high_pp.to_numpy()
        net = rows.mean_signed_shapley_pp.to_numpy()
        valid = np.isfinite(left) & np.isfinite(right)
        reach = np.abs(np.r_[left[valid], right[valid], left_lo[valid], right_hi[valid]])
        extent = float(np.nanmax(reach, initial=0))
        limit = MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]).tick_values(0, extent * 1.02)[-1] if extent else 1
        if extent:
            # The zero line is drawn, so 0 keeps its tick without a label; end labels never collide.
            ax.set_xticks([-limit, 0, limit], [f"−{limit:.2g}", "", f"+{limit:.2g}"])
            ticks = ax.get_xticklabels()
            ticks[0].set_ha("left")
            ticks[-1].set_ha("right")
        else:
            ax.set_xticks([0], ["0"])
        ax.set_xlim(-limit, limit)
        ax.axvline(0, color=RULE, linewidth=.55)
        ax.barh(y[valid], left[valid], color=NEGATIVE, height=.66, linewidth=0)
        ax.barh(y[valid], right[valid], color=POSITIVE, height=.66, linewidth=0)
        for lo, hi in ((left_lo, left_hi), (right_lo, right_hi)):
            ok = valid & np.isfinite(lo) & np.isfinite(hi) & (np.abs(hi - lo) > 0)
            ax.hlines(y[ok], lo[ok], hi[ok], color=INK, linewidth=.45, alpha=.75, zorder=3)
        ax.scatter(net[valid], y[valid], marker="D", s=9, color="white", edgecolors=INK, linewidths=.65, zorder=4)
        both_zero = valid & (left == 0) & (right == 0)
        ax.scatter(np.zeros(both_zero.sum()), y[both_zero], s=4, color=MUTED, linewidths=0, zorder=3)
        ax.set_title(GROUP_LABELS.get(group, group), color=INK, fontsize=FONT_SIZE, pad=3)
        ax.tick_params(axis="x", length=1.7, pad=1.5, labelsize=FONT_SIZE - .65)
        ax.tick_params(axis="y", length=0, pad=2.3)
        trim(ax)
    axes[0].set_ylim(len(y) - .45, -.55)
    axes[0].set_yticks(y, labels)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)
    left_pos, right_pos = axes[0].get_position(), axes[-1].get_position()
    fig.text((left_pos.x0 + right_pos.x1) / 2, left_pos.y0 - .026,
             "Mean Shapley contribution (pp), input-specific scales  ·  brackets: mean predicted risk",
             color=INK, fontsize=FONT_SIZE - .1, ha="center", va="top")


B_LEGEND = [Patch(color=NEGATIVE, label="Lowers risk"), Patch(color=POSITIVE, label="Raises risk"),
            Line2D([], [], marker="D", markersize=3.4, linestyle="none", markerfacecolor="white",
                   markeredgecolor=INK, markeredgewidth=.65, label="Net")]


# =============================================================================
# Notebook cell 8
# =============================================================================

LEVEL_SHORT = {"Underweight": "Under", "Overweight": "Over", "Obese": "Obese",
               "Previous": "Former", "Current": "Current", "Special occasions": "Occas.",
               "1–3/month": "1–3/mo", "1–2/week": "1–2/wk", "3–4/week": "3–4/wk", "Daily": "Daily"}
C_DOMAINS = ("BMI", "Smoking", "Alcohol")


def c_levels(h, domain):
    data = C.loc[C.horizon_months.eq(h) & C.domain.eq(domain)]
    levels = data.sort_values("level_rank").level.unique().tolist()
    if not levels:
        raise ValueError(f"Missing lifestyle categories for {domain} at {h} months.")
    return data, levels


def panel_c(fig, axes, cax, h):
    for i, (ax, domain) in enumerate(zip(axes, C_DOMAINS)):
        data, levels = c_levels(h, domain)
        data = data.loc[data.condition_id.isin(DISPLAY_CONDITIONS)]

        def values(column):
            return data.pivot(index="condition_id", columns="level", values=column).reindex(
                index=DISPLAY_CONDITIONS, columns=levels).to_numpy(dtype=float)

        ratio = values("risk_ratio")
        known = np.isfinite(ratio) & (ratio > 0)
        log_data = np.full(ratio.shape, np.nan)
        np.log(ratio, out=log_data, where=known)
        ax.imshow(np.ma.masked_invalid(log_data), cmap=RISK_RATIOS, norm=RATIO_NORM, aspect="auto",
                  interpolation="nearest")
        excludes_one = (values("ratio_ci_low") > 1) | (values("ratio_ci_high") < 1)
        for r, c in zip(*np.nonzero(~known)):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor="white", edgecolor=GRID,
                                   hatch="//////", linewidth=0, zorder=1))
        for r, c in zip(*np.nonzero(known)):
            background = RISK_RATIOS(RATIO_NORM(log_data[r, c]))
            ink = "white" if luminance(background) < .3 else INK
            if C_VALUES:
                ax.text(c, r, f"{ratio[r, c]:.2f}", ha="center", va="center", fontsize=SMALLEST,
                        color=ink, alpha=1 if excludes_one[r, c] else .42, zorder=3)
            elif excludes_one[r, c]:
                ax.scatter(c, r, s=3, color=ink, linewidths=0, zorder=3)
        ax.set_xticks(np.arange(len(levels)), [LEVEL_SHORT.get(level, level) for level in levels])
        ax.set_yticks(np.arange(len(DISPLAY_CONDITIONS)),
                      [CONDITION_LABELS[k] for k in DISPLAY_CONDITIONS] if i == 0 else [])
        ax.set_xticks(np.arange(len(levels) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(DISPLAY_CONDITIONS) + 1) - .5, minor=True)
        ax.grid(which="minor", color="white", linewidth=.65)
        ax.tick_params(which="both", length=0, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{domain} vs {data.reference.iloc[0].lower()}", loc="center", color=INK, pad=4,
                     fontsize=FONT_SIZE + .5)
    candidates = np.array([.125, .25, .5, .67, .8, .9, .95, 1, 1.05, 1.1, 1.25, 1.5, 2, 4, 8])
    ticks = candidates[np.abs(np.log(candidates)) <= ratio_limit]
    if len(ticks) > 5:
        lower, upper = ticks[ticks < 1], ticks[ticks > 1]
        ticks = np.unique(np.r_[lower[0], lower[len(lower) // 2], 1., upper[len(upper) // 2], upper[-1]])
    colourbar(fig, cax, RISK_RATIOS, RATIO_NORM, np.log(ticks), [f"{tick:g}" for tick in ticks],
              "Predicted risk ratio", orientation="vertical")


def sina_offsets(x_pt, half_height, seed):
    """Sina-plot offsets: uniform jitter scaled by the local density of contributions.

    Density is a Gaussian-smoothed histogram on a 1-pt grid along the drawn axis, so the row's
    outline shows where contributions concentrate and every point is still drawn.
    """
    if len(x_pt) == 0:
        return np.zeros(0)
    edges = np.arange(np.floor(x_pt.min()) - 1, np.ceil(x_pt.max()) + 2, 1.0)
    counts, _ = np.histogram(x_pt, edges)
    spread = np.std(x_pt) or 1.0
    bandwidth = float(np.clip(.9 * spread * len(x_pt) ** -.2, 1.5, 8))     # points
    kernel_x = np.arange(-4 * bandwidth, 4 * bandwidth + 1)
    kernel = np.exp(-.5 * (kernel_x / bandwidth) ** 2)
    density = np.convolve(counts, kernel / kernel.sum(), mode="same")
    local = np.interp(x_pt, edges[:-1] + .5, density)
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, len(x_pt)) * half_height * local / density.max()


def thin_dense_bins(x_pt, max_per_bin, seed):
    """Random subset with at most `max_per_bin` points per 1-pt bin; sparse bins keep every point."""
    keep = np.ones(len(x_pt), dtype=bool)
    if not max_per_bin or len(x_pt) == 0:
        return keep
    rng = np.random.default_rng(seed)
    bins = np.floor(x_pt).astype(int)
    for b in np.unique(bins):
        idx = np.flatnonzero(bins == b)
        if len(idx) > max_per_bin:
            keep[idx] = False
            keep[rng.choice(idx, max_per_bin, replace=False)] = True
    return keep


def asinh_ticks(lo_val, hi_val, linear_width, to_unit, compact=False):
    """Clean, non-overlapping 1–2–5 ticks on an asinh axis (adapted from v3)."""
    ladder = np.array([.001, .002, .005, .01, .02, .05, .1, .2, .5, 1, 2, 5, 10, 20, 50, 100])
    base = ladder[ladder >= (1.6 if compact else .75) * linear_width]
    candidates = np.r_[-base[::-1], 0., base]
    candidates = candidates[(candidates >= lo_val * .98) & (candidates <= hi_val * .98)]
    t_lo, t_hi = to_unit.transform(np.array([lo_val, hi_val]))
    span_t = max(float(t_hi - t_lo), 1e-9)
    min_gap = .13 if compact else .085
    t_zero = float(to_unit.transform(np.array([0.]))[0])
    kept = [0.]
    for side in (candidates[candidates > 0], candidates[candidates < 0][::-1]):
        last = t_zero
        for v in side:
            tv = float(to_unit.transform(np.array([v]))[0])
            if abs(tv - last) / span_t >= min_gap:
                kept.append(float(v)); last = tv
    return sorted(kept)


BEE_DOT = 1.7          # marker area, pt²  (≈1.3 pt diameter): every sampled point is drawn


def panel_d(fig, ax, cax, h, row_labels=True, compact=False):
    data = D.loc[D.horizon_months.eq(h)]
    summary = D_SUMMARY.loc[D_SUMMARY.horizon_months.eq(h)].set_index("group").reindex(GROUP_ORDER)
    scale, _ = BEE_AXIS[h]
    extent = float(np.abs(data.contribution_pp).max())
    span = 1.06 * extent if extent > 0 else 1
    if scale == "symlog":
        ax.set_xscale("symlog", linthresh=BEE_LINTHRESH[h], linscale=1)
    elif scale == "asinh":
        ax.set_xscale("asinh", linear_width=BEE_ASINH_WIDTH[h])
    ax.set_xlim(-span, span)
    ax.set_ylim(len(GROUP_ORDER) - .5, -.6)
    # Work in points so that dots of a fixed physical size never overlap where there is room.
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())   # inches
    width_pt, height_pt = bbox.width * 72, bbox.height * 72
    to_unit = ax.xaxis.get_transform()
    lo, hi = to_unit.transform(np.array([-span, span]))
    row_pt = height_pt / (len(GROUP_ORDER) + .1)
    ax.axvline(0, color=RULE, linewidth=.65, zorder=1)
    ax.text(1.035, -.62, "At zero", transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=FONT_SIZE - .45, color=MUTED)
    for row, group in enumerate(GROUP_ORDER):
        shown = data.loc[data.group.eq(group)]
        nonzero = shown.contribution_pp.ne(0)
        x = shown.loc[nonzero, "contribution_pp"].to_numpy()
        values = shown.loc[nonzero, "recorded_value_percentile"].to_numpy()
        x_pt = (to_unit.transform(x) - lo) / (hi - lo) * width_pt
        offsets = sina_offsets(x_pt, .42 * row_pt, seed=1000 * h + row) / row_pt
        keep = thin_dense_bins(x_pt, BEE_MAX_PER_BIN, seed=2000 * h + row)
        x_all = x
        x, values, offsets = x[keep], values[keep], offsets[keep]
        known = np.isfinite(values)
        order = np.random.default_rng(row).permutation(int(known.sum()))   # no colour drawn systematically on top
        ax.axhline(row, color=GRID, linewidth=.45, zorder=0)
        ax.scatter(x[~known], row + offsets[~known], color=MISSING_VALUE, s=BEE_DOT, linewidths=0,
                   rasterized=True, zorder=2)
        ax.scatter(x[known][order], row + offsets[known][order], c=values[known][order], cmap=VALUE_RAMP,
                   vmin=0, vmax=1, s=BEE_DOT, linewidths=0, alpha=.85, rasterized=True, zorder=3)
        if len(x_all):   # summaries use every non-zero sampled point, thinned or not
            q1, median, q3 = np.percentile(x_all, [25, 50, 75])
            ax.hlines(row, q1, q3, color=IQR_INK, linewidth=2.3 if compact else 2.6, alpha=.92, zorder=5,
                      capstyle="butt")
            ax.scatter(median, row, s=11 if compact else 12, color="white", edgecolors=IQR_INK,
                       linewidths=.75, zorder=6)
        if not nonzero.all():
            ax.scatter(0, row, s=10, marker="|", color=MUTED, linewidths=.9, zorder=4)
        fraction = summary.loc[group, "zero_fraction_eligible_equal_condition_weight"]
        ax.text(1.035, row, f"{fraction:.0%}" if np.isfinite(fraction) else "–", va="center",
                transform=ax.get_yaxis_transform(), fontsize=FONT_SIZE - .35, color=INK)
    ax.set_yticks(np.arange(len(GROUP_ORDER)),
                  [GROUP_LABELS.get(group, group) for group in GROUP_ORDER] if row_labels else [])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", pad=2)
    if scale == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(7, symmetric=True))
    elif scale == "asinh":
        ax.set_xticks(asinh_ticks(-span, span, BEE_ASINH_WIDTH[h], ax.xaxis.get_transform(), compact=compact))
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    else:
        # Decade ticks from the linear threshold outward; every other decade when there are many.
        first = int(np.round(np.log10(BEE_LINTHRESH[h])))
        last = int(np.floor(np.log10(span)))
        decades = list(range(first, last + 1))
        if len(decades) > 4:
            decades = decades[::-1][::2][::-1]
        ticks = [10.0 ** k for k in decades]
        ax.set_xticks([-t for t in ticks[::-1]] + [0] + ticks)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: minus(f"{v:g}")))
    ax.set_xlabel(("Shapley contribution (pp)" if compact else
                   "Shapley contribution (percentage points of predicted risk)")
                  + {"symlog": "  ·  symmetric log axis", "asinh": "  ·  asinh axis"}.get(scale, ""), labelpad=2.5)
    trim(ax)
    colourbar(fig, cax, VALUE_RAMP, Normalize(0, 1), [0, 1], ["Lower", "Higher"],
              "Recorded input percentile  ·  light grey: unavailable")


# =============================================================================
# Notebook cell 10
# =============================================================================

def caption(h):
    refs = manifest["references"]
    lifestyle = ", ".join(f"{domain} to {value}" for domain, value in refs["lifestyle_levels"].items())
    fixed = " and ".join(name.lower() for name in manifest["fixed_as_observed"])
    b = manifest["bootstrap"]
    crp = (f"Measured CRP values were set to the training median ({refs['crp_training_median']:.3g} mg/L)."
           if refs["crp_history"] == "median" else "CRP history was deleted.")
    townsend = (f"Observed Townsend scores were set to the lower empirical training median "
                f"(z = {minus(format(refs['townsend_training_median_z'], '.3f'))}) with its original quintile."
                if refs["townsend"] == "median" else "Townsend deprivation was set to its encoded missing state.")
    phrase = {("BMI", "Normal"): "normal BMI", ("Smoking", "Never"): "never smoking",
              ("Alcohol", "Never"): "never drinking"}
    named = [phrase.get((d, C.loc[C.domain.eq(d), "reference"].iloc[0]),
                        f"{d} {C.loc[C.domain.eq(d), 'reference'].iloc[0].lower()}") for d in C_DOMAINS]
    c_refs = ", ".join(named[:-1]) + " or " + named[-1]
    scale, _ = BEE_AXIS[h]
    axis_note = {
        "symlog": (f" The horizontal axis is symmetric logarithmic, linear between −{BEE_LINTHRESH[h]:g} and "
                   f"+{BEE_LINTHRESH[h]:g} percentage points."),
        "asinh": (f" The horizontal axis is an inverse hyperbolic sine scale with linear width "
                  f"{BEE_ASINH_WIDTH[h]:g} percentage points (near-linear around zero, logarithmic in the tails)."),
    }.get(scale, "")
    if BEE_MAX_PER_BIN:
        axis_note += (f" For legibility, at most {BEE_MAX_PER_BIN} randomly chosen points are drawn per 1-pt "
                      f"interval of the axis; outlines and summaries use every sampled point.")
    sample = D.loc[D.horizon_months.eq(h) & D.group.eq(GROUP_ORDER[0])]
    per_condition = sample.condition_id.value_counts()
    n_sampled_conditions = int(sample.condition_id.nunique())
    n_pairs = len(sample)
    all_conditions = set(BEE_CONDITION_IDS) == set(range(int(manifest["n_fitted_conditions"])))
    which = (f"the {n_sampled_conditions} fitted conditions with eligible participants" if all_conditions
             else f"{n_sampled_conditions} illustrative conditions")
    cap = int(manifest["beeswarm"]["max_sample_per_condition"])
    sample_scope = (f"{int(per_condition.iloc[0])} from each of {which}" if per_condition.nunique() == 1 else
                    f"up to {cap} from each of {which}, and every eligible pair where a condition had fewer")
    share_note = (" Bracketed percentages give each input's mean share of the total absolute attribution, "
                  "averaged equally across conditions." if SHARE is not None else "")
    c_note = ("Cells give the ratio. Faded values have intervals that include 1."
              if C_VALUES else "Dots mark intervals that exclude 1.")
    pieces = {
        "overall": (f"Mean absolute Shapley contribution, averaged within each condition and then equally across "
                    f"all fitted conditions with eligible participants.{share_note}"),
        "direction": (f"Mean decreases (blue, left) and increases (red, right) in risk across all eligible "
                      f"participants for each illustrative condition and input group; diamonds give the net mean. "
                      f"Each input has its own symmetric scale. Brackets give the mean predicted risk in the "
                      f"condition's eligible risk set."),
        "lifestyle": (f"Ratios of mean predicted risks with every mapped record set to the stated category versus "
                      f"{c_refs}, among the same eligible participants with mapped records. {c_note} Hatched "
                      f"cells have no eligible participants."),
        "individual": (f"Individual Shapley contributions for {n_pairs:,} randomly sampled eligible participant–condition "
                       f"pairs ({sample_scope}), on the same input rows as {LETTERS['overall']}; the same "
                       f"pairs appear in every row. The vertical spread of each row is proportional to the density of "
                       f"contributions. Colour gives the participant's recorded value of that input as a percentile of "
                       f"the test cohort; light grey marks unavailable values. "
                       f"The ink bar and ring give the interquartile range and median of non-zero sampled "
                       f"contributions. The right column gives the share of all eligible pairs with an exact zero "
                       f"contribution, weighting conditions equally.{axis_note}"),
    }
    panels = " ".join(f"{letter}, {pieces[key]}" for key, letter in sorted(LETTERS.items(), key=lambda kv: kv[1]))
    return (
        f"Fig. 4 | Input contributions to predicted risk in the fitted CRESCENT model. "
        f"Predictions cover {h} months after {manifest['entry_delay_months']:g}-month entry and account for "
        f"competing death. Each condition was evaluated in its eligible subset of the "
        f"{manifest['n_test_participants']:,} held-out participants. Exact group Shapley contributions over "
        f"{len(GROUP_ORDER)} input groups ({manifest['n_coalitions']} coalitions) were computed in percentage "
        f"points (pp) of risk with recorded {fixed} held fixed. References: recorded lifestyle categories were set "
        f"to {lifestyle}. Landmark age was set to the training median ({refs['landmark_age_years']:.1f} years), "
        f"with event times moved by the same amount so that time since each event was unchanged. Prior diagnosis "
        f"records were deleted. {crp} {townsend} {panels} Intervals are pointwise "
        f"{100 * (1 - b['alpha']):g}% participant-bootstrap percentile intervals ({b['n']} resamples), "
        f"conditional on the fitted model."
    )


def draw_stacked(h):
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 9.4))
    left, right = .17, .925

    a = fig.add_axes((left, .85, right - left, .1))
    panel_a(a, h)

    bg = fig.add_gridspec(1, len(GROUP_ORDER), left=left, right=right, bottom=.585, top=.765, wspace=.42)
    axes_b = []
    for j in range(len(GROUP_ORDER)):
        axes_b.append(fig.add_subplot(bg[0, j], sharey=axes_b[0] if axes_b else None))
    panel_b(fig, axes_b, h)

    cg = fig.add_gridspec(1, 4, left=left, right=right + .02, bottom=.318, top=.508,
                          width_ratios=[3, 2, 5, .14], wspace=.06)
    axes_c = [fig.add_subplot(cg[0, j]) for j in range(3)]
    panel_c(fig, axes_c, fig.add_subplot(cg[0, 3]), h)

    d = fig.add_axes((left, .118, .86 - left, .145))
    bar_d = fig.add_axes((.415, .055, .2, .0055))
    panel_d(fig, d, bar_d, h)

    for y, key in ((.968, "overall"), (.8, "direction"), (.543, "lifestyle"), (.284, "individual")):
        panel_heading(fig, y, LETTERS[key])
    n_conditions = int(A.loc[A.horizon_months.eq(h), "n_eligible_conditions"].max())
    a.set_xlabel(f"Mean absolute Shapley contribution (pp), {h}-month risk, equal weight across "
                 f"{n_conditions} fitted conditions" + ("  ·  brackets: share of total" if SHARE is not None else ""),
                 labelpad=2)
    families = [Patch(facecolor=colour, label=name) for name, colour in FAMILY_COLOURS.items()]
    fig.legend(handles=families, loc="center right", bbox_to_anchor=(right, .968), ncol=4,
               fontsize=FONT_SIZE - .75, columnspacing=.9, handlelength=.8, handletextpad=.4, borderaxespad=0)
    fig.legend(handles=B_LEGEND, loc="center right", bbox_to_anchor=(right, .8), ncol=3,
               fontsize=FONT_SIZE - .7, handlelength=1, handletextpad=.4, columnspacing=1.1, borderaxespad=0)
    if DEV:
        fig.text(.5, .993, "DEVELOPMENT SUBSET: NOT FOR PUBLICATION", va="top", ha="center",
                 color="#9a2626", fontsize=FONT_SIZE, fontweight="bold")
    return fig


def draw_paired(h):
    """Overall (A) and individual (B) contributions side by side on shared input rows."""
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 9.0))
    left, right = .17, .925
    split = .405                      # A occupies left..split; B starts after a gutter
    n_conditions = int(A.loc[A.horizon_months.eq(h), "n_eligible_conditions"].max())

    a = fig.add_axes((left, .735, split - left, .21))
    panel_a(a, h, headroom=1.85)
    a.set_xlabel("Mean |contribution| (pp)", labelpad=2)
    d = fig.add_axes((split + .045, .735, .862 - split - .045, .21))
    bar_d = fig.add_axes((.53, .668, .2, .0045))
    panel_d(fig, d, bar_d, h, row_labels=False, compact=True)
    for y_row in range(len(GROUP_ORDER)):   # faint guide carries each row from A into B
        a.axhline(y_row, color=GRID, linewidth=.45, zorder=0)

    bg = fig.add_gridspec(1, len(GROUP_ORDER), left=left, right=right, bottom=.385, top=.565, wspace=.42)
    axes_b = []
    for j in range(len(GROUP_ORDER)):
        axes_b.append(fig.add_subplot(bg[0, j], sharey=axes_b[0] if axes_b else None))
    panel_b(fig, axes_b, h)

    cg = fig.add_gridspec(1, 4, left=left, right=right + .02, bottom=.045, top=.29,
                          width_ratios=[3, 2, 5, .14], wspace=.06)
    axes_c = [fig.add_subplot(cg[0, j]) for j in range(3)]
    panel_c(fig, axes_c, fig.add_subplot(cg[0, 3]), h)

    panel_heading(fig, .962, LETTERS["overall"])
    panel_heading(fig, .962, LETTERS["individual"], x=split + .01)
    panel_heading(fig, .606, LETTERS["direction"])
    panel_heading(fig, .33, LETTERS["lifestyle"])
    families = [Patch(facecolor=colour, label=name) for name, colour in FAMILY_COLOURS.items()]
    fig.legend(handles=families, loc="center right", bbox_to_anchor=(right, .985), ncol=4,
               fontsize=FONT_SIZE - .75, columnspacing=.9, handlelength=.8, handletextpad=.4, borderaxespad=0)
    fig.legend(handles=B_LEGEND, loc="center right", bbox_to_anchor=(right, .606), ncol=3,
               fontsize=FONT_SIZE - .7, handlelength=1, handletextpad=.4, columnspacing=1.1, borderaxespad=0)
    if DEV:
        fig.text(.5, .998, "DEVELOPMENT SUBSET: NOT FOR PUBLICATION", va="top", ha="center",
                 color="#9a2626", fontsize=FONT_SIZE, fontweight="bold")
    return fig


def draw_figure(h):
    return draw_paired(h) if LAYOUT == "paired" else draw_stacked(h)


OUTPUTS = []


def save(fig, stem, text):
    for extension in ("pdf", "svg", "png"):
        path = FIGURE_DIR / f"{stem}.{extension}"
        fig.savefig(path, dpi=450)
        OUTPUTS.append(path)
    path = FIGURE_DIR / f"{stem}_caption.txt"
    path.write_text(text + "\n", encoding="utf-8")
    OUTPUTS.append(path)


for h in HORIZONS:
    fig = draw_figure(h)
    save(fig, f"Fig4_{h}m" + ("_stacked" if LAYOUT == "stacked" else "")
         + ("_symlog" if BEE_AXIS[h][0] == "symlog" and BEE_AXIS_SETTING == "symlog" else ""),
         caption(h))
    if h == HORIZONS[0]:
        plt.show()
    plt.close(fig)
print("Saved Figure 4 and captions to:", FIGURE_DIR)


# =============================================================================
# Notebook cell 12
# =============================================================================

def supplementary_caption(h):
    b = manifest["bootstrap"]
    parts = ["Supplementary Fig. | Robustness of the Figure 4 attributions. "]
    if SHARE is not None:
        parts.append("a, Mean share of the total absolute Shapley attribution per input group, averaged equally "
                     "across conditions whose total attribution was positive. ")
    if BRIER is not None:
        parts.append("b, Change in the fixed-model IPCW Brier score when a single input group was set to its "
                     "Figure 4 reference with all other inputs observed, averaged equally across eligible "
                     "conditions (×10⁻⁴). Positive values indicate that the observed input reduced prediction "
                     "error. ")
    if AGE_AUDIT is not None and AGE_TAILS is not None:
        audit = AGE_AUDIT.iloc[0]
        parts.append(
            f"c, Setting landmark age to the training median moved "
            f"{int(audit.diagnosis_records_newly_below_18_n):,} of {int(audit.diagnosis_records_n):,} "
            f"pre-landmark diagnosis records ({100 * audit.fraction_records_newly_below_18:.2f}%) in "
            f"{int(audit.participants_with_newly_below_18_n):,} participants "
            f"({100 * audit.fraction_participants_with_newly_below_18:.2f}%) to before age 18. Points give the "
            f"median (circle) and 95th percentile (bar end) of the absolute age contribution for participants with and "
            f"without such a record. ")
    parts.append(f"Intervals are {100 * (1 - b['alpha']):g}% participant-bootstrap percentile intervals "
                 f"({b['n']} resamples), conditional on the fitted model.")
    return "".join(parts)


def draw_supplement(h):
    panels = [p for p, ok in (("share", SHARE is not None), ("brier", BRIER is not None),
                              ("age", AGE_AUDIT is not None and AGE_TAILS is not None)) if ok]
    if not panels:
        return None
    widths = {"share": 1, "brier": 1, "age": 1.25}
    fig = plt.figure(figsize=(FIGURE_WIDTH_IN, 3.2))
    grid = fig.add_gridspec(1, len(panels), left=.1, right=.98, bottom=.16, top=.78,
                            width_ratios=[widths[p] for p in panels], wspace=.42)
    labels = [GROUP_LABELS.get(g, g) for g in GROUP_ORDER]
    y = np.arange(len(GROUP_ORDER))
    for i, panel in enumerate(panels):
        ax = fig.add_subplot(grid[0, i])
        letter = "abc"[i]
        if panel == "share":
            d = SHARE.loc[SHARE.horizon_months.eq(h)].set_index("group").reindex(GROUP_ORDER)
            ax.barh(y, d.mean_attribution_share_pct, height=.56, color=[GROUP_COLOURS[g] for g in GROUP_ORDER],
                    linewidth=0)
            ax.hlines(y, d.ci_low_pct, d.ci_high_pct, color=INK, linewidth=.7)
            for row, value, end in zip(y, d.mean_attribution_share_pct, np.fmax(d.ci_high_pct, d.mean_attribution_share_pct)):
                ax.text(end + 1, row, f"{value:.0f}%", va="center", fontsize=FONT_SIZE - .5, color=INK)
            ax.set_xlim(0, float(np.nanmax(d.ci_high_pct)) * 1.25)
            ax.set_yticks(y, labels)
            ax.set_xlabel("Mean share of total attribution (%)")
            title = "Attribution share"
        elif panel == "brier":
            d = BRIER.loc[BRIER.horizon_months.eq(h)].set_index("group").reindex(GROUP_ORDER)
            change, low, high = (d[c].to_numpy() * 1e4 for c in ("change_ipcw_brier", "change_ci_low", "change_ci_high"))
            colours = [POSITIVE if v > 0 else NEGATIVE for v in change]
            ax.axvline(0, color=RULE, linewidth=.6)
            ax.hlines(y, low, high, color=INK, linewidth=.7, zorder=2)
            ax.scatter(change, y, s=20, color=colours, edgecolors="white", linewidths=.5, zorder=3)
            reach = float(np.nanmax(np.abs(np.r_[low, high]))) * 1.15 or 1
            ax.set_xlim(-reach, reach)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: minus(f"{v:.2g}")))
            ax.set_yticks(y, labels if i == 0 else [])
            ax.set_xlabel(r"Δ IPCW Brier when set to reference ($\times10^{-4}$)")
            ax.text(.98, 1.0, "observed input helps →", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=SMALLEST, color=MUTED)
            title = "Prediction error"
        else:
            t = AGE_TAILS.loc[AGE_TAILS.horizon_months.eq(h) & AGE_TAILS.condition_id.isin(DISPLAY_CONDITIONS)]
            order = DISPLAY_CONDITIONS
            yc = np.arange(len(order))
            for stratum, colour, shift in (("none newly below 18", "#8a8894", -.17), ("newly below 18", "#c9584f", .17)):
                s = t.loc[t.stratum.eq(stratum)].set_index("condition_id").reindex(order)
                ax.hlines(yc + shift, s.median_abs_age_shapley_pp, s.p95_abs_age_shapley_pp, color=colour,
                          linewidth=.8)
                ax.scatter(s.median_abs_age_shapley_pp, yc + shift, s=9, color=colour, linewidths=0,
                           label="Diagnosis moved below 18" if stratum == "newly below 18" else "None moved")
            ax.set_yticks(yc, [CONDITION_LABELS[k] for k in order])
            ax.set_ylim(len(order) - .4, -.6)
            ax.set_xlabel("|Age contribution| (pp), median to p95")
            ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=SMALLEST + .5,
                      handletextpad=.2, columnspacing=.8, borderaxespad=.2)
            ax.set_xlim(left=0)
            title = "Landmark-age audit"
        if panel != "age":
            ax.set_ylim(len(GROUP_ORDER) - .45, -.6)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        trim(ax)
        ax.set_title(letter, loc="left", fontweight="bold", color=INK, fontsize=FONT_SIZE + 1.5,
                     pad=14, x=-.3 if i == 0 or panel == "age" else -.05)
    if DEV:
        fig.text(.5, .99, "DEVELOPMENT SUBSET: NOT FOR PUBLICATION", va="top", ha="center",
                 color="#9a2626", fontsize=FONT_SIZE, fontweight="bold")
    return fig


for h in HORIZONS:
    fig = draw_supplement(h)
    if fig is None:
        print("No supplementary inputs in this export; skipped.")
        continue
    save(fig, f"FigS_explainability_{h}m", supplementary_caption(h))
    if h == HORIZONS[0]:
        plt.show()
    plt.close(fig)


# =============================================================================
# Notebook cell 14
# =============================================================================

def plotted_tables():
    sel = lambda frame: frame.loc[frame.condition_id.isin(DISPLAY_CONDITIONS)]
    sheets = {
        "Fig4a": A.loc[A.scope.eq("all conditions")],
        "Fig4b": sel(B)[["horizon_months", "condition_id", "abbreviation", "condition_name", "group",
                         "eligible_n", "mean_predicted_risk_pct", "mean_increase_pp", "increase_ci_low_pp",
                         "increase_ci_high_pp", "mean_decrease_pp", "decrease_ci_low_pp", "decrease_ci_high_pp",
                         "mean_signed_shapley_pp", "positive_n", "negative_n", "zero_n"]],
        "Fig4c": sel(C)[["horizon_months", "domain", "level", "reference", "condition_id", "abbreviation",
                         "eligible_n", "mean_level_risk", "mean_reference_risk", "mean_diff_pp", "ci_low",
                         "ci_high", "risk_ratio", "ratio_ci_low", "ratio_ci_high"]],
        "Fig4d_points": D,
        "Fig4d_summary": D_SUMMARY,
    }
    if SHARE is not None:
        sheets["FigS_a_share"] = SHARE
    if BRIER is not None:
        sheets["FigS_b_brier"] = BRIER
    if AGE_TAILS is not None:
        sheets["FigS_c_age_tails"] = sel(AGE_TAILS)
    if AGE_AUDIT is not None:
        sheets["FigS_c_age_audit"] = AGE_AUDIT
    for name, frame in sheets.items():
        if "horizon_months" in frame and frame.horizon_months.isin(HORIZONS).all() is False:
            raise ValueError(f"{name}: unexpected horizon.")
    return sheets


provenance = pd.DataFrame([
    ("export_dir", str(EXPORT_DIR)), ("export_manifest_sha256", MANIFEST_SHA256),
    ("export_created_utc", manifest.get("created_utc")),
    ("checkpoint_sha256", manifest.get("source", {}).get("checkpoint_sha256")),
    ("n_test_participants", manifest["n_test_participants"]), ("development_subset", DEV),
    ("bootstrap", json.dumps(manifest["bootstrap"])), ("references", json.dumps(manifest["references"], ensure_ascii=False)),
    ("layout", LAYOUT), ("beeswarm_axis", json.dumps({str(h): v for h, v in BEE_AXIS.items()})),
    ("plotted_utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
], columns=["field", "value"])

sheets = plotted_tables()
source_path = FIGURE_DIR / "Fig4_source_data.xlsx"
try:
    with pd.ExcelWriter(source_path, engine="openpyxl") as writer:
        provenance.to_excel(writer, sheet_name="Provenance", index=False)
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
    OUTPUTS.append(source_path)
except ImportError:
    folder = FIGURE_DIR / "Fig4_source_data"
    folder.mkdir(exist_ok=True)
    for name, frame in {"Provenance": provenance, **sheets}.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
        OUTPUTS.append(folder / f"{name}.csv")
    print("openpyxl is not installed; wrote source data as CSV files in", folder)

figure_manifest = {
    "schema": "crescent.fig4.figures.v4",
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "export_dir": str(EXPORT_DIR), "export_manifest_sha256": MANIFEST_SHA256,
    "settings": {"beeswarm_axis": {str(h): {"scale": s, "reason": r} for h, (s, r) in BEE_AXIS.items()},
                 "symlog_linear_threshold_pp": {str(h): v for h, v in BEE_LINTHRESH.items()},
                 "asinh_linear_width_pp": {str(h): v for h, v in BEE_ASINH_WIDTH.items()},
                 "beeswarm_max_per_bin": BEE_MAX_PER_BIN, "layout": LAYOUT, "panel_c_values": C_VALUES, "value_ramp": {"name": VALUE_RAMP_NAME, "stops": VALUE_RAMP_STOPS}},
    "outputs": {str(p.relative_to(FIGURE_DIR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUTPUTS},
}
(FIGURE_DIR / "figure_manifest.json").write_text(json.dumps(figure_manifest, indent=2), encoding="utf-8")
print(f"Wrote {len(OUTPUTS)} files and figure_manifest.json to {FIGURE_DIR}")
