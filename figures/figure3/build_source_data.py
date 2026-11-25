"""Fig. 3, step 1 (HPC only): build the aggregate figure tables from the locked CRESCENT run.

Needs the participant-level test predictions, so it runs only where the UK Biobank data
are held. It writes the tables into the run folder (as the notebook did) and then copies
the aggregate ones into `figures/figure3/source_data/`, which `plot.py` reads.

    python figures/figure3/build_source_data.py

Inputs are read from the repository's `runs/` folder (git-ignored; see figures/README.md):
`runs/CRESCENT_Final/` (written by run_crescent.py, including the stage 8 calibration exports
of `evaluation/export_figure3_inputs.py`) and `runs/delphi_2m/` for the Panel H comparator.
Set CRESCENT_RUNS_DIR to read them from another location.

Code from the notebook `Figure2_CRESCENT_publication` (cell 3 unchanged); only the paths
of cell 1 were replaced. The final block, which copies the published tables, was added for
the repository.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import *  # noqa: E402,F401,F403  (notebook cell 1 settings)

# =============================================================================
# Paths (replace the HPC paths of notebook cell 1)
# =============================================================================

REPO = HERE.parents[1]
RUNS = Path(os.environ.get("CRESCENT_RUNS_DIR", REPO / "runs")).expanduser().resolve()
CRESCENT_RUN_DIR = RUNS / "CRESCENT_Final"

CRESCENT_METRICS_PATH = (
    CRESCENT_RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
)
CRESCENT_PREDICTION_PATH = (
    CRESCENT_RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
)
CRESCENT_TEST_SAMPLES_PATH = (
    CRESCENT_RUN_DIR / "canonical_test_samples.pkl"
)
CRESCENT_FIG2_DIR = CRESCENT_RUN_DIR / "figure2_revised_inputs_2yr"
CRESCENT_FIG2_OUTPUT_DIR = (
    CRESCENT_RUN_DIR / "figure2_comparison_outputs"
)

FIG2_COMPARATOR_METRICS_PATH = (
    RUNS / "delphi_2m" / "test_per_condition_metrics_by_horizon.csv"
)
PROJECT_ROOT = CRESCENT_RUN_DIR

print("Figure 2 setup")
print("  project root:        ", PROJECT_ROOT)
print("  CRESCENT run:        ", CRESCENT_RUN_DIR)
print("  CRESCENT metrics:    ", CRESCENT_METRICS_PATH)
print("  prediction arrays:   ", CRESCENT_PREDICTION_PATH)
print("  canonical test data: ", CRESCENT_TEST_SAMPLES_PATH)
print("  comparator metrics:  ", FIG2_COMPARATOR_METRICS_PATH)
print("  Figure 2 inputs:     ", CRESCENT_FIG2_DIR)
print("  Figure 2 outputs:    ", CRESCENT_FIG2_OUTPUT_DIR)


# =============================================================================
# Notebook cell 3
# =============================================================================

# =============================================================================
# BUILD ALL CRESCENT FIGURE-2 INPUT TABLES FROM LOCKED OUTPUTS
#
# Model-dependent sources:
#   - test_prediction_arrays_death_adjusted.npz
#   - canonical_test_samples.pkl
#   - test_per_condition_metrics_by_horizon.csv
#   - calibration CSVs created by the new CRESCENT export cell
#
# Optional non-model source:
#   - static condition-to-ICD chapter mapping for labels/colours only
# =============================================================================

from pathlib import Path
import json
import shutil
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        roc_curve,
    )
except Exception as exc:
    raise ImportError(
        "scikit-learn is required to generate the Figure-2 AUROC, AUPRC, "
        "and ROC-curve tables."
    ) from exc


# =============================================================================
# Required locked CRESCENT artifacts
# =============================================================================

CRESCENT_RUN_DIR = Path(CRESCENT_RUN_DIR).expanduser().resolve()
CRESCENT_METRICS_PATH = Path(CRESCENT_METRICS_PATH).expanduser().resolve()
CRESCENT_PREDICTION_PATH = Path(CRESCENT_PREDICTION_PATH).expanduser().resolve()
CRESCENT_TEST_SAMPLES_PATH = Path(CRESCENT_TEST_SAMPLES_PATH).expanduser().resolve()
CRESCENT_FIG2_DIR = Path(CRESCENT_FIG2_DIR).expanduser().resolve()
CRESCENT_FIG2_OUTPUT_DIR = Path(CRESCENT_FIG2_OUTPUT_DIR).expanduser().resolve()
CRESCENT_FIG2_DIR.mkdir(parents=True, exist_ok=True)
CRESCENT_FIG2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if CRESCENT_METRICS_PATH.parent != CRESCENT_RUN_DIR:
    raise RuntimeError(
        "CRESCENT_METRICS_PATH is not inside CRESCENT_RUN_DIR: "
        f"{CRESCENT_METRICS_PATH}"
    )

required_locked_files = [
    CRESCENT_METRICS_PATH,
    CRESCENT_PREDICTION_PATH,
    CRESCENT_TEST_SAMPLES_PATH,
]
missing_locked_files = [str(path) for path in required_locked_files if not path.exists()]
if missing_locked_files:
    raise FileNotFoundError(
        "Run the revised CRESCENT notebook through locked evaluation before "
        "running this comparison notebook. Missing:\n  - "
        + "\n  - ".join(missing_locked_files)
    )

calibration_deciles_path = (
    CRESCENT_FIG2_DIR
    / "figure2_revised_panelH_calibration_deciles_2yr.csv"
)
calibration_summary_2yr_path = (
    CRESCENT_FIG2_DIR
    / "figure2_revised_panelH_calibration_summary_2yr.csv"
)
calibration_summary_all_candidates = [
    CRESCENT_RUN_DIR
    / "figure2_revised_panelH_calibration_summary_all_horizons.csv",
    CRESCENT_FIG2_DIR
    / "figure2_revised_panelH_calibration_summary_all_horizons.csv",
    CRESCENT_RUN_DIR
    / "test_calibration_error_by_horizon_condition_sex.csv",
    CRESCENT_FIG2_DIR
    / "test_calibration_error_by_horizon_condition_sex.csv",
]
calibration_summary_all_path = next(
    (path for path in calibration_summary_all_candidates if path.exists()),
    None,
)

missing_calibration = [
    str(path)
    for path in [calibration_deciles_path, calibration_summary_2yr_path]
    if not path.exists()
]
if calibration_summary_all_path is None:
    missing_calibration.extend(str(path) for path in calibration_summary_all_candidates)

if missing_calibration:
    raise FileNotFoundError(
        "The new CRESCENT Figure-2 calibration export cell has not produced "
        "all required files. Run that cell after Cell 7 in the model notebook. "
        "Missing/checked:\n  - " + "\n  - ".join(missing_calibration)
    )

for path in [
    calibration_deciles_path,
    calibration_summary_2yr_path,
    calibration_summary_all_path,
]:
    try:
        path.resolve().relative_to(CRESCENT_RUN_DIR)
    except ValueError as exc:
        raise RuntimeError(
            "A calibration input resolved outside CRESCENT_RUN_DIR: "
            f"{path}"
        ) from exc


# =============================================================================
# Load locked arrays and canonical cohort metadata
# =============================================================================

z = np.load(CRESCENT_PREDICTION_PATH, allow_pickle=True)
test_meta = pd.read_pickle(CRESCENT_TEST_SAMPLES_PATH).reset_index(drop=True)
condition_metrics = pd.read_csv(CRESCENT_METRICS_PATH)
H = pd.read_csv(calibration_deciles_path)
Hsummary = pd.read_csv(calibration_summary_2yr_path)
ALL_CAL = pd.read_csv(calibration_summary_all_path)

if "eid" not in test_meta.columns:
    raise KeyError("canonical_test_samples.pkl is missing the 'eid' column.")

required_npz = {
    "eid",
    "condition_names",
    "horizons_months",
    "gap_months",
    "index_age_years",
    "at_risk",
    "event_obs",
    "t_tilde",
    "C_months",
    "death_observed_post_index",
    "death_time_months",
}
missing_npz = sorted(required_npz - set(z.files))
if missing_npz:
    raise KeyError(
        "CRESCENT prediction NPZ is missing required arrays: "
        f"{missing_npz}"
    )

pred_eid = np.asarray(z["eid"], dtype=object).astype(str)
meta_eid = test_meta["eid"].astype(str).to_numpy()
if pred_eid.tolist() != meta_eid.tolist():
    raise AssertionError(
        "The prediction NPZ and canonical_test_samples.pkl have different "
        "participant ordering."
    )

condition_names = [str(x).strip() for x in z["condition_names"].tolist()]
horizons = [int(x) for x in z["horizons_months"].tolist()]
gap_months = float(np.asarray(z["gap_months"]).reshape(-1)[0])
N = int(len(pred_eid))
K = int(len(condition_names))

required_horizons = [int(x) for x in FIG2_HORIZONS_MONTHS]
missing_horizons = sorted(set(required_horizons) - set(horizons))
if missing_horizons:
    raise ValueError(
        "CRESCENT prediction arrays are missing required horizons: "
        f"{missing_horizons}"
    )

required_risk_keys = {
    f"risk_death_adjusted_h{h}" for h in required_horizons
}
missing_risk_keys = sorted(required_risk_keys - set(z.files))
if missing_risk_keys:
    raise KeyError(
        "CRESCENT prediction NPZ is missing required risk arrays: "
        f"{missing_risk_keys}"
    )

risk_by_horizon = {
    int(h): np.asarray(z[f"risk_death_adjusted_h{int(h)}"], dtype=np.float64)
    for h in required_horizons
}
for h, risk in risk_by_horizon.items():
    if risk.shape != (N, K):
        raise ValueError(
            f"risk_death_adjusted_h{h} has shape {risk.shape}; expected {(N, K)}."
        )
    if not np.isfinite(risk).all() or np.any((risk < -1e-7) | (risk > 1.0 + 1e-7)):
        raise ValueError(f"Invalid CRESCENT risk values at horizon {h} months.")

at_risk = np.asarray(z["at_risk"], dtype=np.float64) > 0.5
event_obs = np.asarray(z["event_obs"], dtype=np.float64) > 0.5
t_tilde = np.asarray(z["t_tilde"], dtype=np.float64)
C_months = np.asarray(z["C_months"], dtype=np.float64)
death_post = np.asarray(z["death_observed_post_index"], dtype=np.float64) > 0.5
death_time = np.asarray(z["death_time_months"], dtype=np.float64)
index_age_years = np.asarray(z["index_age_years"], dtype=np.float64)

if (
    at_risk.shape != (N, K)
    or event_obs.shape != (N, K)
    or t_tilde.shape != (N, K)
):
    raise ValueError("CRESCENT disease target arrays have inconsistent shapes.")
for array_name, array in {
    "C_months": C_months,
    "death_observed_post_index": death_post,
    "death_time_months": death_time,
    "index_age_years": index_age_years,
}.items():
    if array.shape != (N,):
        raise ValueError(
            f"{array_name} has shape {array.shape}; expected {(N,)}."
        )


# =============================================================================
# Sex labels
# =============================================================================

def _normalise_text_sex(values):
    values = pd.Series(values).astype(str).str.strip().str.lower()
    mapping = {
        "m": "Male", "male": "Male", "man": "Male", "men": "Male",
        "f": "Female", "female": "Female", "woman": "Female", "women": "Female",
    }
    return values.map(mapping).fillna("Unknown").to_numpy(dtype=object)


def _resolve_sex(meta):
    for column in ["sex", "sex_label", "gender", "gender_label"]:
        if column in meta.columns:
            labels = _normalise_text_sex(meta[column])
            if np.isin(labels, ["Male", "Female"]).any():
                return labels, f"text_column:{column}"

    if "gender_idx" not in meta.columns:
        raise KeyError("canonical_test_samples.pkl contains no sex/gender field.")

    numeric = pd.to_numeric(meta["gender_idx"], errors="coerce")
    labels = (
        numeric.map(GENDER_INDEX_TO_SEX)
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=object)
    )
    return labels, "gender_idx_mapping"


sex, sex_source = _resolve_sex(test_meta)
sex_counts = pd.Series(sex).value_counts(dropna=False).to_dict()
if int(np.sum(sex == "Male")) == 0 or int(np.sum(sex == "Female")) == 0:
    raise ValueError(
        "Both Male and Female strata are required. Verify GENDER_INDEX_TO_SEX."
    )


# =============================================================================
# Observed competing-risk arrays and metrics
# =============================================================================

TIME_EPS = 1e-8


def make_observed_arrays(horizon_months):
    """Locked fixed-horizon labels using Cell-7 event_obs tie resolution."""
    h = float(horizon_months)
    endpoint = float(gap_months) + h
    C_mat = C_months[:, None]
    death_mat = death_time[:, None]
    death_obs_mat = death_post[:, None]

    condition_before_gap = (
        event_obs & np.isfinite(t_tilde) & (t_tilde <= gap_months + TIME_EPS)
    )
    death_before_gap = (
        death_obs_mat
        & np.isfinite(death_mat)
        & (death_mat <= gap_months + TIME_EPS)
    )

    eligible = (
        at_risk
        & (C_mat >= gap_months - TIME_EPS)
        & (~condition_before_gap)
        & (~death_before_gap)
    )

    condition_candidate = (
        eligible
        & event_obs
        & np.isfinite(t_tilde)
        & (t_tilde > gap_months + TIME_EPS)
        & (t_tilde <= endpoint + TIME_EPS)
    )

    death_candidate = (
        eligible
        & death_obs_mat
        & np.isfinite(death_mat)
        & (death_mat > gap_months + TIME_EPS)
        & (death_mat <= endpoint + TIME_EPS)
    )

    # Respect event chronology explicitly. A condition/death tie is assigned
    # to the condition, matching the condition-first tie rule used upstream.
    condition_event = condition_candidate & (
        (~death_candidate) | (t_tilde <= death_mat + TIME_EPS)
    )
    death_competing = death_candidate & (~condition_event)

    observed_through = eligible & (C_mat >= endpoint - TIME_EPS)
    known = condition_event | death_competing | observed_through

    status = np.zeros((N, K), dtype=np.int16)
    status[condition_event] = 1
    status[death_competing] = 2

    time_rel = np.full((N, K), h, dtype=np.float64)
    time_rel[condition_event] = t_tilde[condition_event] - gap_months
    if death_competing.any():
        death_rel = np.repeat(death_mat - gap_months, K, axis=1)
        time_rel[death_competing] = death_rel[death_competing]

    censor_event = eligible & (~condition_event) & (~death_competing) & (~observed_through)
    if censor_event.any():
        censor_rel = np.repeat(C_mat - gap_months, K, axis=1)
        time_rel[censor_event] = censor_rel[censor_event]

    time_rel = np.clip(time_rel, 0.0, h)

    return {
        "eligible": eligible,
        "known": known,
        "y": condition_event.astype(np.int8),
        "status": status,
        "time_rel": time_rel,
        "condition_event": condition_event,
        "death_competing": death_competing,
    }


def aalen_johansen_cif(time, status, horizon_months):
    time = np.asarray(time, dtype=np.float64)
    status = np.asarray(status, dtype=np.int16)
    h = float(horizon_months)
    valid = np.isfinite(time) & (time >= 0.0)
    time = time[valid]
    status = status[valid]
    if time.size == 0:
        return np.nan

    status_h = np.where(time <= h + TIME_EPS, status, 0)
    time_h = np.minimum(time, h)
    event_times = np.unique(time_h[(status_h == 1) | (status_h == 2)])

    survival = 1.0
    cif = 0.0
    for event_time in np.sort(event_times):
        n_risk = int(np.sum(time_h >= event_time - TIME_EPS))
        if n_risk <= 0:
            continue
        d_condition = int(
            np.sum((np.abs(time_h - event_time) <= TIME_EPS) & (status_h == 1))
        )
        d_death = int(
            np.sum((np.abs(time_h - event_time) <= TIME_EPS) & (status_h == 2))
        )
        cif += survival * d_condition / n_risk
        survival *= max(0.0, 1.0 - (d_condition + d_death) / n_risk)

    return float(cif)


def binary_metrics(y, risk, known):
    y = np.asarray(y, dtype=np.int8)
    risk = np.asarray(risk, dtype=np.float64)
    known = np.asarray(known, dtype=bool)
    valid = known & np.isfinite(risk)
    yv = y[valid]
    rv = risk[valid]

    out = {"auroc": np.nan, "auprc": np.nan}
    if yv.size >= 2 and len(np.unique(yv)) == 2:
        out["auroc"] = float(roc_auc_score(yv, rv))
        out["auprc"] = float(average_precision_score(yv, rv))
    return out


observed_by_horizon = {
    int(h): make_observed_arrays(int(h))
    for h in required_horizons
}


# =============================================================================
# Static condition metadata / colour key
# =============================================================================

condition_base = pd.DataFrame(
    {
        "condition_id": np.arange(K, dtype=int),
        "condition_name": condition_names,
    }
)

# These names and clinical chapter assignments came from the supplied
# Con_chapters__V3.csv. Match all 60 conditions to the LOCKED model's names,
# retaining the exact model order. Never manufacture an ICD chapter from a
# similarly worded disease label or an unrelated run's colour key.
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




color_key = _validated_color_key(
    pd.DataFrame(
        FIG2_CHAPTER_CROSSWALK,
        columns=["condition_name", "ICD_Chapter", "Chapter_Name"],
    ),
    FIG2_CHAPTER_CROSSWALK_PROVENANCE,
)
color_key["hex_color"] = color_key["ICD_Chapter"].map(
    lambda chapter: ICD_CHAPTER_COLORS[_chapter_number(chapter)]
)
color_key_path = CRESCENT_FIG2_DIR / "figure2_revised_condition_color_key.csv"
color_key.to_csv(color_key_path, index=False)
print("Verified ICD chapter mapping:", FIG2_CHAPTER_CROSSWALK_PROVENANCE)
print("  chapters:", color_key["ICD_Chapter"].nunique(),
      "conditions:", len(color_key))

chapter_lookup = color_key.set_index("condition_name")[
    ["ICD_Chapter", "Chapter_Name"]
].to_dict("index")


# =============================================================================
# Per-condition sex-stratified AUROC/AUPRC tables
# =============================================================================

strata = {
    "All": np.ones(N, dtype=bool),
    "Male": sex == "Male",
    "Female": sex == "Female",
}

obs_2yr = observed_by_horizon[int(FIG2_PRIMARY_HORIZON_MONTHS)]
event_n_2yr_all = obs_2yr["condition_event"].sum(axis=0).astype(int)

performance_rows = []
for horizon_months in required_horizons:
    obs = observed_by_horizon[int(horizon_months)]
    risk = risk_by_horizon[int(horizon_months)]

    for condition_id, condition_name in enumerate(condition_names):
        meta = chapter_lookup[condition_name]

        for sex_label, participant_mask in strata.items():
            participant_mask = np.asarray(participant_mask, dtype=bool)
            eligible = participant_mask & obs["eligible"][:, condition_id]
            known = participant_mask & obs["known"][:, condition_id]
            cases = participant_mask & obs["condition_event"][:, condition_id]
            deaths = participant_mask & obs["death_competing"][:, condition_id]
            metrics = binary_metrics(
                obs["y"][:, condition_id],
                risk[:, condition_id],
                known,
            )

            performance_rows.append(
                {
                    "Model": FIG2_TARGET_MODEL,
                    "model": FIG2_TARGET_MODEL,
                    "endpoint_type": "condition",
                    "risk_type": "death_adjusted",
                    "condition_id": int(condition_id),
                    "condition_name": condition_name,
                    "horizon_months": int(horizon_months),
                    "sex": sex_label,
                    "eligible_n": int(eligible.sum()),
                    "known_n": int(known.sum()),
                    "event_n": int(cases.sum()),
                    "death_competing_n": int(deaths.sum()),
                    "control_n": int(max(int(known.sum()) - int(cases.sum()), 0)),
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "frequency_for_plot": int(event_n_2yr_all[condition_id]),
                    "event_n_2yr_all": int(event_n_2yr_all[condition_id]),
                    "ICD_Chapter": meta["ICD_Chapter"],
                    "Chapter_Name": meta["Chapter_Name"],
                }
            )

performance = pd.DataFrame(performance_rows)

B = performance[
    performance["horizon_months"].eq(int(FIG2_PRIMARY_HORIZON_MONTHS))
].copy()
B_path = CRESCENT_FIG2_DIR / "figure2_revised_panelB_frequency_sex_2yr.csv"
B.to_csv(B_path, index=False)

C = B[B["sex"].isin(["Male", "Female"])].copy()
C_path = CRESCENT_FIG2_DIR / "figure2_revised_panelC_chapter_sex_auroc_2yr.csv"
C.to_csv(C_path, index=False)

pair = (
    C.pivot_table(
        index=[
            "condition_id",
            "condition_name",
            "ICD_Chapter",
            "Chapter_Name",
            "frequency_for_plot",
        ],
        columns="sex",
        values="auroc",
        aggfunc="first",
    )
    .reset_index()
)
for column in ["Male", "Female"]:
    if column not in pair.columns:
        pair[column] = np.nan
pair["auroc_female_minus_male"] = pair["Female"] - pair["Male"]
Cpair = pair
Cpair_path = (
    CRESCENT_FIG2_DIR
    / "figure2_revised_panelC_chapter_sex_pair_deltas_2yr.csv"
)
Cpair.to_csv(Cpair_path, index=False)

E = performance.copy()
E_path = CRESCENT_FIG2_DIR / "figure2_revised_panelE_horizon_sex_auroc.csv"
E.to_csv(E_path, index=False)


# =============================================================================
# Selected endpoints: fixed prespecified condition list
# =============================================================================

calibrated_2yr_all = H[
    pd.to_numeric(H["horizon_months"], errors="coerce").eq(
        int(FIG2_PRIMARY_HORIZON_MONTHS)
    )
    & H["calibration_stratum"].astype(str).eq("All")
    & pd.to_numeric(H["predicted_mean"], errors="coerce").notna()
    & pd.to_numeric(H["observed_aj_cif"], errors="coerce").notna()
]
calibration_conditions = set(calibrated_2yr_all["condition_name"].astype(str).str.strip())

duplicate_condition_names = condition_base[
    condition_base["condition_name"].duplicated(keep=False)
]["condition_name"].unique().tolist()
if duplicate_condition_names:
    raise RuntimeError(
        "The CRESCENT condition axis contains duplicate names: "
        + ", ".join(duplicate_condition_names[:10])
    )

condition_to_id = condition_base.set_index("condition_name")["condition_id"].to_dict()
missing_selected = [
    condition
    for condition in FIG2_SELECTED_CONDITIONS
    if condition not in condition_to_id
]
if missing_selected:
    raise RuntimeError(
        "The following prespecified Figure 2 conditions are absent from the "
        "CRESCENT prediction arrays:\n  - "
        + "\n  - ".join(missing_selected)
    )

selection = pd.DataFrame(
    {
        "selection_rank": np.arange(
            1, len(FIG2_SELECTED_CONDITIONS) + 1, dtype=int
        ),
        "condition_name": list(FIG2_SELECTED_CONDITIONS),
    }
)
selection["condition_id"] = selection["condition_name"].map(condition_to_id).astype(int)
selection["event_n_2yr_all"] = selection["condition_id"].map(
    lambda condition_id: int(event_n_2yr_all[int(condition_id)])
)
selection["has_2yr_calibration"] = selection["condition_name"].isin(
    calibration_conditions
)

zero_event_conditions = selection.loc[
    selection["event_n_2yr_all"].le(0), "condition_name"
].tolist()
missing_calibration_conditions = selection.loc[
    ~selection["has_2yr_calibration"], "condition_name"
].tolist()
selection_failures = []
if zero_event_conditions:
    selection_failures.append(
        "no observed 2-year events: " + ", ".join(zero_event_conditions)
    )
if missing_calibration_conditions:
    selection_failures.append(
        "no finite 2-year all-participant calibration rows: "
        + ", ".join(missing_calibration_conditions)
    )
if selection_failures:
    raise RuntimeError(
        "The prespecified Figure 2 condition set is incomplete:\n  - "
        + "\n  - ".join(selection_failures)
    )

selected_path = CRESCENT_FIG2_DIR / "figure2_revised_selected_endpoints_panelA.csv"
selection.to_csv(selected_path, index=False)
selected_ids = selection["condition_id"].astype(int).tolist()


# =============================================================================
# Panel A: age- and sex-stratified predicted risk and observed AJ CIF
# =============================================================================

finite_age = index_age_years[np.isfinite(index_age_years)]
if finite_age.size == 0:
    raise ValueError("No finite index ages are available for Panel A.")

width = float(FIG2_AGE_BIN_WIDTH_YEARS)
age_start = width * np.floor(float(finite_age.min()) / width)
age_stop = width * np.ceil(float(finite_age.max()) / width) + width
age_edges = np.arange(age_start, age_stop + 1e-9, width, dtype=float)
if age_edges.size < 2:
    age_edges = np.asarray([age_start, age_start + width], dtype=float)
age_bin_id = np.digitize(index_age_years, age_edges[1:-1], right=False)

A_rows = []
risk_2yr = risk_by_horizon[int(FIG2_PRIMARY_HORIZON_MONTHS)]
for condition_id in selected_ids:
    condition_name = condition_names[condition_id]
    for sex_label in ["Male", "Female"]:
        sex_mask = sex == sex_label
        for bin_number in range(len(age_edges) - 1):
            lower = float(age_edges[bin_number])
            upper = float(age_edges[bin_number + 1])
            participant_mask = (
                sex_mask
                & np.isfinite(index_age_years)
                & (age_bin_id == bin_number)
            )
            eligible = participant_mask & obs_2yr["eligible"][:, condition_id]
            eligible_n = int(eligible.sum())
            if eligible_n < int(FIG2_MIN_AGE_BIN_ELIGIBLE):
                continue

            known = participant_mask & obs_2yr["known"][:, condition_id]
            cases = participant_mask & obs_2yr["condition_event"][:, condition_id]
            observed = aalen_johansen_cif(
                obs_2yr["time_rel"][eligible, condition_id],
                obs_2yr["status"][eligible, condition_id],
                int(FIG2_PRIMARY_HORIZON_MONTHS),
            )

            A_rows.append(
                {
                    "Model": FIG2_TARGET_MODEL,
                    "endpoint_type": "condition",
                    "risk_type": "death_adjusted",
                    "condition_id": int(condition_id),
                    "condition_name": condition_name,
                    "horizon_months": int(FIG2_PRIMARY_HORIZON_MONTHS),
                    "sex": sex_label,
                    "age_bin_lower": lower,
                    "age_bin_upper": upper,
                    "age_bin_mid": 0.5 * (lower + upper),
                    "eligible_n": eligible_n,
                    "known_n": int(known.sum()),
                    "event_n": int(cases.sum()),
                    "mean_predicted_risk": float(
                        np.mean(risk_2yr[eligible, condition_id])
                    ),
                    "observed_aj_cif": observed,
                }
            )

A = pd.DataFrame(A_rows)
A_path = CRESCENT_FIG2_DIR / "figure2_revised_panelA_age_sex_rates_2yr.csv"
A.to_csv(A_path, index=False)


# =============================================================================
# Panel F: 2-year ROC curves for the selected CRESCENT endpoints
# =============================================================================

F_rows = []
for condition_id in selected_ids:
    condition_name = condition_names[condition_id]
    known = obs_2yr["known"][:, condition_id]
    y = obs_2yr["y"][:, condition_id][known].astype(int)
    score = risk_2yr[:, condition_id][known].astype(float)

    if y.size < 2 or len(np.unique(y)) < 2:
        continue

    fpr, tpr, threshold = roc_curve(y, score, drop_intermediate=False)
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))

    for fpr_value, tpr_value, threshold_value in zip(fpr, tpr, threshold):
        F_rows.append(
            {
                "Model": FIG2_TARGET_MODEL,
                "endpoint_type": "condition",
                "risk_type": "death_adjusted",
                "condition_id": int(condition_id),
                "condition_name": condition_name,
                "horizon_months": int(FIG2_PRIMARY_HORIZON_MONTHS),
                "sex": "All",
                "fpr": float(fpr_value),
                "tpr": float(tpr_value),
                "threshold": float(threshold_value),
                "auroc": auroc,
                "auprc": auprc,
                "known_n": int(known.sum()),
                "event_n": int(y.sum()),
            }
        )

F = pd.DataFrame(F_rows)
F_path = CRESCENT_FIG2_DIR / "figure2_revised_panelF_roc_curves_2yr.csv"
F.to_csv(F_path, index=False)


# =============================================================================
# Ensure the canonical calibration aliases are present in the CRESCENT folder
# =============================================================================

canonical_all_cal_path = (
    CRESCENT_FIG2_DIR
    / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
)
if calibration_summary_all_path.resolve() != canonical_all_cal_path.resolve():
    shutil.copy2(calibration_summary_all_path, canonical_all_cal_path)

# Validate that the 2-year summary is the same source family as ALL_CAL.
required_calibration_columns = {
    "condition_id", "condition_name", "horizon_months",
    "calibration_stratum", "sex", "ici", "e50", "e90", "total_event_n",
}
missing_cal_columns = sorted(required_calibration_columns - set(ALL_CAL.columns))
if missing_cal_columns:
    raise ValueError(
        "The new all-horizon calibration export is missing columns: "
        f"{missing_cal_columns}"
    )

available_cal_horizons = sorted(
    pd.to_numeric(ALL_CAL["horizon_months"], errors="coerce")
    .dropna().astype(int).unique().tolist()
)
if sorted(required_horizons) != available_cal_horizons:
    raise ValueError(
        "All-horizon CRESCENT calibration file has unexpected horizons. "
        f"Found {available_cal_horizons}; expected {sorted(required_horizons)}."
    )

available_strata = set(ALL_CAL["calibration_stratum"].astype(str))
if not {"All", "Male", "Female"}.issubset(available_strata):
    raise ValueError(
        "CRESCENT calibration exports do not contain All, Male, and Female strata."
    )


# =============================================================================
# Source manifest and hard output checks
# =============================================================================

output_files = {
    "A": A_path,
    "B": B_path,
    "C": C_path,
    "Cpair": Cpair_path,
    "E": E_path,
    "F": F_path,
    "H": calibration_deciles_path,
    "Hsummary": calibration_summary_2yr_path,
    "Hsummary_all": canonical_all_cal_path,
    "colors": color_key_path,
    "selected": selected_path,
}

missing_outputs = [str(path) for path in output_files.values() if not path.exists()]
if missing_outputs:
    raise FileNotFoundError(
        "Figure-2 preparation did not create all required files:\n  - "
        + "\n  - ".join(missing_outputs)
    )

manifest = {
    "model": FIG2_TARGET_MODEL,
    "comparator_model": FIG2_COMPARATOR_MODEL,
    "run_dir": str(CRESCENT_RUN_DIR),
    "metrics_path": str(CRESCENT_METRICS_PATH),
    "comparator_metrics_path": str(FIG2_COMPARATOR_METRICS_PATH),
    "prediction_path": str(CRESCENT_PREDICTION_PATH),
    "canonical_test_samples_path": str(CRESCENT_TEST_SAMPLES_PATH),
    "figure2_input_dir": str(CRESCENT_FIG2_DIR),
    "calibration_deciles_source": str(calibration_deciles_path),
    "calibration_summary_2yr_source": str(calibration_summary_2yr_path),
    "calibration_summary_all_source": str(calibration_summary_all_path),
    "color_key_source": FIG2_CHAPTER_CROSSWALK_PROVENANCE,
    "color_key_is_static_metadata_only": True,
    "sex_source": sex_source,
    "gender_index_to_sex": {str(k): str(v) for k, v in GENDER_INDEX_TO_SEX.items()},
    "sex_counts": {str(k): int(v) for k, v in sex_counts.items()},
    "n_test_participants": N,
    "n_conditions": K,
    "horizons_months": required_horizons,
    "primary_horizon_months": int(FIG2_PRIMARY_HORIZON_MONTHS),
    "gap_months": gap_months,
    "condition_selection_method": "fixed_prespecified_order",
    "selected_condition_ids": selected_ids,
    "selected_condition_names": selection["condition_name"].tolist(),
    "outputs": {key: str(path) for key, path in output_files.items()},
    "model_dependent_sources_all_within_crescent_run": True,
}
manifest_path = CRESCENT_FIG2_DIR / "figure2_crescent_source_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print("\nCRESCENT Figure-2 input build complete")
print("  participants:", N)
print("  conditions:  ", K)
print("  sex counts:  ", sex_counts)
print("  selected:    ", selection[["selection_rank", "condition_name", "event_n_2yr_all"]].to_dict("records"))
print("  input folder:", CRESCENT_FIG2_DIR)
print("  manifest:    ", manifest_path)
for key, path in output_files.items():
    print(f"  {key:12s}: {path}")


# =============================================================================
# Repository addition: copy the aggregate tables into figures/figure3/source_data/
# =============================================================================
# Copies only aggregate tables. Machine paths in the two JSON files are reduced to
# file names, and the ROC table loses its `threshold` column (one predicted risk per
# step of the curve, which plot.py does not use). The script stops if any table has
# an `eid` column and lists counts between 1 and 9 for review before committing.

import shutil

FIG3_SOURCE_DATA = HERE / "source_data"
FIG3_SOURCE_INPUTS = FIG3_SOURCE_DATA / CRESCENT_FIG2_DIR.name
FIG3_SOURCE_INPUTS.mkdir(parents=True, exist_ok=True)


def _strip_paths(value):
    if isinstance(value, dict):
        return {key: _strip_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_strip_paths(item) for item in value]
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return Path(value.replace("\\", "/")).name
    return value


def _check_published_csv(path):
    frame = pd.read_csv(path)
    if any(str(column).strip().lower() == "eid" for column in frame.columns):
        raise RuntimeError(f"{path} has an eid column; it must not be published.")
    notes = []
    for column in frame.columns:
        name = str(column).lower()
        if name.endswith("_n") or name in {"n", "n_pos"}:
            values = pd.to_numeric(frame[column], errors="coerce")
            small = int(values.between(1, 9).sum())
            if small:
                notes.append(f"{path.relative_to(HERE)}  {column}: {small} value(s) between 1 and 9")
    return notes


published = []
for key, path in output_files.items():
    target = FIG3_SOURCE_INPUTS / Path(path).name
    if key == "F":
        roc = pd.read_csv(path)
        roc.drop(columns=["threshold"], errors="ignore").to_csv(target, index=False)
    else:
        shutil.copy2(path, target)
    published.append(target)

target = FIG3_SOURCE_DATA / CRESCENT_METRICS_PATH.name
shutil.copy2(CRESCENT_METRICS_PATH, target)
published.append(target)

comparator_name = f"comparator_{FIG2_COMPARATOR_MODEL}_{FIG2_COMPARATOR_METRICS_PATH.name}"
target = FIG3_SOURCE_DATA / comparator_name
shutil.copy2(FIG2_COMPARATOR_METRICS_PATH, target)
published.append(target)

public_manifest = _strip_paths(manifest)
public_manifest["run_dir"] = "."
public_manifest["comparator_metrics_path"] = comparator_name
with open(FIG3_SOURCE_INPUTS / manifest_path.name, "w") as f:
    json.dump(public_manifest, f, indent=2, default=str)

calibration_metadata_source = CRESCENT_RUN_DIR / "figure2_calibration_export_metadata.json"
if calibration_metadata_source.is_file():
    with open(calibration_metadata_source) as f:
        public_calibration_metadata = _strip_paths(json.load(f))
    public_calibration_metadata["run_dir"] = "."
    with open(FIG3_SOURCE_DATA / calibration_metadata_source.name, "w") as f:
        json.dump(public_calibration_metadata, f, indent=2, default=str)

small_count_notes = []
for path in published:
    if path.suffix == ".csv":
        small_count_notes.extend(_check_published_csv(path))

print("\nPublished aggregate tables in", FIG3_SOURCE_DATA)
for path in sorted(FIG3_SOURCE_DATA.rglob("*")):
    if path.is_file():
        print("  ", path.relative_to(HERE))
if small_count_notes:
    print("\nSmall counts to review before committing:")
    print("\n".join("  " + line for line in small_count_notes))
else:
    print("No counts between 1 and 9.")
