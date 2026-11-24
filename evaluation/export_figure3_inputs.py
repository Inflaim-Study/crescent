"""Stage 8: export CRESCENT inputs for the calibration and performance figure.

The notebook calls this 'Figure 2'; it is Fig. 3 in the manuscript.

Extracted from the CRESCENT_Final notebook (cells 24, 26). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 24
# =============================================================================

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

assert "RUN_DIR" in globals(), "RUN_DIR is missing. Run the model notebook first."

RUN_DIR = Path(RUN_DIR).expanduser().resolve()
FIG2_EXPORT_MODEL_NAME = "CRESCENT"
FIG2_RISK_TYPE = "death_adjusted"
FIG2_HORIZONS_MONTHS = (12, 24, 60, 120)
FIG2_PRIMARY_HORIZON_MONTHS = 24
FIG2_TIME_EPS = 1e-8

_cfg_for_export = globals().get("cfg", None)
FIG2_N_CALIBRATION_BINS = int(
    getattr(_cfg_for_export, "EVAL_N_CALIBRATION_BINS", 10)
)
FIG2_MIN_CALIBRATION_GROUP_N = int(
    getattr(_cfg_for_export, "EVAL_MIN_CALIBRATION_GROUP_N", 100)
)
FIG2_MIN_CALIBRATION_BIN_N = int(
    getattr(_cfg_for_export, "EVAL_MIN_CALIBRATION_BIN_N", 50)
)

if FIG2_N_CALIBRATION_BINS < 2:
    raise ValueError("FIG2_N_CALIBRATION_BINS must be at least 2.")

PREDICTION_PATH = RUN_DIR / "test_prediction_arrays_death_adjusted.npz"
TEST_SAMPLES_PATH = RUN_DIR / "canonical_test_samples.pkl"
PER_CONDITION_METRICS_PATH = (
    RUN_DIR / "test_per_condition_metrics_by_horizon.csv"
)
FIG2_INPUT_DIR = RUN_DIR / "figure2_revised_inputs_2yr"
FIG2_INPUT_DIR.mkdir(parents=True, exist_ok=True)

required_locked_paths = [
    PREDICTION_PATH,
    TEST_SAMPLES_PATH,
    PER_CONDITION_METRICS_PATH,
]
missing_locked_paths = [
    str(path) for path in required_locked_paths if not path.exists()
]
if missing_locked_paths:
    raise FileNotFoundError(
        "Cell 7/canonical export has not produced all required locked files:\n  - "
        + "\n  - ".join(missing_locked_paths)
    )


# =============================================================================
# Helpers
# =============================================================================

def _sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_text_sex(values):
    values = pd.Series(values).astype(str).str.strip().str.lower()
    mapping = {
        "m": "Male",
        "male": "Male",
        "man": "Male",
        "men": "Male",
        "f": "Female",
        "female": "Female",
        "woman": "Female",
        "women": "Female",
    }
    return values.map(mapping).fillna("Unknown").to_numpy(dtype=object)


def _resolve_sex(meta):
    for column in ["sex", "sex_label", "gender", "gender_label"]:
        if column in meta.columns:
            labels = _normalise_text_sex(meta[column])
            if np.isin(labels, ["Male", "Female"]).any():
                return labels, f"text_column:{column}"

    if "gender_idx" not in meta.columns:
        raise KeyError(
            "canonical_test_samples.pkl contains no sex/gender field."
        )

    # Figure-2 plotting contract:
    #   0 = Female, 1 = Male, 2 = Unknown.
    mapping = {0: "Female", 1: "Male", 2: "Unknown"}
    numeric = pd.to_numeric(meta["gender_idx"], errors="coerce")
    labels = (
        numeric.map(mapping)
        .fillna("Unknown")
        .astype(str)
        .to_numpy(dtype=object)
    )
    return labels, "gender_idx_mapping_0_female_1_male_2_unknown"


def _aalen_johansen_cif(time, status, horizon_months):
    """
    Aalen–Johansen cumulative incidence for cause 1.

    status:
      0 = no event / administrative censoring
      1 = condition event
      2 = death competing event
    """
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    status = np.asarray(status, dtype=np.int16).reshape(-1)
    horizon = float(horizon_months)

    valid = np.isfinite(time) & (time >= 0.0)
    time = time[valid]
    status = status[valid]

    if time.size == 0:
        return np.nan

    status_h = np.where(time <= horizon + FIG2_TIME_EPS, status, 0)
    time_h = np.minimum(time, horizon)
    event_mask = (status_h == 1) | (status_h == 2)

    if not bool(event_mask.any()):
        return 0.0

    event_times = np.unique(time_h[event_mask])
    sorted_times = np.sort(time_h)
    n_risk = (
        len(sorted_times)
        - np.searchsorted(sorted_times, event_times, side="left")
    ).astype(np.float64)

    cause1_index = np.searchsorted(
        event_times, time_h[status_h == 1], side="left"
    )
    cause2_index = np.searchsorted(
        event_times, time_h[status_h == 2], side="left"
    )
    d1 = np.bincount(
        cause1_index, minlength=len(event_times)
    ).astype(np.float64)
    d2 = np.bincount(
        cause2_index, minlength=len(event_times)
    ).astype(np.float64)

    valid_risk = n_risk > 0
    if not bool(valid_risk.all()):
        event_times = event_times[valid_risk]
        n_risk = n_risk[valid_risk]
        d1 = d1[valid_risk]
        d2 = d2[valid_risk]

    total_hazard = np.clip((d1 + d2) / n_risk, 0.0, 1.0)
    survival_before = np.ones(len(event_times), dtype=np.float64)
    if len(event_times) > 1:
        survival_before[1:] = np.cumprod(1.0 - total_hazard[:-1])

    cif = np.sum(survival_before * (d1 / n_risk))
    return float(np.clip(cif, 0.0, 1.0))


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    values = values[valid]
    weights = weights[valid]

    if values.size == 0:
        return np.nan

    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    cutoff = float(quantile) * float(cumulative[-1])
    position = int(np.searchsorted(cumulative, cutoff, side="left"))
    position = int(np.clip(position, 0, len(values) - 1))
    return float(values[position])


def _make_observed_arrays(
    event_obs,
    t_tilde,
    at_risk,
    C_months,
    death_observed_post_index,
    death_time_months,
    entry_delay_months,
    horizon_months,
):
    """
    Reproduce the fixed-horizon outcome construction used by comparison Cell 4A.

    event_obs is already the canonical death-aware condition indicator produced
    by data preparation, so death-before-condition and configured ties remain
    resolved exactly as in the locked targets.
    """
    horizon = float(horizon_months)
    endpoint = float(entry_delay_months) + horizon

    C_mat = C_months[:, None]
    death_time_mat = death_time_months[:, None]
    death_obs_mat = death_observed_post_index[:, None]

    condition_before_entry = (
        event_obs
        & np.isfinite(t_tilde)
        & (t_tilde <= float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
    )
    death_before_entry = (
        death_obs_mat
        & np.isfinite(death_time_mat)
        & (death_time_mat <= float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
    )

    eligible = (
        at_risk
        & (C_mat > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
        & (~condition_before_entry)
        & (~death_before_entry)
    )

    condition_event = (
        eligible
        & event_obs
        & np.isfinite(t_tilde)
        & (t_tilde > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
        & (t_tilde <= endpoint + FIG2_TIME_EPS)
    )

    death_competing = (
        eligible
        & death_obs_mat
        & np.isfinite(death_time_mat)
        & (death_time_mat > float(entry_delay_months) + ENTRY_TIME_EPS_MONTHS)
        & (death_time_mat <= endpoint + FIG2_TIME_EPS)
        & (~condition_event)
    )

    observed_through = eligible & (C_mat >= endpoint - FIG2_TIME_EPS)
    known = condition_event | death_competing | observed_through

    status = np.zeros(event_obs.shape, dtype=np.int16)
    status[condition_event] = 1
    status[death_competing] = 2

    time_rel = np.full(event_obs.shape, horizon, dtype=np.float64)
    time_rel[condition_event] = (
        t_tilde[condition_event] - float(entry_delay_months)
    )

    if bool(death_competing.any()):
        death_rel = np.broadcast_to(
            death_time_mat - float(entry_delay_months), event_obs.shape
        )
        time_rel[death_competing] = death_rel[death_competing]

    censor_event = (
        eligible
        & (~condition_event)
        & (~death_competing)
        & (~observed_through)
    )
    if bool(censor_event.any()):
        censor_rel = np.broadcast_to(
            C_mat - float(entry_delay_months), event_obs.shape
        )
        time_rel[censor_event] = censor_rel[censor_event]

    time_rel = np.clip(time_rel, 0.0, horizon)

    return {
        "eligible": eligible,
        "known": known,
        "condition_event": condition_event,
        "death_competing": death_competing,
        "status": status,
        "time_rel": time_rel,
    }


# =============================================================================
# Load and validate the locked cohort and prediction arrays
# =============================================================================

test_meta = pd.read_pickle(TEST_SAMPLES_PATH).reset_index(drop=True)
required_meta_columns = {"eid", "index_age_years", "gender_idx"}
missing_meta_columns = sorted(required_meta_columns - set(test_meta.columns))
if missing_meta_columns:
    raise KeyError(
        "canonical_test_samples.pkl is missing required Figure-2 columns: "
        f"{missing_meta_columns}"
    )

with np.load(PREDICTION_PATH, allow_pickle=True) as locked_npz:
    npz_payload = {key: locked_npz[key] for key in locked_npz.files}

required_base_npz = {
    "eid",
    "condition_names",
    "horizons_months",
    "entry_delay_months",
    "at_risk",
    "event_obs",
    "t_tilde",
    "C_months",
    "death_observed_post_index",
    "death_time_months",
}
missing_base_npz = sorted(required_base_npz - set(npz_payload))
if missing_base_npz:
    raise KeyError(
        "The locked prediction NPZ is missing required arrays: "
        f"{missing_base_npz}"
    )

pred_eid = np.asarray(npz_payload["eid"], dtype=object).astype(str)
meta_eid = test_meta["eid"].astype(str).to_numpy()
if pred_eid.tolist() != meta_eid.tolist():
    raise AssertionError(
        "Prediction NPZ and canonical_test_samples.pkl have different "
        "participant ordering."
    )

index_age_years = pd.to_numeric(
    test_meta["index_age_years"], errors="coerce"
).to_numpy(dtype=np.float32)
if index_age_years.shape != (len(pred_eid),):
    raise ValueError("Canonical index_age_years has an unexpected shape.")
if not np.isfinite(index_age_years).all():
    raise ValueError("Canonical index_age_years contains non-finite values.")

npz_augmented = "index_age_years" not in npz_payload
if "index_age_years" in npz_payload:
    existing_index_age = np.asarray(
        npz_payload["index_age_years"], dtype=np.float32
    ).reshape(-1)
    if existing_index_age.shape != index_age_years.shape:
        raise ValueError(
            "Existing NPZ index_age_years has the wrong shape."
        )
    if not np.allclose(
        existing_index_age,
        index_age_years,
        rtol=0.0,
        atol=1e-5,
        equal_nan=False,
    ):
        raise AssertionError(
            "Existing NPZ index_age_years does not match canonical test samples."
        )
else:
    npz_payload["index_age_years"] = index_age_years
    temporary_npz_path = PREDICTION_PATH.with_name(
        PREDICTION_PATH.stem + ".figure2_tmp.npz"
    )
    np.savez_compressed(temporary_npz_path, **npz_payload)
    temporary_npz_path.replace(PREDICTION_PATH)

condition_names = [
    str(value).strip()
    for value in np.asarray(
        npz_payload["condition_names"], dtype=object
    ).tolist()
]
available_horizons = {
    int(value)
    for value in np.asarray(
        npz_payload["horizons_months"]
    ).reshape(-1).tolist()
}
missing_horizons = sorted(
    set(FIG2_HORIZONS_MONTHS) - available_horizons
)
if missing_horizons:
    raise ValueError(
        "Locked predictions are missing Figure-2 horizons: "
        f"{missing_horizons}"
    )

entry_delay_months = float(
    np.asarray(npz_payload["entry_delay_months"], dtype=np.float64).reshape(-1)[0]
)
N = int(len(pred_eid))
K = int(len(condition_names))

at_risk = np.asarray(npz_payload["at_risk"], dtype=np.float64) > 0.5
event_obs = np.asarray(npz_payload["event_obs"], dtype=np.float64) > 0.5
t_tilde = np.asarray(npz_payload["t_tilde"], dtype=np.float64)
C_months = np.asarray(
    npz_payload["C_months"], dtype=np.float64
).reshape(-1)
death_post = np.asarray(
    npz_payload["death_observed_post_index"], dtype=np.float64
).reshape(-1) > 0.5
death_time = np.asarray(
    npz_payload["death_time_months"], dtype=np.float64
).reshape(-1)

for name, array in {
    "at_risk": at_risk,
    "event_obs": event_obs,
    "t_tilde": t_tilde,
}.items():
    if array.shape != (N, K):
        raise ValueError(
            f"{name} has shape {array.shape}; expected {(N, K)}."
        )

for name, array in {
    "C_months": C_months,
    "death_observed_post_index": death_post,
    "death_time_months": death_time,
}.items():
    if array.shape != (N,):
        raise ValueError(
            f"{name} has shape {array.shape}; expected {(N,)}."
        )

risk_by_horizon = {}
for horizon in FIG2_HORIZONS_MONTHS:
    key = f"risk_death_adjusted_h{int(horizon)}"
    if key not in npz_payload:
        raise KeyError(f"Locked prediction NPZ is missing {key}.")
    risk = np.asarray(npz_payload[key], dtype=np.float64)
    if risk.shape != (N, K):
        raise ValueError(
            f"{key} has shape {risk.shape}; expected {(N, K)}."
        )
    if not np.isfinite(risk).all():
        raise ValueError(f"{key} contains non-finite predictions.")
    if np.any((risk < -1e-7) | (risk > 1.0 + 1e-7)):
        raise ValueError(f"{key} contains predictions outside [0, 1].")
    risk_by_horizon[int(horizon)] = np.clip(risk, 0.0, 1.0)

sex, sex_source = _resolve_sex(test_meta)
sex_counts = pd.Series(sex).value_counts(dropna=False).to_dict()
if int(np.sum(sex == "Male")) == 0:
    raise ValueError("No Male participants were resolved.")
if int(np.sum(sex == "Female")) == 0:
    raise ValueError("No Female participants were resolved.")

strata = {
    "All": np.ones(N, dtype=bool),
    "Male": sex == "Male",
    "Female": sex == "Female",
}


# =============================================================================
# Build condition × horizon × sex-stratum AJ calibration
# =============================================================================

decile_rows = []
summary_rows = []

for horizon in FIG2_HORIZONS_MONTHS:
    horizon = int(horizon)
    risk = risk_by_horizon[horizon]
    observed = _make_observed_arrays(
        event_obs=event_obs,
        t_tilde=t_tilde,
        at_risk=at_risk,
        C_months=C_months,
        death_observed_post_index=death_post,
        death_time_months=death_time,
        entry_delay_months=entry_delay_months,
        horizon_months=horizon,
    )

    for condition_id, condition_name in enumerate(condition_names):
        for stratum_name, participant_mask in strata.items():
            eligible_mask = (
                participant_mask
                & observed["eligible"][:, condition_id]
            )
            eligible_n = int(eligible_mask.sum())

            if eligible_n < int(FIG2_MIN_CALIBRATION_GROUP_N):
                continue

            scores = risk[eligible_mask, condition_id].astype(np.float64)
            times = observed["time_rel"][
                eligible_mask, condition_id
            ].astype(np.float64)
            statuses = observed["status"][
                eligible_mask, condition_id
            ].astype(np.int16)
            known_values = observed["known"][
                eligible_mask, condition_id
            ]
            condition_values = observed["condition_event"][
                eligible_mask, condition_id
            ]
            death_values = observed["death_competing"][
                eligible_mask, condition_id
            ]

            quantiles = np.linspace(
                0.0, 1.0, FIG2_N_CALIBRATION_BINS + 1
            )
            edges = np.unique(np.quantile(scores, quantiles))

            if edges.size <= 2:
                bin_id = np.zeros(eligible_n, dtype=np.int16)
                n_bins_actual = 1
            else:
                bin_id = np.searchsorted(
                    edges[1:-1], scores, side="right"
                ).astype(np.int16)
                n_bins_actual = int(bin_id.max()) + 1

            condition_decile_rows = []
            for calibration_bin in range(n_bins_actual):
                in_bin = bin_id == calibration_bin
                bin_n = int(in_bin.sum())
                if bin_n < int(FIG2_MIN_CALIBRATION_BIN_N):
                    continue

                predicted_mean = float(np.mean(scores[in_bin]))
                observed_aj_cif = _aalen_johansen_cif(
                    time=times[in_bin],
                    status=statuses[in_bin],
                    horizon_months=horizon,
                )
                error = (
                    float(observed_aj_cif - predicted_mean)
                    if np.isfinite(observed_aj_cif)
                    else np.nan
                )

                row = {
                    "Model": FIG2_EXPORT_MODEL_NAME,
                    "model": FIG2_EXPORT_MODEL_NAME,
                    "endpoint_type": "condition",
                    "risk_type": FIG2_RISK_TYPE,
                    "condition_id": int(condition_id),
                    "condition_name": condition_name,
                    "horizon_months": horizon,
                    "calibration_stratum": stratum_name,
                    "sex": stratum_name,
                    "calibration_bin": int(calibration_bin),
                    "n": bin_n,
                    "eligible_n": bin_n,
                    "known_n": int(known_values[in_bin].sum()),
                    "event_n": int(condition_values[in_bin].sum()),
                    "death_competing_n": int(death_values[in_bin].sum()),
                    "predicted_mean": predicted_mean,
                    "predicted_min": float(np.min(scores[in_bin])),
                    "predicted_max": float(np.max(scores[in_bin])),
                    "observed_aj_cif": observed_aj_cif,
                    "observed_minus_predicted": error,
                    "absolute_error": (
                        float(abs(error)) if np.isfinite(error) else np.nan
                    ),
                }
                decile_rows.append(row)
                condition_decile_rows.append(row)

            finite_rows = [
                row
                for row in condition_decile_rows
                if np.isfinite(row["absolute_error"])
            ]
            absolute_errors = np.asarray(
                [row["absolute_error"] for row in finite_rows],
                dtype=np.float64,
            )
            bin_weights = np.asarray(
                [row["n"] for row in finite_rows],
                dtype=np.float64,
            )

            if absolute_errors.size:
                ici = float(
                    np.average(absolute_errors, weights=bin_weights)
                )
                e50 = _weighted_quantile(
                    absolute_errors, bin_weights, 0.50
                )
                e90 = _weighted_quantile(
                    absolute_errors, bin_weights, 0.90
                )
                ici_unweighted = float(np.mean(absolute_errors))
                e50_unweighted = float(
                    np.quantile(absolute_errors, 0.50)
                )
                e90_unweighted = float(
                    np.quantile(absolute_errors, 0.90)
                )
            else:
                ici = np.nan
                e50 = np.nan
                e90 = np.nan
                ici_unweighted = np.nan
                e50_unweighted = np.nan
                e90_unweighted = np.nan

            summary_rows.append(
                {
                    "Model": FIG2_EXPORT_MODEL_NAME,
                    "model": FIG2_EXPORT_MODEL_NAME,
                    "endpoint_type": "condition",
                    "risk_type": FIG2_RISK_TYPE,
                    "condition_id": int(condition_id),
                    "condition_name": condition_name,
                    "horizon_months": horizon,
                    "calibration_stratum": stratum_name,
                    "sex": stratum_name,
                    "ici": ici,
                    "e50": e50,
                    "e90": e90,
                    "ici_unweighted_bins": ici_unweighted,
                    "e50_unweighted_bins": e50_unweighted,
                    "e90_unweighted_bins": e90_unweighted,
                    "n_calibration_bins": int(len(finite_rows)),
                    "total_eligible_n": eligible_n,
                    "total_known_n": int(known_values.sum()),
                    "total_event_n": int(condition_values.sum()),
                    "total_death_competing_n": int(death_values.sum()),
                    "mean_predicted_risk": float(np.mean(scores)),
                }
            )

deciles_all = pd.DataFrame(decile_rows)
summary_all = pd.DataFrame(summary_rows)

if deciles_all.empty:
    raise RuntimeError("No Figure-2 calibration-decile rows were generated.")
if summary_all.empty:
    raise RuntimeError("No Figure-2 calibration-summary rows were generated.")

deciles_all = deciles_all.sort_values(
    [
        "horizon_months",
        "condition_id",
        "calibration_stratum",
        "calibration_bin",
    ]
).reset_index(drop=True)
summary_all = summary_all.sort_values(
    [
        "horizon_months",
        "condition_id",
        "calibration_stratum",
    ]
).reset_index(drop=True)

deciles_2yr = deciles_all[
    deciles_all["horizon_months"].eq(
        FIG2_PRIMARY_HORIZON_MONTHS
    )
].copy()
summary_2yr = summary_all[
    summary_all["horizon_months"].eq(
        FIG2_PRIMARY_HORIZON_MONTHS
    )
].copy()


# =============================================================================
# Save canonical Figure-2 calibration files and compatibility aliases
# =============================================================================

DECILES_2YR_PATH = (
    FIG2_INPUT_DIR
    / "figure2_revised_panelH_calibration_deciles_2yr.csv"
)
SUMMARY_2YR_PATH = (
    FIG2_INPUT_DIR
    / "figure2_revised_panelH_calibration_summary_2yr.csv"
)
DECILES_ALL_PATH = (
    FIG2_INPUT_DIR
    / "figure2_revised_panelH_calibration_deciles_all_horizons.csv"
)
SUMMARY_ALL_ROOT_PATH = (
    RUN_DIR
    / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
)
SUMMARY_ALL_FIG2_PATH = (
    FIG2_INPUT_DIR
    / "figure2_revised_panelH_calibration_summary_all_horizons.csv"
)
SUMMARY_ALL_ALIAS_ROOT_PATH = (
    RUN_DIR / "test_calibration_error_by_horizon_condition_sex.csv"
)
SUMMARY_ALL_ALIAS_FIG2_PATH = (
    FIG2_INPUT_DIR
    / "test_calibration_error_by_horizon_condition_sex.csv"
)

deciles_2yr.to_csv(DECILES_2YR_PATH, index=False)
summary_2yr.to_csv(SUMMARY_2YR_PATH, index=False)
deciles_all.to_csv(DECILES_ALL_PATH, index=False)
summary_all.to_csv(SUMMARY_ALL_ROOT_PATH, index=False)
summary_all.to_csv(SUMMARY_ALL_FIG2_PATH, index=False)
summary_all.to_csv(SUMMARY_ALL_ALIAS_ROOT_PATH, index=False)
summary_all.to_csv(SUMMARY_ALL_ALIAS_FIG2_PATH, index=False)


# =============================================================================
# Hard compatibility checks for Figures 1 and 2
# =============================================================================

figure1_metrics = pd.read_csv(PER_CONDITION_METRICS_PATH)
required_figure1_columns = {
    "horizon_months",
    "condition_name",
    "risk_type",
    "event_n",
    "auroc",
}
missing_figure1_columns = sorted(
    required_figure1_columns - set(figure1_metrics.columns)
)
if missing_figure1_columns:
    raise ValueError(
        "Figure-1 CRESCENT metrics are missing columns: "
        f"{missing_figure1_columns}"
    )

required_decile_columns = {
    "condition_id",
    "condition_name",
    "horizon_months",
    "calibration_stratum",
    "sex",
    "calibration_bin",
    "n",
    "known_n",
    "event_n",
    "predicted_mean",
    "observed_aj_cif",
}
required_summary_columns = {
    "condition_id",
    "condition_name",
    "horizon_months",
    "calibration_stratum",
    "sex",
    "ici",
    "e50",
    "e90",
    "total_event_n",
}

missing_decile_columns = sorted(
    required_decile_columns - set(deciles_2yr.columns)
)
missing_summary_columns = sorted(
    required_summary_columns - set(summary_all.columns)
)
if missing_decile_columns:
    raise ValueError(
        "Figure-2 calibration-decile export is missing columns: "
        f"{missing_decile_columns}"
    )
if missing_summary_columns:
    raise ValueError(
        "Figure-2 calibration-summary export is missing columns: "
        f"{missing_summary_columns}"
    )

export_horizons = sorted(
    pd.to_numeric(
        summary_all["horizon_months"], errors="coerce"
    ).dropna().astype(int).unique().tolist()
)
if export_horizons != sorted(FIG2_HORIZONS_MONTHS):
    raise ValueError(
        "All-horizon calibration export has the wrong horizons: "
        f"{export_horizons}"
    )

export_strata = set(summary_all["calibration_stratum"].astype(str))
if not {"All", "Male", "Female"}.issubset(export_strata):
    raise ValueError(
        "Calibration export does not contain All, Male, and Female strata."
    )

if deciles_2yr.empty or summary_2yr.empty:
    raise RuntimeError("The required 2-year calibration exports are empty.")

with np.load(PREDICTION_PATH, allow_pickle=True) as final_npz:
    required_figure2_npz = {
        "eid",
        "condition_names",
        "horizons_months",
        "entry_delay_months",
        "index_age_years",
        "at_risk",
        "event_obs",
        "t_tilde",
        "C_months",
        "death_observed_post_index",
        "death_time_months",
        *{
            f"risk_death_adjusted_h{int(horizon)}"
            for horizon in FIG2_HORIZONS_MONTHS
        },
    }
    missing_figure2_npz = sorted(
        required_figure2_npz - set(final_npz.files)
    )
    if missing_figure2_npz:
        raise KeyError(
            "Final Figure-2 prediction NPZ is missing arrays: "
            f"{missing_figure2_npz}"
        )


# =============================================================================
# Metadata / source audit
# =============================================================================

output_paths = {
    "prediction_npz": PREDICTION_PATH,
    "figure1_per_condition_metrics": PER_CONDITION_METRICS_PATH,
    "calibration_deciles_2yr": DECILES_2YR_PATH,
    "calibration_summary_2yr": SUMMARY_2YR_PATH,
    "calibration_deciles_all_horizons": DECILES_ALL_PATH,
    "calibration_summary_all_horizons_root": SUMMARY_ALL_ROOT_PATH,
    "calibration_summary_all_horizons_figure2": SUMMARY_ALL_FIG2_PATH,
    "calibration_summary_alias_root": SUMMARY_ALL_ALIAS_ROOT_PATH,
    "calibration_summary_alias_figure2": SUMMARY_ALL_ALIAS_FIG2_PATH,
}

metadata = {
    "model": FIG2_EXPORT_MODEL_NAME,
    "risk_type": FIG2_RISK_TYPE,
    "run_dir": str(RUN_DIR),
    "n_test_participants": N,
    "n_conditions": K,
    "horizons_months": list(FIG2_HORIZONS_MONTHS),
    "primary_horizon_months": FIG2_PRIMARY_HORIZON_MONTHS,
    "entry_delay_months": entry_delay_months,
    "n_calibration_bins_requested": FIG2_N_CALIBRATION_BINS,
    "minimum_calibration_group_n": FIG2_MIN_CALIBRATION_GROUP_N,
    "minimum_calibration_bin_n": FIG2_MIN_CALIBRATION_BIN_N,
    "sex_source": sex_source,
    "gender_index_to_sex": {
        "0": "Female",
        "1": "Male",
        "2": "Unknown",
    },
    "sex_counts": {
        str(key): int(value) for key, value in sex_counts.items()
    },
    "npz_index_age_years_added": bool(npz_augmented),
    "observed_calibration": (
        "Aalen-Johansen condition CIF with death as competing event, "
        "using canonical death-aware event_obs and the Cell-7 entry delay"
    ),
    "calibration_error_definition": (
        "ICI/E50/E90 are participant-count-weighted absolute errors across "
        "quantile calibration bins within condition, horizon, and stratum"
    ),
    "outputs": {
        key: str(path) for key, path in output_paths.items()
    },
    "sha256": {
        key: _sha256_file(path) for key, path in output_paths.items()
    },
    "figure1_crescent_inputs_ready": True,
    "figure2_crescent_inputs_ready_for_comparison_cell4a": True,
}

METADATA_PATH = RUN_DIR / "figure2_calibration_export_metadata.json"
with open(METADATA_PATH, "w") as handle:
    json.dump(metadata, handle, indent=2, default=str)

print("=" * 88)
print("A10/CRESCENT Figure-2 calibration export complete")
print("=" * 88)
print("RUN_DIR:", RUN_DIR)
print("Participants:", N)
print("Conditions:", K)
print("Sex counts:", sex_counts)
print("Added index_age_years to NPZ:", npz_augmented)
print("Figure 1 CRESCENT input: READY")
print("Figure 2 CRESCENT calibration inputs: READY for comparison Cell 4A")
print("Calibration rows:", len(deciles_all))
print("Calibration summary rows:", len(summary_all))
print("Metadata:", METADATA_PATH)
for label, path in output_paths.items():
    print(f"  {label}: {path}")


# =============================================================================
# Notebook cell 26
# =============================================================================

# =============================================================================
# COMPLETE FIGURE 2 MODEL-DEPENDENT INPUT EXPORT — DELAYED ENTRY
#
# Panel H still requires an independent comparator with the same landmark,
# delayed-entry design, target definitions, and test participants. This block exports only
# CRESCENT-derived data from the locked predictions and canonical test rows.
# =============================================================================

import os
import shutil
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

FIG2_TARGET_MODEL = FIG2_EXPORT_MODEL_NAME
CRESCENT_FIG2_DIR = FIG2_INPUT_DIR
CRESCENT_RUN_DIR = RUN_DIR
FIG2_AGE_BIN_WIDTH_YEARS = 5
FIG2_MIN_AGE_BIN_ELIGIBLE = 10
GENDER_INDEX_TO_SEX = {0: "Female", 1: "Male", 2: "Unknown"}

FIG2_SELECTED_CONDITIONS = (
    "Cerebrovascular disease",
    "Chronic kidney disease",
    "Chronic lower respiratory diseases",
    "Dementia",
    "Dermatitis and eczema",
    "Diabetes",
    "Hypertensive diseases",
    "Ischaemic heart diseases",
    "Inflammatory polyarthropathies",
    "Mood disorders",
)


required_horizons = [int(h) for h in FIG2_HORIZONS_MONTHS]
H = deciles_2yr
Hsummary = summary_2yr
ALL_CAL = summary_all
test_meta = test_meta.reset_index(drop=True)
death_time = death_time.astype(np.float64)
death_post = death_post.astype(bool)
index_age_years = index_age_years.astype(np.float64)
project_root = (RUN_DIR.parents[2] if RUN_DIR.parent.name == "runs2"
                and RUN_DIR.parent.parent.name.startswith("Models") else Path.cwd())
color_source_setting = os.getenv("FIG2_COLOR_KEY_SOURCE", "").strip()
if color_source_setting:
    FIG2_COLOR_KEY_SOURCE = Path(color_source_setting).expanduser().resolve()
else:
    FIG2_COLOR_KEY_SOURCE = (
        project_root / "Models2" / "runs2"
        / "RevisedModel_death_adjusted_reader_temporal_kernel_V1"
        / "figure2_revised_inputs_2yr"
        / "figure2_revised_condition_color_key.csv"
    ).resolve()

color_source = FIG2_COLOR_KEY_SOURCE

def make_observed_arrays(horizon_months):
    output = _make_observed_arrays(
        event_obs=event_obs, t_tilde=t_tilde, at_risk=at_risk,
        C_months=C_months, death_observed_post_index=death_post,
        death_time_months=death_time, entry_delay_months=entry_delay_months,
        horizon_months=horizon_months,
    )
    output["y"] = output["condition_event"].astype(np.int8)
    return output

def binary_metrics(y, risk, known):
    valid = np.asarray(known, dtype=bool) & np.isfinite(risk)
    values = np.asarray(y, dtype=np.int8)[valid]
    scores = np.asarray(risk, dtype=np.float64)[valid]
    if values.size < 2 or np.unique(values).size != 2:
        return {"auroc": np.nan, "auprc": np.nan}
    return {"auroc": float(roc_auc_score(values, scores)),
            "auprc": float(average_precision_score(values, scores))}

aalen_johansen_cif = _aalen_johansen_cif
observed_by_horizon = {
    int(h): make_observed_arrays(int(h)) for h in required_horizons
}

# This is the same complete-case plotting estimand as the supplied Figure 2
# script. The main model's primary IPCW AUROC remains in the evaluation CSV.
# Static condition metadata / colour key
# =============================================================================

condition_base = pd.DataFrame(
    {
        "condition_id": np.arange(K, dtype=int),
        "condition_name": condition_names,
    }
)

color_source = Path(FIG2_COLOR_KEY_SOURCE).expanduser().resolve()
if color_source.exists():
    external_color = pd.read_csv(color_source)
    if "condition_name" not in external_color.columns:
        warnings.warn(
            f"Ignoring colour key without condition_name: {color_source}"
        )
        external_color = pd.DataFrame(columns=["condition_name"])
else:
    external_color = pd.DataFrame(columns=["condition_name"])

for column in ["ICD_Chapter", "Chapter_Name"]:
    if column not in external_color.columns:
        external_color[column] = "Unmapped"

external_color = (
    external_color[["condition_name", "ICD_Chapter", "Chapter_Name"]]
    .drop_duplicates("condition_name")
    .copy()
)
external_color["condition_name"] = external_color["condition_name"].astype(str).str.strip()

color_key = condition_base.merge(external_color, on="condition_name", how="left")
color_key["ICD_Chapter"] = color_key["ICD_Chapter"].fillna("Unmapped").astype(str)
color_key["Chapter_Name"] = color_key["Chapter_Name"].fillna("Unmapped").astype(str)
color_key_path = CRESCENT_FIG2_DIR / "figure2_revised_condition_color_key.csv"
color_key.to_csv(color_key_path, index=False)

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
        meta = chapter_lookup.get(
            condition_name,
            {"ICD_Chapter": "Unmapped", "Chapter_Name": "Unmapped"},
        )

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



# Verify every expected panel input before marking this model export complete.
fig2_panel_paths = {
    "A": A_path, "B": B_path, "C": C_path, "Cpair": Cpair_path,
    "E": E_path, "F": F_path, "H": DECILES_2YR_PATH,
    "Hsummary": SUMMARY_2YR_PATH, "Hsummary_all": SUMMARY_ALL_FIG2_PATH,
    "colors": color_key_path, "selected": selected_path,
}
for label, path in fig2_panel_paths.items():
    if not path.is_file():
        raise FileNotFoundError(f"Figure 2 {label} export missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise RuntimeError(f"Figure 2 {label} export is empty: {path}")

required_panel_columns = {
    "A": {"condition_id", "condition_name", "sex", "age_bin_mid",
          "mean_predicted_risk", "observed_aj_cif", "eligible_n"},
    "B": {"condition_id", "condition_name", "sex", "event_n",
          "auroc", "auprc", "frequency_for_plot"},
    "C": {"condition_id", "condition_name", "sex", "ICD_Chapter",
          "Chapter_Name", "auroc"},
    "Cpair": {"condition_id", "Female", "Male",
              "auroc_female_minus_male"},
    "E": {"condition_id", "condition_name", "sex",
          "horizon_months", "auroc"},
    "F": {"condition_id", "condition_name", "fpr", "tpr",
          "threshold", "auroc"},
    "H": required_decile_columns,
    "Hsummary": required_summary_columns,
    "selected": {"selection_rank", "condition_name", "condition_id"},
    "colors": {"condition_name", "ICD_Chapter", "Chapter_Name"},
}
for label, required in required_panel_columns.items():
    existing = set(pd.read_csv(fig2_panel_paths[label], nrows=0).columns)
    if required - existing:
        raise ValueError(f"Figure 2 {label} missing {sorted(required-existing)}")

if len(selection) != len(FIG2_SELECTED_CONDITIONS):
    raise AssertionError("Selected Figure 2 condition list is incomplete")
if set(E["horizon_months"].astype(int)) != set(FIG2_HORIZONS_MONTHS):
    raise AssertionError("Figure 2 horizon table has the wrong time axis")
if not set(A["condition_name"]).issubset(set(selection["condition_name"])):
    raise AssertionError("Panel A contains an unselected condition")

plot_comparator_setting = os.getenv("FIG2_COMPARATOR_METRICS_PATH", "").strip()
if plot_comparator_setting:
    plot_comparator_path = Path(plot_comparator_setting).expanduser().resolve()
else:
    plot_comparator_path = (
        project_root / "OtherModels5" / "runs2"
        / "Delphi2M_Canonical_A10_SixMonthDelayedEntry_Fresh50k"
        / "test_per_condition_metrics_by_horizon.csv"
    ).resolve()

panel_h_contract = {
    "status": "requires_matched_comparator_validation",
    "comparator_metrics_path": str(plot_comparator_path),
    "crescent_entry_delay_months": float(entry_delay_months),
    "required_comparator_entry_delay_months": float(entry_delay_months),
    "required_test_eid_alignment": True,
    "required_condition_target_alignment": True,
}
comparator_config_path = plot_comparator_path.parent / "config.json"
if comparator_config_path.is_file():
    comparator_config = json.loads(comparator_config_path.read_text())
    comparator_settings = comparator_config.get("config", comparator_config)
    comparator_entry = comparator_settings.get("ENTRY_DELAY_MONTHS")
    comparator_contract = comparator_config.get("delayed_entry_contract", {})
    if comparator_entry is None:
        comparator_entry = comparator_contract.get("entry_delay_months")
    matched_delay = comparator_entry is not None and np.isclose(float(comparator_entry), entry_delay_months, atol=1e-8)
    matched_likelihood = comparator_contract.get("training_likelihood") == delayed_entry_contract()["training_likelihood"]
    panel_h_contract["comparator_delayed_entry_verified"] = bool(matched_delay and matched_likelihood)
    if not (matched_delay and matched_likelihood):
        panel_h_contract["status"] = "incompatible_or_unverified_delayed_entry"
        warnings.warn("Panel H requires a comparator trained with the same six-month delayed-entry likelihood; a forecast offset alone is insufficient.", RuntimeWarning)
    else:
        panel_h_contract["status"] = "entry_contract_verified_targets_and_eids_still_require_alignment"

figure2_manifest = {
    "model": FIG2_TARGET_MODEL,
    "comparator_model": "Delphi-2M",
    "run_dir": str(RUN_DIR),
    "metrics_path": str(PER_CONDITION_METRICS_PATH),
    "comparator_metrics_path": str(plot_comparator_path),
    "prediction_path": str(PREDICTION_PATH),
    "canonical_test_samples_path": str(TEST_SAMPLES_PATH),
    "figure2_input_dir": str(FIG2_INPUT_DIR),
    "n_test_participants": int(N),
    "n_conditions": int(K),
    "entry_delay_months": float(entry_delay_months),
    "horizons_months": list(FIG2_HORIZONS_MONTHS),
    "selected_condition_ids": selected_ids,
    "selected_condition_names": selection["condition_name"].tolist(),
    "sex_source": sex_source,
    "color_key_source": str(color_source) if color_source.is_file() else None,
    "panel_h_contract": panel_h_contract,
    "plot_auroc_estimator": "complete_case_known_outcomes_unweighted",
    "primary_auroc_source": "test_per_condition_metrics_by_horizon.csv:auc_ipcw_cr",
    "model_dependent_sources_all_within_crescent_run": True,
    "outputs": {label: str(path) for label, path in fig2_panel_paths.items()},
    "sha256": {label: _sha256_file(path) for label, path in fig2_panel_paths.items()},
}
figure2_manifest_path = FIG2_INPUT_DIR / "figure2_crescent_source_manifest.json"
with figure2_manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(figure2_manifest, handle, indent=2, default=str)

metadata["figure2_crescent_inputs_ready_for_comparison_cell4a"] = True
metadata["figure2_panels_exported"] = list(fig2_panel_paths)
metadata["panel_h_contract"] = panel_h_contract
metadata["outputs"].update({label: str(path)
                           for label, path in fig2_panel_paths.items()})
metadata["sha256"].update({label: _sha256_file(path)
                          for label, path in fig2_panel_paths.items()})
metadata["outputs"]["figure2_source_manifest"] = str(figure2_manifest_path)
metadata["sha256"]["figure2_source_manifest"] = _sha256_file(figure2_manifest_path)
with METADATA_PATH.open("w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2, default=str)

print("CRESCENT Figure 2 inputs ready:", FIG2_INPUT_DIR)
for label, path in fig2_panel_paths.items():
    print(f"  {label:12s}: {path}")
print("Panel H:", panel_h_contract["status"])
