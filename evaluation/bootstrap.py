"""Stage 7: participant-level bootstrap confidence intervals.

Extracted from the CRESCENT_Final notebook (cell 22). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 22
# =============================================================================

import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    JOBLIB_AVAILABLE = False

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def decorator(function):
            return function

        return decorator


# =============================================================================
# USER-EDITABLE CONFIGURATION
# =============================================================================

CI_BOOTSTRAP_N = int(getattr(cfg, "EVAL_BOOTSTRAP_N", 200))
CI_BOOTSTRAP_SEED = int(getattr(cfg, "EVAL_BOOTSTRAP_SEED", 101))
CI_ALPHA = float(getattr(cfg, "EVAL_BOOTSTRAP_ALPHA", 0.05))

# Fixed-horizon discrimination estimates retained for main/supplementary use.
CI_POINT_HORIZONS_MONTHS = tuple(
    int(value)
    for value in getattr(cfg, "EVAL_CI_POINT_HORIZONS_MONTHS", (24, 60))
)

# Wolbers-style C-index: rank by the predicted CIF at tau and include event-of-
# interest cases observed by tau.
CI_CINDEX_TAU_MONTHS = int(getattr(cfg, "EVAL_CINDEX_TAU_MONTHS", 60))

# None means all conditions. To test the cell quickly, use a short tuple such
# as (0, 1, 2), then restore None for the publication run.
CI_CONDITION_IDS = None

# Threading shares the large prediction arrays instead of copying them into
# separate processes. Reduce this if the notebook server has limited CPUs.
CI_N_JOBS = max(1, min(8, int(os.cpu_count() or 1)))
CI_BOOTSTRAP_CHUNK_SIZE = 25

# A condition-level CI is withheld if fewer than this fraction of bootstrap
# estimates are finite. The counts and status remain in the output.
CI_MIN_VALID_BOOTSTRAP_FRACTION = float(
    getattr(cfg, "EVAL_CI_MIN_VALID_BOOTSTRAP_FRACTION", 0.90)
)

# For a macro estimate, use a fixed point-valid condition set. A bootstrap
# macro estimate is retained only if at least this fraction of that fixed set
# has a finite estimate in the replicate.
CI_MACRO_MIN_CONDITION_FRACTION = float(
    getattr(cfg, "EVAL_CI_MACRO_MIN_CONDITION_FRACTION", 0.90)
)

# Existing valid chunk files are reused. Set True only when deliberately
# replacing chunks for this exact configuration.
CI_FORCE_RECOMPUTE = False


# =============================================================================
# REQUIRED FILES AND LOCKED SETTINGS
# =============================================================================

if "RUN_DIR" not in globals():
    raise RuntimeError(
        "RUN_DIR is missing. Run the model setup cell before Cell 7B."
    )

RUN_DIR = Path(RUN_DIR).expanduser().resolve()

TRAIN_SAMPLES_PATH = RUN_DIR / "canonical_train_samples.pkl"
TEST_SAMPLES_PATH = RUN_DIR / "canonical_test_samples.pkl"
DENSE_PREDICTION_PATH = (
    RUN_DIR / "test_dense_prediction_arrays_for_integrated_metrics.npz"
)
DENSE_POINT_METRICS_PATH = (
    RUN_DIR / "test_dense_metrics_by_condition_and_horizon.csv"
)
INTEGRATED_POINT_METRICS_PATH = (
    RUN_DIR / "test_integrated_metrics_by_condition.csv"
)

required_paths = [
    TRAIN_SAMPLES_PATH,
    TEST_SAMPLES_PATH,
    DENSE_PREDICTION_PATH,
    DENSE_POINT_METRICS_PATH,
    INTEGRATED_POINT_METRICS_PATH,
]
missing_paths = [str(path) for path in required_paths if not path.is_file()]
if missing_paths:
    raise FileNotFoundError(
        "Cell 7B is missing required prior-cell artifacts:\n"
        + "\n".join(missing_paths)
    )

if not (0.0 < float(CI_ALPHA) < 1.0):
    raise ValueError("CI_ALPHA must be strictly between 0 and 1.")
if int(CI_BOOTSTRAP_N) < 1:
    raise ValueError("CI_BOOTSTRAP_N must be at least 1.")
if int(CI_BOOTSTRAP_CHUNK_SIZE) < 1:
    raise ValueError("CI_BOOTSTRAP_CHUNK_SIZE must be at least 1.")
if not (0.0 < float(CI_MIN_VALID_BOOTSTRAP_FRACTION) <= 1.0):
    raise ValueError(
        "CI_MIN_VALID_BOOTSTRAP_FRACTION must lie in (0, 1]."
    )
if not (0.0 < float(CI_MACRO_MIN_CONDITION_FRACTION) <= 1.0):
    raise ValueError(
        "CI_MACRO_MIN_CONDITION_FRACTION must lie in (0, 1]."
    )

MODEL_NAME_CI = str(
    getattr(
        globals().get("cfg", object()),
        "MODEL_VARIANT",
        "CRESCENT_ONE_MONTH_CAP_CRP_3LAYER_TEMPORAL_INTERACTION_NLL_SHARED_STATE",
    )
)
PROB_EPS_CI = float(
    getattr(globals().get("cfg", object()), "PROB_EPS", 1e-7)
)
CENSOR_SURVIVAL_FLOOR_CI = float(
    getattr(
        globals().get("cfg", object()),
        "EVAL_CENSOR_SURVIVAL_FLOOR",
        0.01,
    )
)
CR_TARGET_CENSOR_CI = int(
    getattr(globals().get("cfg", object()), "CR_TARGET_CENSOR", 0)
)
CR_TARGET_CONDITION_CI = int(
    getattr(globals().get("cfg", object()), "CR_TARGET_CONDITION", 1)
)
CR_TARGET_DEATH_CI = int(
    getattr(globals().get("cfg", object()), "CR_TARGET_DEATH", 2)
)
TIME_EPS_CI = 1e-8


# =============================================================================
# LOAD AND VALIDATE LOCKED ARRAYS
# =============================================================================

train_samples_ci = pd.read_pickle(TRAIN_SAMPLES_PATH)
test_samples_ci = pd.read_pickle(TEST_SAMPLES_PATH)

with np.load(DENSE_PREDICTION_PATH, allow_pickle=True) as dense_npz_ci:
    required_npz_keys = {
        "eid",
        "condition_names",
        "dense_horizons_months",
        "entry_delay_months",
        "risk_death_adjusted",
    }
    missing_npz_keys = sorted(required_npz_keys - set(dense_npz_ci.files))
    if missing_npz_keys:
        raise KeyError(
            "Dense prediction NPZ is missing keys: "
            f"{missing_npz_keys}"
        )

    dense_eids_ci = np.asarray(dense_npz_ci["eid"]).astype(str)
    condition_names_ci = [
        str(value)
        for value in np.asarray(
            dense_npz_ci["condition_names"],
            dtype=object,
        ).tolist()
    ]
    dense_horizons_ci = tuple(
        int(value)
        for value in np.asarray(
            dense_npz_ci["dense_horizons_months"],
            dtype=np.int64,
        ).tolist()
    )
    entry_delay_months_ci = float(
        np.asarray(dense_npz_ci["entry_delay_months"]).reshape(-1)[0]
    )
    dense_risk_ci = np.asarray(
        dense_npz_ci["risk_death_adjusted"],
        dtype=np.float64,
    )

expected_integration_start_ci = int(
    getattr(cfg, "EVAL_INTEGRATION_START_MONTHS", 12)
)
expected_integration_end_ci = int(
    getattr(cfg, "EVAL_INTEGRATION_END_MONTHS", 120)
)
if (
    dense_horizons_ci[0] != expected_integration_start_ci
    or dense_horizons_ci[-1] != expected_integration_end_ci
):
    raise AssertionError(
        "Dense-grid integration window does not match the locked V11 "
        f"configuration: grid={dense_horizons_ci[0]}–{dense_horizons_ci[-1]}, "
        f"expected={expected_integration_start_ci}–{expected_integration_end_ci}."
    )

test_eids_ci = test_samples_ci["eid"].astype(str).to_numpy()
train_eids_ci = train_samples_ci["eid"].astype(str).to_numpy()

if pd.Series(test_eids_ci).duplicated().any():
    raise AssertionError(
        "canonical_test_samples.pkl is not one row per participant."
    )
if pd.Series(train_eids_ci).duplicated().any():
    raise AssertionError(
        "canonical_train_samples.pkl is not one row per participant."
    )
if dense_eids_ci.tolist() != test_eids_ci.tolist():
    raise AssertionError(
        "Dense prediction rows do not match canonical test eid order."
    )

n_test_ci = int(len(test_samples_ci))
n_conditions_ci = int(len(condition_names_ci))
n_dense_horizons_ci = int(len(dense_horizons_ci))

expected_risk_shape_ci = (
    n_test_ci,
    n_conditions_ci,
    n_dense_horizons_ci,
)
if dense_risk_ci.shape != expected_risk_shape_ci:
    raise ValueError(
        "Dense risk cube has shape "
        f"{dense_risk_ci.shape}; expected {expected_risk_shape_ci}."
    )
if not np.isfinite(dense_risk_ci).all():
    raise ValueError("Dense risk cube contains non-finite values.")
if np.any((dense_risk_ci < -1e-7) | (dense_risk_ci > 1.0 + 1e-7)):
    raise ValueError("Dense risk cube contains values outside [0, 1].")
dense_risk_ci = np.clip(dense_risk_ci, 0.0, 1.0)

if tuple(sorted(set(dense_horizons_ci))) != dense_horizons_ci:
    raise AssertionError(
        "Dense horizons must be unique and strictly increasing."
    )

requested_horizons_ci = tuple(
    int(value) for value in CI_POINT_HORIZONS_MONTHS
)
if len(set(requested_horizons_ci)) != len(requested_horizons_ci):
    raise ValueError("CI_POINT_HORIZONS_MONTHS contains duplicates.")
missing_requested_horizons_ci = sorted(
    set(requested_horizons_ci) - set(dense_horizons_ci)
)
if missing_requested_horizons_ci:
    raise ValueError(
        "Requested point horizons are absent from the dense grid: "
        f"{missing_requested_horizons_ci}"
    )
if int(CI_CINDEX_TAU_MONTHS) not in set(dense_horizons_ci):
    raise ValueError(
        "CI_CINDEX_TAU_MONTHS must be present in the dense prediction grid."
    )

if CI_CONDITION_IDS is None:
    selected_condition_ids_ci = tuple(range(n_conditions_ci))
else:
    selected_condition_ids_ci = tuple(
        int(value) for value in CI_CONDITION_IDS
    )
    if len(set(selected_condition_ids_ci)) != len(
        selected_condition_ids_ci
    ):
        raise ValueError("CI_CONDITION_IDS contains duplicates.")
    invalid_condition_ids_ci = [
        value
        for value in selected_condition_ids_ci
        if value < 0 or value >= n_conditions_ci
    ]
    if invalid_condition_ids_ci:
        raise ValueError(
            "CI_CONDITION_IDS contains out-of-range values: "
            f"{invalid_condition_ids_ci}"
        )

dense_horizon_to_index_ci = {
    int(horizon): int(index)
    for index, horizon in enumerate(dense_horizons_ci)
}


# =============================================================================
# COMPETING-RISK OUTCOMES AFTER THE DELAYED-ENTRY LANDMARK
# =============================================================================

def make_ci_condition_outcomes(frame):
    """Canonical competing-risk outcomes conditional on delayed-entry eligibility."""
    return make_delayed_entry_outcomes_numpy(
        np.stack(frame["cr_at_risk"].to_numpy()),
        np.stack(frame["cr_event_type"].to_numpy()),
        np.stack(frame["cr_event_time_months"].to_numpy()),
        frame["C_months"].to_numpy(dtype=float),
    )


train_outcomes_ci = make_ci_condition_outcomes(train_samples_ci)
test_outcomes_ci = make_ci_condition_outcomes(test_samples_ci)


# =============================================================================
# FROZEN TRAIN-SPLIT CENSORING DISTRIBUTIONS
# =============================================================================

class FrozenTrainingCensoringKM:
    """
    Marginal Kaplan–Meier estimator of administrative-censoring survival.

    before(t) returns G(t-); after(t) returns G(t). Both are estimated once
    from the canonical training split and are not refitted in test bootstraps.
    """

    def __init__(self, exit_time, censor_event, floor):
        time_values = np.asarray(exit_time, dtype=np.float64).reshape(-1)
        event_values = np.asarray(censor_event, dtype=bool).reshape(-1)
        valid = np.isfinite(time_values) & (time_values >= 0.0)
        time_values = time_values[valid]
        event_values = event_values[valid]
        self.floor = float(floor)

        if time_values.size == 0:
            self.times = np.asarray([], dtype=np.float64)
            self.survival_after = np.asarray([], dtype=np.float64)
            return

        order = np.argsort(time_values, kind="mergesort")
        time_values = time_values[order]
        event_values = event_values[order].astype(np.int64)

        unique_time, first = np.unique(
            time_values,
            return_index=True,
        )
        at_risk = time_values.size - first
        censor_count = np.add.reduceat(event_values, first)
        factors = 1.0 - censor_count / np.maximum(at_risk, 1)

        self.times = unique_time
        self.survival_after = np.cumprod(
            np.clip(factors, 0.0, 1.0)
        )

    def _query(self, query, side):
        query_array = np.asarray(query, dtype=np.float64)
        original_shape = query_array.shape
        flat_query = query_array.reshape(-1)
        indices = (
            np.searchsorted(self.times, flat_query, side=side) - 1
        )
        result = np.ones_like(flat_query, dtype=np.float64)
        valid = indices >= 0
        result[valid] = self.survival_after[indices[valid]]
        result = np.maximum(result, self.floor)

        if original_shape == ():
            return float(result[0])
        return result.reshape(original_shape)

    def before(self, query):
        return self._query(query, side="left")

    def after(self, query):
        return self._query(query, side="right")


condition_censoring_kms_ci = []
condition_cindex_censoring_kms_ci = []
for condition_id_ci in range(n_conditions_ci):
    selected_train_ci = train_outcomes_ci["eligible"][
        :, condition_id_ci
    ]
    status_train_ci = train_outcomes_ci["status"][
        selected_train_ci,
        condition_id_ci,
    ]
    time_train_ci = train_outcomes_ci["exit_time"][
        selected_train_ci,
        condition_id_ci,
    ]
    condition_censoring_kms_ci.append(
        FrozenTrainingCensoringKM(
            exit_time=time_train_ci,
            censor_event=(status_train_ci == 0),
            floor=CENSOR_SURVIVAL_FLOOR_CI,
        )
    )
    condition_cindex_censoring_kms_ci.append(
        FrozenTrainingCensoringKM(
            exit_time=(
                train_outcomes_ci["exit_time_untruncated"][
                    selected_train_ci,
                    condition_id_ci,
                ]
            ),
            censor_event=(status_train_ci == 0),
            floor=CENSOR_SURVIVAL_FLOOR_CI,
        )
    )


# =============================================================================
# CANONICAL METRIC PRIMITIVES
# =============================================================================

def make_ci_binary_window(
    eligible,
    status,
    exit_time,
    horizon,
    censoring_km,
):
    eligible = np.asarray(eligible, dtype=bool)
    status = np.asarray(status, dtype=np.int8)
    exit_time = np.asarray(exit_time, dtype=np.float64)
    horizon = float(horizon)

    case = (
        eligible
        & (status == 1)
        & (exit_time <= horizon + TIME_EPS_CI)
    )
    competing = (
        eligible
        & (status == 2)
        & (exit_time <= horizon + TIME_EPS_CI)
    )
    observed_through = (
        eligible
        & (exit_time >= horizon - TIME_EPS_CI)
    )
    known = case | competing | observed_through
    y = case.astype(np.float64)

    weights = np.zeros_like(exit_time, dtype=np.float64)
    event_or_competing = case | competing
    if bool(event_or_competing.any()):
        weights[event_or_competing] = (
            1.0
            / censoring_km.before(exit_time[event_or_competing])
        )
    observed_control = observed_through & (~event_or_competing)
    if bool(observed_control.any()):
        weights[observed_control] = (
            1.0 / float(censoring_km.before(horizon))
        )

    weights[~np.isfinite(weights)] = 0.0
    return {
        "case": case,
        "competing": competing,
        "observed_through": observed_through,
        "known": known,
        "y": y,
        "weights": weights,
    }


def ci_ipcw_auc_auprc(window, prediction):
    prediction = np.asarray(prediction, dtype=np.float64)
    known = np.asarray(window["known"], dtype=bool)
    y = np.asarray(window["y"], dtype=np.float64)
    weights = np.asarray(window["weights"], dtype=np.float64)

    valid = (
        known
        & np.isfinite(prediction)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if (
        not bool(valid.any())
        or np.unique(y[valid].astype(np.int8)).size < 2
    ):
        return np.nan, np.nan

    auc = roc_auc_score(
        y[valid],
        prediction[valid],
        sample_weight=weights[valid],
    )
    auprc = average_precision_score(
        y[valid],
        prediction[valid],
        sample_weight=weights[valid],
    )
    return float(auc), float(auprc)


def ci_ipcw_ht_brier(y, prediction, weights, eligible_n):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.clip(
        np.asarray(prediction, dtype=np.float64),
        PROB_EPS_CI,
        1.0 - PROB_EPS_CI,
    )
    weights = np.asarray(weights, dtype=np.float64)

    valid = (
        np.isfinite(y)
        & np.isfinite(prediction)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if int(eligible_n) <= 0 or not bool(valid.any()):
        return np.nan

    return float(
        np.sum(
            weights[valid]
            * np.square(y[valid] - prediction[valid])
        )
        / float(int(eligible_n))
    )


def ci_aalen_johansen_cif(exit_time, status, horizon):
    """
    Aalen–Johansen cumulative incidence for condition status=1 with competing
    death status=2. Duplicated rows are retained, so a bootstrap sample is
    evaluated as its empirical resample.
    """

    time_values = np.asarray(exit_time, dtype=np.float64).reshape(-1)
    state_values = np.asarray(status, dtype=np.int8).reshape(-1)
    valid = np.isfinite(time_values) & (time_values >= 0.0)
    time_values = time_values[valid]
    state_values = state_values[valid]

    if time_values.size == 0:
        return np.nan

    order = np.argsort(time_values, kind="mergesort")
    time_values = time_values[order]
    state_values = state_values[order]

    unique_time, first = np.unique(
        time_values,
        return_index=True,
    )
    at_risk = time_values.size - first
    primary_events = np.add.reduceat(
        (state_values == 1).astype(np.int64),
        first,
    )
    all_events = np.add.reduceat(
        (state_values != 0).astype(np.int64),
        first,
    )

    event_mask = (
        (unique_time <= float(horizon) + TIME_EPS_CI)
        & (all_events > 0)
    )
    if not bool(event_mask.any()):
        return 0.0

    all_cause_hazard = all_events / np.maximum(at_risk, 1)
    survival_after = np.cumprod(
        np.clip(1.0 - all_cause_hazard, 0.0, 1.0)
    )
    survival_before = np.concatenate(
        [np.asarray([1.0]), survival_after[:-1]]
    )
    cif_increments = (
        survival_before
        * primary_events
        / np.maximum(at_risk, 1)
    )
    return float(np.sum(cif_increments[event_mask]))


def ci_integrate(values, horizons):
    value_array = np.asarray(values, dtype=np.float64)
    horizon_array = np.asarray(horizons, dtype=np.float64)
    valid = np.isfinite(value_array) & np.isfinite(horizon_array)
    if int(valid.sum()) < 2:
        return np.nan

    x = horizon_array[valid]
    y = value_array[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    span = float(x[-1] - x[0])
    if span <= 0.0:
        return np.nan

    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(y, x) / span)


# =============================================================================
# WOLBERS-STYLE TRUNCATED IPCW COMPETING-RISK C-INDEX
#
# Comparable pair types follow Wolbers et al., Biostatistics 2014:
# A: i has the event of interest by tau and T_i < T_j (or j is censored at
#    the same time), weighted by 1 / {G(T_i-) G(T_i)}.
# B: i has the event of interest by tau and j has a competing event at or
#    before T_i, weighted by 1 / {G(T_i-) G(T_j-)}.
# Prediction ties contribute one half.
# =============================================================================

@njit(cache=False, nogil=True)
def _ci_fenwick_add(tree, index, value):
    size = tree.shape[0]
    while index < size:
        tree[index] += value
        index += index & -index


@njit(cache=False, nogil=True)
def _ci_fenwick_sum(tree, index):
    total = 0.0
    while index > 0:
        total += tree[index]
        index -= index & -index
    return total


@njit(cache=False, nogil=True)
def _wolbers_cindex_core(
    exit_time,
    status,
    risk_rank,
    g_before,
    g_after,
    tau,
    n_risk_ranks,
):
    n_rows = exit_time.shape[0]
    order = np.argsort(exit_time)

    weighted_concordant = 0.0
    weighted_pairs = 0.0

    # Pair type A: later observed time, plus censoring tied at the case time.
    tree_a = np.zeros(n_risk_ranks + 1, dtype=np.float64)
    total_a = 0.0
    end = n_rows

    while end > 0:
        start = end - 1
        current_time = exit_time[order[start]]
        while (
            start > 0
            and exit_time[order[start - 1]] == current_time
        ):
            start -= 1

        # Same-time censored comparators are usable for pair type A.
        for position in range(start, end):
            row = order[position]
            if status[row] == 0:
                _ci_fenwick_add(
                    tree_a,
                    risk_rank[row],
                    1.0,
                )
                total_a += 1.0

        if current_time <= tau:
            for position in range(start, end):
                row = order[position]
                if status[row] == 1:
                    rank = risk_rank[row]
                    lower = _ci_fenwick_sum(tree_a, rank - 1)
                    through = _ci_fenwick_sum(tree_a, rank)
                    tied = through - lower
                    pair_weight = (
                        1.0 / (g_before[row] * g_after[row])
                    )
                    weighted_pairs += pair_weight * total_a
                    weighted_concordant += pair_weight * (
                        lower + 0.5 * tied
                    )

        # All non-censored observations at this time become later comparators
        # for cases at earlier times.
        for position in range(start, end):
            row = order[position]
            if status[row] != 0:
                _ci_fenwick_add(
                    tree_a,
                    risk_rank[row],
                    1.0,
                )
                total_a += 1.0

        end = start

    # Pair type B: competing event at or before the case time.
    tree_b = np.zeros(n_risk_ranks + 1, dtype=np.float64)
    total_b = 0.0
    start = 0

    while start < n_rows:
        end = start + 1
        current_time = exit_time[order[start]]
        while (
            end < n_rows
            and exit_time[order[end]] == current_time
        ):
            end += 1

        # Same-time competing events are usable for pair type B.
        for position in range(start, end):
            row = order[position]
            if status[row] == 2:
                comparator_weight = 1.0 / g_before[row]
                _ci_fenwick_add(
                    tree_b,
                    risk_rank[row],
                    comparator_weight,
                )
                total_b += comparator_weight

        if current_time <= tau:
            for position in range(start, end):
                row = order[position]
                if status[row] == 1:
                    rank = risk_rank[row]
                    lower = _ci_fenwick_sum(tree_b, rank - 1)
                    through = _ci_fenwick_sum(tree_b, rank)
                    tied = through - lower
                    case_weight = 1.0 / g_before[row]
                    weighted_pairs += case_weight * total_b
                    weighted_concordant += case_weight * (
                        lower + 0.5 * tied
                    )

        start = end

    if weighted_pairs <= 0.0:
        return np.nan, weighted_concordant, weighted_pairs

    return (
        weighted_concordant / weighted_pairs,
        weighted_concordant,
        weighted_pairs,
    )


def _wolbers_cindex_naive(
    exit_time,
    status,
    risk,
    g_before,
    g_after,
    tau,
):
    weighted_concordant = 0.0
    weighted_pairs = 0.0
    n_rows = int(len(exit_time))

    for i in range(n_rows):
        if status[i] != 1 or exit_time[i] > float(tau):
            continue

        for j in range(n_rows):
            if i == j:
                continue

            if (
                exit_time[i] < exit_time[j]
                or (
                    exit_time[i] == exit_time[j]
                    and status[j] == 0
                )
            ):
                weight = 1.0 / (
                    g_before[i] * g_after[i]
                )
            elif status[j] == 2:
                weight = 1.0 / (
                    g_before[i] * g_before[j]
                )
            else:
                continue

            weighted_pairs += weight
            if risk[i] > risk[j]:
                weighted_concordant += weight
            elif risk[i] == risk[j]:
                weighted_concordant += 0.5 * weight

    estimate = (
        weighted_concordant / weighted_pairs
        if weighted_pairs > 0.0
        else np.nan
    )
    return estimate, weighted_concordant, weighted_pairs


def ci_wolbers_cindex(
    exit_time,
    status,
    risk,
    eligible,
    censoring_km,
    tau,
):
    exit_time = np.asarray(exit_time, dtype=np.float64)
    status = np.asarray(status, dtype=np.int8)
    risk = np.asarray(risk, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)

    valid = (
        eligible
        & np.isfinite(exit_time)
        & np.isfinite(risk)
        & np.isin(status, [0, 1, 2])
    )
    exit_time = exit_time[valid]
    status = status[valid]
    risk = risk[valid]

    if exit_time.size < 2:
        return {
            "estimate": np.nan,
            "weighted_concordant": 0.0,
            "weighted_pairs": 0.0,
        }

    _, inverse_rank = np.unique(risk, return_inverse=True)
    risk_rank = inverse_rank.astype(np.int64) + 1
    n_risk_ranks = int(risk_rank.max())
    g_before = np.asarray(
        censoring_km.before(exit_time),
        dtype=np.float64,
    )
    g_after = np.asarray(
        censoring_km.after(exit_time),
        dtype=np.float64,
    )

    estimate, weighted_concordant, weighted_pairs = (
        _wolbers_cindex_core(
            exit_time=exit_time,
            status=status,
            risk_rank=risk_rank,
            g_before=g_before,
            g_after=g_after,
            tau=float(tau),
            n_risk_ranks=n_risk_ranks,
        )
    )
    return {
        "estimate": float(estimate),
        "weighted_concordant": float(weighted_concordant),
        "weighted_pairs": float(weighted_pairs),
    }


def validate_wolbers_implementation():
    test_time = np.asarray(
        [2.0, 4.0, 4.0, 5.0, 1.0, 3.0, 6.0, 2.0],
        dtype=np.float64,
    )
    test_status = np.asarray(
        [1, 0, 2, 1, 2, 0, 1, 1],
        dtype=np.int8,
    )
    test_risk = np.asarray(
        [0.8, 0.2, 0.2, 0.6, 0.1, 0.4, 0.6, 0.8],
        dtype=np.float64,
    )
    test_g_before = np.asarray(
        [0.95, 0.90, 0.90, 0.85, 1.00, 0.95, 0.80, 0.95],
        dtype=np.float64,
    )
    test_g_after = np.asarray(
        [0.95, 0.85, 0.85, 0.85, 1.00, 0.90, 0.80, 0.95],
        dtype=np.float64,
    )
    _, inverse_rank = np.unique(test_risk, return_inverse=True)

    fast = _wolbers_cindex_core(
        exit_time=test_time,
        status=test_status,
        risk_rank=inverse_rank.astype(np.int64) + 1,
        g_before=test_g_before,
        g_after=test_g_after,
        tau=5.0,
        n_risk_ranks=int(inverse_rank.max()) + 1,
    )
    slow = _wolbers_cindex_naive(
        exit_time=test_time,
        status=test_status,
        risk=test_risk,
        g_before=test_g_before,
        g_after=test_g_after,
        tau=5.0,
    )

    if not np.allclose(
        np.asarray(fast, dtype=np.float64),
        np.asarray(slow, dtype=np.float64),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError(
            "Optimized Wolbers C-index failed its naive-reference test: "
            f"fast={fast}, slow={slow}"
        )


validate_wolbers_implementation()

if not NUMBA_AVAILABLE:
    warnings.warn(
        "numba is unavailable. The exact C-index fallback is correct but may "
        "be slow. Install numba before the 2,000-replicate publication run.",
        RuntimeWarning,
    )
    if int(CI_BOOTSTRAP_N) >= 100:
        raise ImportError(
            "The publication bootstrap requires numba for the exact "
            "Wolbers C-index. Install numba in the notebook environment, "
            "restart the kernel if required, and rerun Cell 7B. For a small "
            "logic smoke test only, CI_BOOTSTRAP_N may be set below 100."
        )


# =============================================================================
# FIXED TRAIN-SPLIT NULL RISKS
# =============================================================================

train_null_risk_ci = np.full(
    (n_conditions_ci, n_dense_horizons_ci),
    np.nan,
    dtype=np.float64,
)

for condition_id_ci in selected_condition_ids_ci:
    selected_train_ci = train_outcomes_ci["eligible"][
        :, condition_id_ci
    ]
    train_time_ci = train_outcomes_ci["exit_time"][
        selected_train_ci,
        condition_id_ci,
    ]
    train_status_ci = train_outcomes_ci["status"][
        selected_train_ci,
        condition_id_ci,
    ]

    for grid_index_ci, horizon_ci in enumerate(dense_horizons_ci):
        train_null_risk_ci[
            condition_id_ci,
            grid_index_ci,
        ] = ci_aalen_johansen_cif(
            exit_time=train_time_ci,
            status=train_status_ci,
            horizon=float(horizon_ci),
        )


# =============================================================================
# ONE POINT/BOOTSTRAP SAMPLE
# =============================================================================

def ci_metric_row(
    bootstrap_id,
    condition_id,
    metric,
    estimate,
    horizon_months=np.nan,
    integration_start_months=np.nan,
    integration_end_months=np.nan,
    eligible_n=0,
    event_n=0,
    comparable_pair_weight=np.nan,
):
    return {
        "bootstrap_id": int(bootstrap_id),
        "model_name": MODEL_NAME_CI,
        "risk_type": "death_adjusted",
        "condition_id": int(condition_id),
        "condition_name": condition_names_ci[int(condition_id)],
        "metric": str(metric),
        "estimate": (
            float(estimate) if np.isfinite(estimate) else np.nan
        ),
        "horizon_months": (
            int(horizon_months)
            if np.isfinite(horizon_months)
            else np.nan
        ),
        "integration_start_months": (
            int(integration_start_months)
            if np.isfinite(integration_start_months)
            else np.nan
        ),
        "integration_end_months": (
            int(integration_end_months)
            if np.isfinite(integration_end_months)
            else np.nan
        ),
        "eligible_n": int(eligible_n),
        "event_n": int(event_n),
        "comparable_pair_weight": (
            float(comparable_pair_weight)
            if np.isfinite(comparable_pair_weight)
            else np.nan
        ),
    }


def evaluate_ci_sample(index, bootstrap_id):
    index = np.asarray(index, dtype=np.int64)
    rows = []

    for condition_id in selected_condition_ids_ci:
        eligible = test_outcomes_ci["eligible"][
            index,
            condition_id,
        ]
        status = test_outcomes_ci["status"][
            index,
            condition_id,
        ]
        exit_time = test_outcomes_ci["exit_time"][
            index,
            condition_id,
        ]
        exit_time_untruncated = test_outcomes_ci[
            "exit_time_untruncated"
        ][
            index,
            condition_id,
        ]
        prediction_by_horizon = dense_risk_ci[
            index,
            condition_id,
            :,
        ]
        censoring_km = condition_censoring_kms_ci[
            condition_id
        ]
        eligible_n = int(eligible.sum())

        # Fixed-horizon AUROC and AUPRC.
        for horizon in requested_horizons_ci:
            grid_index = dense_horizon_to_index_ci[horizon]
            prediction = prediction_by_horizon[:, grid_index]
            window = make_ci_binary_window(
                eligible=eligible,
                status=status,
                exit_time=exit_time,
                horizon=float(horizon),
                censoring_km=censoring_km,
            )
            auc, auprc = ci_ipcw_auc_auprc(
                window=window,
                prediction=prediction,
            )
            event_n = int(window["case"].sum())

            rows.append(
                ci_metric_row(
                    bootstrap_id=bootstrap_id,
                    condition_id=condition_id,
                    metric="auc_ipcw_cr",
                    estimate=auc,
                    horizon_months=horizon,
                    eligible_n=eligible_n,
                    event_n=event_n,
                )
            )
            rows.append(
                ci_metric_row(
                    bootstrap_id=bootstrap_id,
                    condition_id=condition_id,
                    metric="auprc_ipcw_cr",
                    estimate=auprc,
                    horizon_months=horizon,
                    eligible_n=eligible_n,
                    event_n=event_n,
                )
            )

        # Dense-grid Brier and calibration components.
        brier_values = []
        null_brier_values = []
        calibration_values = []

        for grid_index, horizon in enumerate(dense_horizons_ci):
            prediction = prediction_by_horizon[:, grid_index]
            window = make_ci_binary_window(
                eligible=eligible,
                status=status,
                exit_time=exit_time,
                horizon=float(horizon),
                censoring_km=censoring_km,
            )
            brier = ci_ipcw_ht_brier(
                y=window["y"],
                prediction=prediction,
                weights=window["weights"],
                eligible_n=eligible_n,
            )

            null_prediction = np.full_like(
                prediction,
                train_null_risk_ci[condition_id, grid_index],
                dtype=np.float64,
            )
            null_brier = ci_ipcw_ht_brier(
                y=window["y"],
                prediction=null_prediction,
                weights=window["weights"],
                eligible_n=eligible_n,
            )

            observed_risk = ci_aalen_johansen_cif(
                exit_time=exit_time[eligible],
                status=status[eligible],
                horizon=float(horizon),
            )
            mean_prediction = (
                float(np.mean(prediction[eligible]))
                if eligible_n > 0
                else np.nan
            )
            absolute_calibration_error = (
                abs(mean_prediction - observed_risk)
                if (
                    np.isfinite(mean_prediction)
                    and np.isfinite(observed_risk)
                )
                else np.nan
            )

            brier_values.append(brier)
            null_brier_values.append(null_brier)
            calibration_values.append(
                absolute_calibration_error
            )

        ibs = ci_integrate(brier_values, dense_horizons_ci)
        ibs_null = ci_integrate(
            null_brier_values,
            dense_horizons_ci,
        )
        scaled_ibs = (
            1.0 - ibs / ibs_null
            if (
                np.isfinite(ibs)
                and np.isfinite(ibs_null)
                and ibs_null > 0.0
            )
            else np.nan
        )
        integrated_abs_calibration_error = ci_integrate(
            calibration_values,
            dense_horizons_ci,
        )

        end_horizon = int(dense_horizons_ci[-1])
        event_n_end = int(
            (
                eligible
                & (status == 1)
                & (
                    exit_time
                    <= end_horizon + TIME_EPS_CI
                )
            ).sum()
        )

        for metric, estimate in [
            ("ibs_ipcw_ht", ibs),
            ("ibs_null_ipcw_ht", ibs_null),
            ("scaled_ibs_ipcw_ht", scaled_ibs),
            (
                "integrated_abs_calibration_error",
                integrated_abs_calibration_error,
            ),
        ]:
            rows.append(
                ci_metric_row(
                    bootstrap_id=bootstrap_id,
                    condition_id=condition_id,
                    metric=metric,
                    estimate=estimate,
                    integration_start_months=dense_horizons_ci[0],
                    integration_end_months=dense_horizons_ci[-1],
                    eligible_n=eligible_n,
                    event_n=event_n_end,
                )
            )

        # Genuine truncated competing-risk concordance.
        cindex_tau = int(CI_CINDEX_TAU_MONTHS)
        cindex_grid_index = dense_horizon_to_index_ci[
            cindex_tau
        ]
        cindex_result = ci_wolbers_cindex(
            exit_time=exit_time_untruncated,
            status=status,
            risk=prediction_by_horizon[:, cindex_grid_index],
            eligible=eligible,
            censoring_km=(
                condition_cindex_censoring_kms_ci[
                    condition_id
                ]
            ),
            tau=float(cindex_tau),
        )
        cindex_event_n = int(
            (
                eligible
                & (status == 1)
                & (
                    exit_time_untruncated
                    <= cindex_tau + TIME_EPS_CI
                )
            ).sum()
        )
        rows.append(
            ci_metric_row(
                bootstrap_id=bootstrap_id,
                condition_id=condition_id,
                metric="cindex_ipcw_cr",
                estimate=cindex_result["estimate"],
                horizon_months=cindex_tau,
                eligible_n=eligible_n,
                event_n=cindex_event_n,
                comparable_pair_weight=(
                    cindex_result["weighted_pairs"]
                ),
            )
        )

    return pd.DataFrame(rows)


# =============================================================================
# LOCKED POINT ESTIMATES AND CROSS-CHECK AGAINST EXISTING TABLES
# =============================================================================

print("=" * 88)
print("CELL 7B — participant-bootstrap confidence intervals")
print("=" * 88)
print(
    {
        "model_name": MODEL_NAME_CI,
        "test_participants": n_test_ci,
        "conditions_selected": len(selected_condition_ids_ci),
        "bootstrap_replicates": int(CI_BOOTSTRAP_N),
        "point_horizons_months": requested_horizons_ci,
        "cindex_tau_months": int(CI_CINDEX_TAU_MONTHS),
        "dense_horizons_months": dense_horizons_ci,
        "n_jobs": int(CI_N_JOBS),
        "numba_available": bool(NUMBA_AVAILABLE),
    }
)

cell7b_start_time = time.perf_counter()

point_metrics_ci = evaluate_ci_sample(
    index=np.arange(n_test_ci, dtype=np.int64),
    bootstrap_id=-1,
)


def _point_lookup(frame, metric, condition_id, horizon=np.nan):
    selected = (
        frame["metric"].astype(str).eq(str(metric))
        & frame["condition_id"].eq(int(condition_id))
    )
    if np.isfinite(horizon):
        selected &= frame["horizon_months"].eq(int(horizon))
    else:
        selected &= frame["horizon_months"].isna()

    values = frame.loc[selected, "estimate"]
    if len(values) != 1:
        raise AssertionError(
            "Point metric lookup was not one-to-one for "
            f"metric={metric}, condition={condition_id}, horizon={horizon}."
        )
    return float(values.iloc[0])


dense_point_reference_ci = pd.read_csv(DENSE_POINT_METRICS_PATH)
integrated_point_reference_ci = pd.read_csv(
    INTEGRATED_POINT_METRICS_PATH
)

if "risk_type" in dense_point_reference_ci.columns:
    dense_point_reference_ci = dense_point_reference_ci[
        dense_point_reference_ci["risk_type"]
        .astype(str)
        .eq("death_adjusted")
    ].copy()
if "risk_type" in integrated_point_reference_ci.columns:
    integrated_point_reference_ci = (
        integrated_point_reference_ci[
            integrated_point_reference_ci["risk_type"]
            .astype(str)
            .eq("death_adjusted")
        ].copy()
    )

validation_differences_ci = []

for condition_id_ci in selected_condition_ids_ci:
    for horizon_ci in requested_horizons_ci:
        reference_row_ci = dense_point_reference_ci[
            dense_point_reference_ci["condition_id"].eq(
                int(condition_id_ci)
            )
            & dense_point_reference_ci["horizon_months"].eq(
                int(horizon_ci)
            )
        ]
        if len(reference_row_ci) != 1:
            raise AssertionError(
                "Dense point-reference lookup was not one-to-one for "
                f"condition={condition_id_ci}, horizon={horizon_ci}."
            )

        for metric_ci, reference_column_ci in [
            ("auc_ipcw_cr", "auc_ipcw_cr"),
            ("auprc_ipcw_cr", "auprc_ipcw_cr"),
        ]:
            calculated_ci = _point_lookup(
                point_metrics_ci,
                metric_ci,
                condition_id_ci,
                horizon_ci,
            )
            reference_ci = float(
                reference_row_ci[reference_column_ci].iloc[0]
            )
            if np.isfinite(calculated_ci) and np.isfinite(reference_ci):
                difference_ci = abs(calculated_ci - reference_ci)
                validation_differences_ci.append(difference_ci)
                if not np.isclose(
                    calculated_ci,
                    reference_ci,
                    rtol=5e-5,
                    atol=1e-7,
                ):
                    raise AssertionError(
                        "Cell 7B discrimination point estimate does not "
                        "match the canonical dense table: "
                        f"condition={condition_id_ci}, "
                        f"horizon={horizon_ci}, metric={metric_ci}, "
                        f"calculated={calculated_ci}, "
                        f"reference={reference_ci}."
                    )
            elif not (
                np.isnan(calculated_ci) and np.isnan(reference_ci)
            ):
                raise AssertionError(
                    "Cell 7B and canonical dense table disagree on a "
                    "finite/non-finite point estimate."
                )

    reference_integrated_row_ci = (
        integrated_point_reference_ci[
            integrated_point_reference_ci["condition_id"].eq(
                int(condition_id_ci)
            )
        ]
    )
    if len(reference_integrated_row_ci) != 1:
        raise AssertionError(
            "Integrated point-reference lookup was not one-to-one for "
            f"condition={condition_id_ci}."
        )

    for metric_ci, reference_column_ci in [
        ("ibs_ipcw_ht", "ibs_ipcw_ht"),
        ("ibs_null_ipcw_ht", "ibs_null_ipcw_ht"),
        ("scaled_ibs_ipcw_ht", "scaled_ibs_ipcw_ht"),
        (
            "integrated_abs_calibration_error",
            "integrated_abs_calibration_error",
        ),
    ]:
        calculated_ci = _point_lookup(
            point_metrics_ci,
            metric_ci,
            condition_id_ci,
            np.nan,
        )
        reference_ci = float(
            reference_integrated_row_ci[
                reference_column_ci
            ].iloc[0]
        )
        if np.isfinite(calculated_ci) and np.isfinite(reference_ci):
            difference_ci = abs(calculated_ci - reference_ci)
            validation_differences_ci.append(difference_ci)
            if not np.isclose(
                calculated_ci,
                reference_ci,
                rtol=5e-5,
                atol=1e-7,
            ):
                raise AssertionError(
                    "Cell 7B integrated point estimate does not match "
                    "the canonical integrated table: "
                    f"condition={condition_id_ci}, metric={metric_ci}, "
                    f"calculated={calculated_ci}, "
                    f"reference={reference_ci}."
                )
        elif not (
            np.isnan(calculated_ci) and np.isnan(reference_ci)
        ):
            raise AssertionError(
                "Cell 7B and canonical integrated table disagree on a "
                "finite/non-finite point estimate."
            )

max_point_validation_difference_ci = (
    float(max(validation_differences_ci))
    if validation_differences_ci
    else 0.0
)
print(
    "Point-estimate validation passed; maximum absolute difference:",
    max_point_validation_difference_ci,
)


# =============================================================================
# REPRODUCIBLE, RESUMABLE PARTICIPANT BOOTSTRAP
# =============================================================================

def sha256_text_ci(values):
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


eid_axis_sha256_ci = sha256_text_ci(test_eids_ci)
condition_axis_sha256_ci = sha256_text_ci(condition_names_ci)

configuration_ci = {
    "schema_version": (
        "crescent-participant-bootstrap-ci.v3"
    ),
    "implementation_version": (
        "2026-09-30-delayed-entry-contract-v3"
    ),
    "model_name": MODEL_NAME_CI,
    "bootstrap_n": int(CI_BOOTSTRAP_N),
    "bootstrap_seed": int(CI_BOOTSTRAP_SEED),
    "alpha": float(CI_ALPHA),
    "ci_method": "percentile",
    "resampling_unit": "eid",
    "point_horizons_months": list(requested_horizons_ci),
    "cindex_tau_months": int(CI_CINDEX_TAU_MONTHS),
    "dense_horizons_months": list(dense_horizons_ci),
    "selected_condition_ids": list(selected_condition_ids_ci),
    "entry_delay_months": float(entry_delay_months_ci),
    "test_n": int(n_test_ci),
    "condition_n": int(n_conditions_ci),
    "prediction_sha256": sha256_file(DENSE_PREDICTION_PATH),
    "training_samples_sha256": sha256_file(TRAIN_SAMPLES_PATH),
    "test_samples_sha256": sha256_file(TEST_SAMPLES_PATH),
    "censor_survival_floor": float(CENSOR_SURVIVAL_FLOOR_CI),
    "delayed_entry_contract": delayed_entry_contract(),
    "eid_axis_sha256": eid_axis_sha256_ci,
    "condition_axis_sha256": condition_axis_sha256_ci,
    "censoring_source": "canonical_train_samples.pkl",
    "censoring_frozen_across_bootstraps": True,
    "model_frozen_across_bootstraps": True,
    "integrated_auroc_included": False,
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
}
configuration_json_ci = json.dumps(
    configuration_ci,
    sort_keys=True,
    separators=(",", ":"),
)
configuration_signature_ci = hashlib.sha256(
    configuration_json_ci.encode("utf-8")
).hexdigest()[:16]

chunk_directory_ci = (
    RUN_DIR
    / f"test_bootstrap_ci_chunks_{configuration_signature_ci}"
)
chunk_directory_ci.mkdir(parents=True, exist_ok=True)

chunk_configuration_path_ci = (
    chunk_directory_ci / "configuration.json"
)


def write_json_atomic_ci(payload, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    temporary.replace(path)


if chunk_configuration_path_ci.is_file():
    with open(
        chunk_configuration_path_ci,
        "r",
        encoding="utf-8",
    ) as handle:
        existing_configuration_ci = json.load(handle)
    if existing_configuration_ci != configuration_ci:
        raise RuntimeError(
            "Existing bootstrap chunks have a different configuration. "
            "Do not combine them."
        )
else:
    write_json_atomic_ci(
        configuration_ci,
        chunk_configuration_path_ci,
    )


def evaluate_bootstrap_replicate_ci(bootstrap_id):
    # This seed rule can be repeated for comparator models after their eid
    # order has been aligned and verified by eid_axis_sha256.
    seed_sequence = np.random.SeedSequence(
        [int(CI_BOOTSTRAP_SEED), int(bootstrap_id)]
    )
    rng = np.random.default_rng(seed_sequence)
    bootstrap_index = rng.integers(
        low=0,
        high=n_test_ci,
        size=n_test_ci,
        dtype=np.int64,
    )
    return evaluate_ci_sample(
        index=bootstrap_index,
        bootstrap_id=int(bootstrap_id),
    )


def read_valid_chunk_ci(path, expected_ids):
    if not Path(path).is_file():
        return None
    frame = pd.read_csv(path)
    observed_ids = sorted(
        frame["bootstrap_id"].astype(int).unique().tolist()
    )
    if observed_ids != sorted(int(value) for value in expected_ids):
        raise RuntimeError(
            f"Bootstrap chunk has unexpected replicate IDs: {path}"
        )
    expected_rows = (
        len(expected_ids)
        * len(selected_condition_ids_ci)
        * (
            2 * len(requested_horizons_ci)
            + 4
            + 1
        )
    )
    if len(frame) != expected_rows:
        raise RuntimeError(
            f"Bootstrap chunk has {len(frame)} rows; "
            f"expected {expected_rows}: {path}"
        )
    return frame


bootstrap_chunk_frames_ci = []

for chunk_start_ci in range(
    0,
    int(CI_BOOTSTRAP_N),
    int(CI_BOOTSTRAP_CHUNK_SIZE),
):
    chunk_stop_ci = min(
        chunk_start_ci + int(CI_BOOTSTRAP_CHUNK_SIZE),
        int(CI_BOOTSTRAP_N),
    )
    chunk_ids_ci = list(range(chunk_start_ci, chunk_stop_ci))
    chunk_path_ci = (
        chunk_directory_ci
        / (
            f"bootstrap_{chunk_start_ci:05d}_"
            f"{chunk_stop_ci - 1:05d}.csv.gz"
        )
    )

    existing_chunk_ci = None
    if not bool(CI_FORCE_RECOMPUTE):
        existing_chunk_ci = read_valid_chunk_ci(
            chunk_path_ci,
            chunk_ids_ci,
        )

    if existing_chunk_ci is not None:
        bootstrap_chunk_frames_ci.append(existing_chunk_ci)
        print(
            f"Reused bootstrap replicates "
            f"{chunk_start_ci + 1}–{chunk_stop_ci} "
            f"of {CI_BOOTSTRAP_N}."
        )
        continue

    if JOBLIB_AVAILABLE and int(CI_N_JOBS) > 1:
        results_ci = Parallel(
            n_jobs=int(CI_N_JOBS),
            prefer="threads",
        )(
            delayed(evaluate_bootstrap_replicate_ci)(
                bootstrap_id_ci
            )
            for bootstrap_id_ci in chunk_ids_ci
        )
    else:
        results_ci = [
            evaluate_bootstrap_replicate_ci(
                bootstrap_id_ci
            )
            for bootstrap_id_ci in chunk_ids_ci
        ]

    chunk_frame_ci = pd.concat(
        results_ci,
        ignore_index=True,
    )
    expected_rows_ci = (
        len(chunk_ids_ci)
        * len(selected_condition_ids_ci)
        * (
            2 * len(requested_horizons_ci)
            + 4
            + 1
        )
    )
    if len(chunk_frame_ci) != expected_rows_ci:
        raise RuntimeError(
            "Bootstrap worker returned an unexpected number of rows: "
            f"{len(chunk_frame_ci)} != {expected_rows_ci}."
        )

    temporary_chunk_path_ci = chunk_path_ci.with_name(
        chunk_path_ci.name + ".tmp"
    )
    chunk_frame_ci.to_csv(
        temporary_chunk_path_ci,
        index=False,
        compression="gzip",
    )
    temporary_chunk_path_ci.replace(chunk_path_ci)
    bootstrap_chunk_frames_ci.append(chunk_frame_ci)

    print(
        f"Completed bootstrap replicates "
        f"{chunk_start_ci + 1}–{chunk_stop_ci} "
        f"of {CI_BOOTSTRAP_N}."
    )

bootstrap_replicates_ci = pd.concat(
    bootstrap_chunk_frames_ci,
    ignore_index=True,
)

observed_bootstrap_ids_ci = sorted(
    bootstrap_replicates_ci[
        "bootstrap_id"
    ].astype(int).unique().tolist()
)
expected_bootstrap_ids_ci = list(
    range(int(CI_BOOTSTRAP_N))
)
if observed_bootstrap_ids_ci != expected_bootstrap_ids_ci:
    raise RuntimeError(
        "Combined bootstrap table does not contain the exact requested "
        "replicate ID set."
    )


# =============================================================================
# PER-CONDITION PERCENTILE CONFIDENCE INTERVALS
# =============================================================================

key_columns_ci = [
    "condition_id",
    "condition_name",
    "metric",
    "horizon_key",
    "integration_start_key",
    "integration_end_key",
]


def add_ci_group_keys(frame):
    frame = frame.copy()
    frame["horizon_key"] = (
        pd.to_numeric(frame["horizon_months"], errors="coerce")
        .fillna(-1)
        .astype(int)
    )
    frame["integration_start_key"] = (
        pd.to_numeric(
            frame["integration_start_months"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )
    frame["integration_end_key"] = (
        pd.to_numeric(
            frame["integration_end_months"],
            errors="coerce",
        )
        .fillna(-1)
        .astype(int)
    )
    return frame


point_keyed_ci = add_ci_group_keys(point_metrics_ci)
replicate_keyed_ci = add_ci_group_keys(
    bootstrap_replicates_ci
)

grouped_replicates_ci = (
    replicate_keyed_ci
    .groupby(key_columns_ci, sort=False)["estimate"]
    .agg(
        n_bootstrap_total="size",
        n_valid_bootstrap="count",
        ci_lower=lambda values: values.quantile(
            float(CI_ALPHA) / 2.0
        ),
        ci_upper=lambda values: values.quantile(
            1.0 - float(CI_ALPHA) / 2.0
        ),
    )
    .reset_index()
)

point_columns_for_merge_ci = (
    key_columns_ci
    + [
        "model_name",
        "risk_type",
        "estimate",
        "horizon_months",
        "integration_start_months",
        "integration_end_months",
        "eligible_n",
        "event_n",
        "comparable_pair_weight",
    ]
)
per_condition_ci = point_keyed_ci[
    point_columns_for_merge_ci
].merge(
    grouped_replicates_ci,
    on=key_columns_ci,
    how="left",
    validate="one_to_one",
)

per_condition_ci["valid_bootstrap_fraction"] = (
    per_condition_ci["n_valid_bootstrap"]
    / per_condition_ci["n_bootstrap_total"]
)
per_condition_ci["ci_level"] = 1.0 - float(CI_ALPHA)
per_condition_ci["ci_method"] = (
    "participant_bootstrap_percentile"
)
per_condition_ci["ci_status"] = np.where(
    per_condition_ci["valid_bootstrap_fraction"]
    >= float(CI_MIN_VALID_BOOTSTRAP_FRACTION),
    "ok",
    "insufficient_valid_replicates",
)

insufficient_ci = per_condition_ci["ci_status"].ne("ok")
per_condition_ci.loc[
    insufficient_ci,
    ["ci_lower", "ci_upper"],
] = np.nan

per_condition_ci = per_condition_ci.drop(
    columns=[
        "horizon_key",
        "integration_start_key",
        "integration_end_key",
    ]
)


# =============================================================================
# MACRO ESTIMATES OVER A FIXED POINT-VALID CONDITION SET
# =============================================================================

macro_key_columns_ci = [
    "metric",
    "horizon_key",
    "integration_start_key",
    "integration_end_key",
]

macro_point_rows_ci = []
macro_replicate_rows_ci = []

for macro_key_ci, point_group_ci in point_keyed_ci.groupby(
    macro_key_columns_ci,
    sort=False,
):
    (
        macro_metric_ci,
        macro_horizon_key_ci,
        macro_start_key_ci,
        macro_end_key_ci,
    ) = macro_key_ci

    point_valid_ci = point_group_ci[
        np.isfinite(point_group_ci["estimate"])
    ].copy()
    fixed_condition_ids_ci = sorted(
        point_valid_ci["condition_id"].astype(int).tolist()
    )
    fixed_condition_n_ci = int(len(fixed_condition_ids_ci))

    if fixed_condition_n_ci == 0:
        macro_point_estimate_ci = np.nan
    else:
        macro_point_estimate_ci = float(
            point_valid_ci["estimate"].mean()
        )

    macro_point_rows_ci.append(
        {
            "model_name": MODEL_NAME_CI,
            "risk_type": "death_adjusted",
            "metric": str(macro_metric_ci),
            "horizon_months": (
                int(macro_horizon_key_ci)
                if int(macro_horizon_key_ci) >= 0
                else np.nan
            ),
            "integration_start_months": (
                int(macro_start_key_ci)
                if int(macro_start_key_ci) >= 0
                else np.nan
            ),
            "integration_end_months": (
                int(macro_end_key_ci)
                if int(macro_end_key_ci) >= 0
                else np.nan
            ),
            "estimate": macro_point_estimate_ci,
            "fixed_condition_n": fixed_condition_n_ci,
            "fixed_condition_ids": "|".join(
                str(value) for value in fixed_condition_ids_ci
            ),
        }
    )

    replicate_group_ci = replicate_keyed_ci[
        replicate_keyed_ci["metric"].astype(str).eq(
            str(macro_metric_ci)
        )
        & replicate_keyed_ci["horizon_key"].eq(
            int(macro_horizon_key_ci)
        )
        & replicate_keyed_ci["integration_start_key"].eq(
            int(macro_start_key_ci)
        )
        & replicate_keyed_ci["integration_end_key"].eq(
            int(macro_end_key_ci)
        )
        & replicate_keyed_ci["condition_id"].isin(
            fixed_condition_ids_ci
        )
    ].copy()

    required_finite_condition_n_ci = (
        int(
            math.ceil(
                float(CI_MACRO_MIN_CONDITION_FRACTION)
                * fixed_condition_n_ci
            )
        )
        if fixed_condition_n_ci > 0
        else 1
    )

    if fixed_condition_n_ci == 0:
        grouped_macro_replicates_ci = [
            (bootstrap_id_ci, None)
            for bootstrap_id_ci in range(int(CI_BOOTSTRAP_N))
        ]
    else:
        grouped_macro_replicates_ci = list(
            replicate_group_ci.groupby(
                "bootstrap_id",
                sort=False,
            )
        )

    for bootstrap_id_ci, bootstrap_group_ci in (
        grouped_macro_replicates_ci
    ):
        if bootstrap_group_ci is None:
            finite_values_ci = pd.Series([], dtype=float)
        else:
            finite_values_ci = bootstrap_group_ci.loc[
                np.isfinite(bootstrap_group_ci["estimate"]),
                "estimate",
            ]
        finite_condition_n_ci = int(len(finite_values_ci))
        macro_estimate_ci = (
            float(finite_values_ci.mean())
            if (
                finite_condition_n_ci
                >= required_finite_condition_n_ci
            )
            else np.nan
        )
        macro_replicate_rows_ci.append(
            {
                "bootstrap_id": int(bootstrap_id_ci),
                "model_name": MODEL_NAME_CI,
                "risk_type": "death_adjusted",
                "aggregation_level": "macro",
                "condition_id": -1,
                "condition_name": "MACRO",
                "metric": str(macro_metric_ci),
                "estimate": macro_estimate_ci,
                "horizon_months": (
                    int(macro_horizon_key_ci)
                    if int(macro_horizon_key_ci) >= 0
                    else np.nan
                ),
                "integration_start_months": (
                    int(macro_start_key_ci)
                    if int(macro_start_key_ci) >= 0
                    else np.nan
                ),
                "integration_end_months": (
                    int(macro_end_key_ci)
                    if int(macro_end_key_ci) >= 0
                    else np.nan
                ),
                "fixed_condition_n": fixed_condition_n_ci,
                "finite_condition_n": finite_condition_n_ci,
            }
        )

macro_point_ci = pd.DataFrame(macro_point_rows_ci)
macro_replicates_ci = pd.DataFrame(
    macro_replicate_rows_ci
)

macro_summary_rows_ci = []
for macro_row_ci in macro_point_ci.itertuples(index=False):
    selected_macro_replicates_ci = macro_replicates_ci[
        macro_replicates_ci["metric"].astype(str).eq(
            str(macro_row_ci.metric)
        )
        & (
            pd.to_numeric(
                macro_replicates_ci["horizon_months"],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
            == (
                int(macro_row_ci.horizon_months)
                if np.isfinite(macro_row_ci.horizon_months)
                else -1
            )
        )
        & (
            pd.to_numeric(
                macro_replicates_ci[
                    "integration_start_months"
                ],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
            == (
                int(macro_row_ci.integration_start_months)
                if np.isfinite(
                    macro_row_ci.integration_start_months
                )
                else -1
            )
        )
        & (
            pd.to_numeric(
                macro_replicates_ci[
                    "integration_end_months"
                ],
                errors="coerce",
            )
            .fillna(-1)
            .astype(int)
            == (
                int(macro_row_ci.integration_end_months)
                if np.isfinite(
                    macro_row_ci.integration_end_months
                )
                else -1
            )
        )
    ]["estimate"]

    valid_macro_ci = selected_macro_replicates_ci[
        np.isfinite(selected_macro_replicates_ci)
    ]
    n_valid_macro_ci = int(len(valid_macro_ci))
    valid_fraction_macro_ci = (
        n_valid_macro_ci / int(CI_BOOTSTRAP_N)
    )
    macro_status_ci = (
        "ok"
        if valid_fraction_macro_ci
        >= float(CI_MIN_VALID_BOOTSTRAP_FRACTION)
        else "insufficient_valid_replicates"
    )

    macro_summary_rows_ci.append(
        {
            **macro_row_ci._asdict(),
            "ci_lower": (
                float(
                    valid_macro_ci.quantile(
                        float(CI_ALPHA) / 2.0
                    )
                )
                if macro_status_ci == "ok"
                else np.nan
            ),
            "ci_upper": (
                float(
                    valid_macro_ci.quantile(
                        1.0 - float(CI_ALPHA) / 2.0
                    )
                )
                if macro_status_ci == "ok"
                else np.nan
            ),
            "n_bootstrap_total": int(CI_BOOTSTRAP_N),
            "n_valid_bootstrap": n_valid_macro_ci,
            "valid_bootstrap_fraction": (
                valid_fraction_macro_ci
            ),
            "ci_level": 1.0 - float(CI_ALPHA),
            "ci_method": (
                "participant_bootstrap_percentile"
            ),
            "ci_status": macro_status_ci,
            "minimum_condition_fraction_per_replicate": (
                float(CI_MACRO_MIN_CONDITION_FRACTION)
            ),
        }
    )

macro_ci = pd.DataFrame(macro_summary_rows_ci)


# =============================================================================
# SAVE FINAL OUTPUTS
# =============================================================================

PER_CONDITION_CI_PATH = (
    RUN_DIR / "test_per_condition_metrics_with_ci.csv"
)
POINT_HORIZON_CI_PATH = (
    RUN_DIR / "test_point_horizon_metrics_with_ci.csv"
)
CINDEX_CI_PATH = (
    RUN_DIR / "test_cindex_by_condition_with_ci.csv"
)
INTEGRATED_CI_PATH = (
    RUN_DIR / "test_integrated_metrics_with_ci.csv"
)
MACRO_CI_PATH = (
    RUN_DIR / "test_macro_metrics_with_ci.csv"
)
BOOTSTRAP_REPLICATES_PATH = (
    RUN_DIR / "test_bootstrap_replicates_metrics.csv.gz"
)
BOOTSTRAP_METADATA_PATH = (
    RUN_DIR / "test_bootstrap_ci_metadata.json"
)


def write_csv_atomic_ci(frame, path, compression=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        compression=compression,
    )
    temporary.replace(path)


point_horizon_ci = per_condition_ci[
    per_condition_ci["metric"].isin(
        ["auc_ipcw_cr", "auprc_ipcw_cr"]
    )
].copy()
cindex_ci = per_condition_ci[
    per_condition_ci["metric"].eq("cindex_ipcw_cr")
].copy()
integrated_ci = per_condition_ci[
    per_condition_ci["metric"].isin(
        [
            "ibs_ipcw_ht",
            "ibs_null_ipcw_ht",
            "scaled_ibs_ipcw_ht",
            "integrated_abs_calibration_error",
        ]
    )
].copy()

write_csv_atomic_ci(
    per_condition_ci,
    PER_CONDITION_CI_PATH,
)
write_csv_atomic_ci(
    point_horizon_ci,
    POINT_HORIZON_CI_PATH,
)
write_csv_atomic_ci(
    cindex_ci,
    CINDEX_CI_PATH,
)
write_csv_atomic_ci(
    integrated_ci,
    INTEGRATED_CI_PATH,
)
write_csv_atomic_ci(
    macro_ci,
    MACRO_CI_PATH,
)

replicate_export_ci = bootstrap_replicates_ci.copy()
replicate_export_ci.insert(
    3,
    "aggregation_level",
    "condition",
)

macro_replicate_export_ci = macro_replicates_ci.copy()
replicate_export_columns_ci = list(replicate_export_ci.columns)
for column_ci in macro_replicate_export_ci.columns:
    if column_ci not in replicate_export_columns_ci:
        replicate_export_columns_ci.append(column_ci)

for missing_column_ci in (
    set(replicate_export_columns_ci)
    - set(replicate_export_ci.columns)
):
    replicate_export_ci[missing_column_ci] = np.nan
for missing_column_ci in (
    set(replicate_export_columns_ci)
    - set(macro_replicate_export_ci.columns)
):
    macro_replicate_export_ci[missing_column_ci] = np.nan
macro_replicate_export_ci = macro_replicate_export_ci[
    replicate_export_columns_ci
]
replicate_export_ci = replicate_export_ci[
    replicate_export_columns_ci
]

replicate_export_ci = pd.concat(
    [
        replicate_export_ci,
        macro_replicate_export_ci,
    ],
    ignore_index=True,
)
write_csv_atomic_ci(
    replicate_export_ci,
    BOOTSTRAP_REPLICATES_PATH,
    compression="gzip",
)

elapsed_seconds_ci = float(
    time.perf_counter() - cell7b_start_time
)
metadata_ci = {
    **configuration_ci,
    "status": "completed",
    "configuration_signature": configuration_signature_ci,
    "bootstrap_seed_rule": (
        "default_rng(SeedSequence([bootstrap_seed, bootstrap_id]))"
        ".integers(0, test_n, size=test_n)"
    ),
    "point_estimate_validation": {
        "passed": True,
        "maximum_absolute_difference": (
            max_point_validation_difference_ci
        ),
        "reference_dense_point_file": str(
            DENSE_POINT_METRICS_PATH
        ),
        "reference_integrated_point_file": str(
            INTEGRATED_POINT_METRICS_PATH
        ),
    },
    "cindex_definition": {
        "name": (
            "Wolbers-style truncated IPCW competing-risk C-index"
        ),
        "tau_months_after_entry": int(
            CI_CINDEX_TAU_MONTHS
        ),
        "prediction_time_months_after_entry": int(
            CI_CINDEX_TAU_MONTHS
        ),
        "prediction": "death-adjusted condition CIF",
        "censoring_model": (
            "condition-specific marginal KM from canonical train split "
            "using untruncated observed follow-up times"
        ),
        "prediction_ties": "counted_as_one_half",
        "reference": (
            "Wolbers M et al. Biostatistics. 2014;15(3):526-539. "
            "doi:10.1093/biostatistics/kxt059"
        ),
    },
    "calibration_definition": {
        "observed_risk": (
            "Aalen-Johansen CIF recomputed within each test bootstrap"
        ),
        "error": (
            "absolute difference between bootstrap mean predicted CIF "
            "and bootstrap Aalen-Johansen CIF at each dense horizon"
        ),
        "integration": (
            "trapezoidal mean over dense horizon grid"
        ),
    },
    "validity_thresholds": {
        "minimum_valid_bootstrap_fraction": float(
            CI_MIN_VALID_BOOTSTRAP_FRACTION
        ),
        "macro_minimum_condition_fraction_per_replicate": float(
            CI_MACRO_MIN_CONDITION_FRACTION
        ),
    },
    "runtime": {
        "elapsed_seconds": elapsed_seconds_ci,
        "n_jobs": int(CI_N_JOBS),
        "joblib_available": bool(JOBLIB_AVAILABLE),
        "numba_available": bool(NUMBA_AVAILABLE),
        "chunk_size": int(CI_BOOTSTRAP_CHUNK_SIZE),
        "chunk_directory": str(chunk_directory_ci),
    },
    "output_files": {
        "per_condition_ci": str(PER_CONDITION_CI_PATH),
        "point_horizon_ci": str(POINT_HORIZON_CI_PATH),
        "cindex_ci": str(CINDEX_CI_PATH),
        "integrated_ci": str(INTEGRATED_CI_PATH),
        "macro_ci": str(MACRO_CI_PATH),
        "bootstrap_replicates": str(
            BOOTSTRAP_REPLICATES_PATH
        ),
        "metadata": str(BOOTSTRAP_METADATA_PATH),
    },
}
write_json_atomic_ci(metadata_ci, BOOTSTRAP_METADATA_PATH)


# =============================================================================
# FINAL CHECKS AND COMPACT DISPLAY
# =============================================================================

expected_per_condition_rows_ci = (
    len(selected_condition_ids_ci)
    * (
        2 * len(requested_horizons_ci)
        + 4
        + 1
    )
)
if len(per_condition_ci) != expected_per_condition_rows_ci:
    raise RuntimeError(
        "Final per-condition CI table has an unexpected row count."
    )
if len(cindex_ci) != len(selected_condition_ids_ci):
    raise RuntimeError(
        "Final C-index CI table must contain one row per selected condition."
    )
if per_condition_ci["n_bootstrap_total"].ne(
    int(CI_BOOTSTRAP_N)
).any():
    raise RuntimeError(
        "At least one CI row does not contain all requested replicates."
    )
if not BOOTSTRAP_METADATA_PATH.is_file():
    raise RuntimeError("Bootstrap metadata file was not saved.")

print("=" * 88)
print("CELL 7B COMPLETE")
print("=" * 88)
print(
    f"Elapsed: {elapsed_seconds_ci / 60.0:.1f} minutes | "
    f"replicates: {CI_BOOTSTRAP_N:,} | "
    f"conditions: {len(selected_condition_ids_ci):,}"
)
print("Per-condition CI:", PER_CONDITION_CI_PATH)
print("Point-horizon CI:", POINT_HORIZON_CI_PATH)
print("C-index CI:", CINDEX_CI_PATH)
print("Integrated CI:", INTEGRATED_CI_PATH)
print("Macro CI:", MACRO_CI_PATH)
print("Bootstrap replicates:", BOOTSTRAP_REPLICATES_PATH)
print("Metadata:", BOOTSTRAP_METADATA_PATH)
print(
    macro_ci[
        [
            "metric",
            "horizon_months",
            "estimate",
            "ci_lower",
            "ci_upper",
            "n_valid_bootstrap",
            "ci_status",
        ]
    ]
)
