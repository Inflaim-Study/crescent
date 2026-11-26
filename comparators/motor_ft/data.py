# MOTOR-FT — Canonical split verification and predictor construction
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 4 ---
if reference_config["delayed_entry_contract"] != delayed_entry_contract():
    raise ValueError("Source likelihood or prediction landmark differs from this notebook.")
manifest = read_json(SOURCE / "model_input_manifest.json")
MODEL_INPUT_HASHES = dict(manifest["input_hashes"])
for filename, key in [(f"canonical_{s}_samples.pkl", f"canonical_{s}_samples_sha256")
                      for s in ("train", "val")] + [
                      ("cell3_vocab_scalers_deathsafe.json", "vocabulary_and_scalers_sha256")]:
    if sha256_file(SOURCE / filename) != MODEL_INPUT_HASHES[key]:
        raise ValueError(f"Canonical file differs from its source manifest: {filename}")
VOCAB_METADATA = read_json(SOURCE / "cell3_vocab_scalers_deathsafe.json")
TARGET_METADATA = read_json(SOURCE / "canonical_target_metadata.json")
MODEL_INPUT_HASHES["target_metadata_sha256"] = sha256_file(SOURCE / "canonical_target_metadata.json")
MODEL_INPUT_HASHES["source_config_sha256"] = sha256_file(SOURCE / "config.json")
SAMPLES = {s: pd.read_pickle(SOURCE / f"canonical_{s}_samples.pkl") for s in ("train", "val")}

CONDITION_META = VOCAB_METADATA["condition_vocab"]
EVENT_META = VOCAB_METADATA["event_vocab"]
TYPE_META = VOCAB_METADATA["event_type_vocab"]
STATIC_META = VOCAB_METADATA["static_categories"]

K = int(TARGET_METADATA["K"])
condition_names = [str(x) for x in TARGET_METADATA["condition_names"]]
if K != len(condition_names):
    raise RuntimeError("Canonical condition_names length differs from K.")
if int(CONDITION_META.get("K", K)) != K:
    raise RuntimeError("Canonical vocabulary and target metadata disagree on K.")

code_to_id = {str(k): int(v) for k, v in CONDITION_META.get("code_to_id", {}).items()}
id_to_code = {int(k): str(v) for k, v in CONDITION_META.get("id_to_code", {}).items()}
if not id_to_code:
    id_to_code = {i: name for i, name in enumerate(condition_names)}

EV_VOCAB_SIZE_MODEL = int(EVENT_META["EV_VOCAB_SIZE_MODEL"])
EV_PAD_ID = int(EVENT_META["EV_PAD_ID"])
EV_UNK_ID = int(EVENT_META["EV_UNK_ID"])
EV_CLS_ID = int(EVENT_META["EV_CLS_ID"])
EV_NEAR_RECRUITMENT_INDEX_ID = int(EVENT_META["EV_NEAR_RECRUITMENT_INDEX_ID"])
EV_TOKEN_TO_ID = {str(k): int(v) for k, v in EVENT_META["EV_TOKEN_TO_ID"].items()}
EV_ID_TO_TOKEN = {int(k): str(v) for k, v in EVENT_META["EV_ID_TO_TOKEN"].items()}

TYPE_VOCAB_SIZE_MODEL = int(TYPE_META["TYPE_VOCAB_SIZE_MODEL"])
TYPE_PAD_ID = int(TYPE_META["TYPE_PAD_ID"])
TYPE_UNK_ID = int(TYPE_META["TYPE_UNK_ID"])
TYPE_CLS_ID = int(TYPE_META["TYPE_CLS_ID"])
TYPE_TO_ID = {
    str(k).strip().lower(): int(v) for k, v in TYPE_META["TYPE_TO_ID"].items()
}
TYPE_ID_TO_TYPE = {int(k): str(v) for k, v in TYPE_META["TYPE_ID_TO_TYPE"].items()}
CRP_TYPE_ID_FOR_AUDIT = TYPE_META.get("CRP_TYPE_ID_FOR_AUDIT")
if CRP_TYPE_ID_FOR_AUDIT is not None:
    CRP_TYPE_ID_FOR_AUDIT = int(CRP_TYPE_ID_FOR_AUDIT)

N_GENDER_MODEL = int(STATIC_META["N_GENDER_MODEL"])
UNKNOWN_GENDER_IDX = int(STATIC_META["UNKNOWN_GENDER_IDX"])
N_ETHNICITY_KNOWN = int(STATIC_META["N_ETHNICITY_KNOWN"])
UNKNOWN_ETHNICITY_IDX = int(STATIC_META["UNKNOWN_ETHNICITY_IDX"])
N_ETHNICITY_MODEL = int(STATIC_META["N_ETHNICITY_MODEL"])
N_TOWNSEND_KNOWN = int(STATIC_META["N_TOWNSEND_KNOWN"])
UNKNOWN_TOWNSEND_Q = int(STATIC_META["UNKNOWN_TOWNSEND_Q"])
N_TOWNSEND_MODEL = int(STATIC_META["N_TOWNSEND_MODEL"])

STATIC_CONTINUOUS_SCALER = VOCAB_METADATA.get("static_continuous_scaler", {})
SAMPLE_CONTINUOUS_COLS_RAW = tuple(
    VOCAB_METADATA.get("sample_continuous_cols_raw", cfg.SAMPLE_CONTINUOUS_COLS_RAW)
)

train_samples = SAMPLES["train"].copy()
val_samples = SAMPLES["val"].copy()



REQUIRED_SAMPLE_COLUMNS = {
    "eid", "index_dt", "index_age_years", "index_age_months",
    "index_offset_days_from_recruitment", "history_event_ids",
    "history_event_type_ids", "history_event_times_months",
    "history_event_values",
    "history_event_value_masks", "continuous_static",
    "continuous_static_raw", "gender_idx", "ethnicity_idx", "townsend_q",
    "at_risk", "event_obs", "event_bin", "t_tilde", "C_months", "C_bin",
    "death_observed", "death_age", "death_observed_post_index",
    "death_time_months", "death_bin", "cr_at_risk", "cr_event_type",
    "cr_event_bin", "cr_event_time_months",
}
ARRAY_K_COLUMNS = (
    "at_risk", "event_obs", "event_bin", "t_tilde", "cr_at_risk",
    "cr_event_type", "cr_event_bin", "cr_event_time_months",
)
EVENT_ARRAY_COLUMNS = (
    "history_event_ids", "history_event_type_ids", "history_event_times_months",
    "history_event_values", "history_event_value_masks",
)


def _array_length(value):
    try:
        return len(value)
    except Exception:
        return -1


def validate_canonical_samples(frame, split):
    missing = sorted(REQUIRED_SAMPLE_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(
            f"{split}: canonical source is missing fields required by this "
            f"CRESCENT architecture: {missing}"
        )
    if frame.empty:
        raise ValueError(f"{split}: canonical sample frame is empty.")
    if frame["eid"].astype(str).duplicated().any():
        raise ValueError(f"{split}: expected exactly one canonical row per eid.")

    for column in ARRAY_K_COLUMNS:
        bad = frame[column].map(_array_length).ne(K)
        if bad.any():
            raise ValueError(
                f"{split}: {column} has {int(bad.sum())} rows not length K={K}."
            )

    for row_index, row in frame.iterrows():
        lengths = [_array_length(row[column]) for column in EVENT_ARRAY_COLUMNS]
        if min(lengths) < 1 or len(set(lengths)) != 1:
            raise ValueError(
                f"{split} row={row_index} eid={row['eid']}: misaligned event arrays {lengths}."
            )
        index_years = float(row["index_age_years"])
        index_months = float(row["index_age_months"])
        if not np.isfinite(index_years) or not np.isfinite(index_months):
            raise ValueError(f"{split} row={row_index}: non-finite index age.")
        if not np.isclose(index_months, 12.0 * index_years, rtol=0.0, atol=1e-4):
            raise ValueError(f"{split} row={row_index}: inconsistent index age units.")
        event_times = np.asarray(row["history_event_times_months"], dtype=float)
        if not np.all(np.isfinite(event_times)):
            raise ValueError(f"{split} row={row_index}: non-finite event time.")
        if np.any(event_times > index_months + 12.0 / 365.25 + 1e-6):
            raise ValueError(
                f"{split} row={row_index}: genuine canonical history crosses the index."
            )



for split, frame in SAMPLES.items():
    validate_canonical_samples(frame, split)
if len(condition_names) != len(set(condition_names)) or any(code_to_id.get(n) != i for i,n in enumerate(condition_names)):
    raise ValueError("Target metadata and the hashed vocabulary disagree on the endpoint axis.")
if manifest["K"] != K:
    raise ValueError("Target count differs from the manifest.")
train_eids = set(train_samples["eid"].astype(str))
val_eids = set(val_samples["eid"].astype(str))
if train_eids & val_eids:
    raise ValueError("Train and validation participants overlap.")

N_CONTINUOUS_STATIC = int(np.asarray(train_samples.iloc[0]["continuous_static"]).size)
if N_CONTINUOUS_STATIC < 1:
    raise ValueError("Canonical continuous_static vectors are empty.")
for split_name, frame in (
    ("train", train_samples),
    ("val", val_samples),
):
    bad = frame["continuous_static"].map(
        lambda value: np.asarray(value).size != N_CONTINUOUS_STATIC
    )
    if bad.any():
        raise ValueError(f"{split_name}: inconsistent continuous_static width.")

CR_TARGET_LABELS = TARGET_METADATA["death_adjusted_design"]["cr_target_labels"]
CR_TARGET_CENSOR = int(next(k for k, v in CR_TARGET_LABELS.items() if "censor" in v or "neither" in v))
CR_TARGET_CONDITION = int(next(k for k, v in CR_TARGET_LABELS.items() if "condition" in v))
CR_TARGET_DEATH = int(next(k for k, v in CR_TARGET_LABELS.items() if "death" in v))
if (
    CR_TARGET_CENSOR != int(cfg.CR_TARGET_CENSOR)
    or CR_TARGET_CONDITION != int(cfg.CR_TARGET_CONDITION)
    or CR_TARGET_DEATH != int(cfg.CR_TARGET_DEATH)
):
    raise RuntimeError("Canonical competing-risk labels differ from CRESCENT config.")


print({s:len(frame) for s,frame in SAMPLES.items()}, "conditions:", K)

# --- Notebook cell 6 ---
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

assert "cfg" in globals() and "haz" in globals() and "RUN_DIR" in globals(), "Run Cell 1 first."
assert "train_samples" in globals() and "val_samples" in globals(), "Run canonical loader Cell 2 first."
assert "K" in globals(), "Run canonical loader Cell 2 first."
assert "EV_PAD_ID" in globals() and "EV_CLS_ID" in globals(), "Run canonical loader Cell 2 first."
assert "TYPE_PAD_ID" in globals() and "TYPE_CLS_ID" in globals(), "Run canonical loader Cell 2 first."
assert "EV_NEAR_RECRUITMENT_INDEX_ID" in globals(), "Run canonical loader Cell 2 first."
assert "N_GENDER_MODEL" in globals(), "Run canonical loader Cell 2 first."
assert "N_ETHNICITY_MODEL" in globals() and "N_ETHNICITY_KNOWN" in globals() and "UNKNOWN_ETHNICITY_IDX" in globals(), "Run canonical loader Cell 2 first."
assert "N_TOWNSEND_MODEL" in globals() and "N_TOWNSEND_KNOWN" in globals() and "UNKNOWN_TOWNSEND_Q" in globals(), "Run canonical loader Cell 2 first."
assert "N_CONTINUOUS_STATIC" in globals(), "Run canonical loader Cell 2 first."

RUN_DIR = Path(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

CR_TARGET_CENSOR = int(getattr(cfg, "CR_TARGET_CENSOR", 0))
CR_TARGET_CONDITION = int(getattr(cfg, "CR_TARGET_CONDITION", 1))
CR_TARGET_DEATH = int(getattr(cfg, "CR_TARGET_DEATH", 2))

FULL_SEQUENCE_MAX_EVENTS = int(getattr(cfg, "FULL_SEQUENCE_MAX_EVENTS", 25))
if FULL_SEQUENCE_MAX_EVENTS < 1:
    raise ValueError("FULL_SEQUENCE_MAX_EVENTS must be positive.")
FULL_SEQUENCE_MAX_LEN = FULL_SEQUENCE_MAX_EVENTS + 1  # one appended CLS token
AGE_TOL_MONTHS = float(getattr(cfg, "AGE_TOL_YEARS", 1.0 / 365.25)) * 12.0
TIME_EPS_MONTHS = 1e-6


# =============================================================================
# Utilities
# =============================================================================

def _to_float32_array(x, expected_len=None, fill_value=0.0):
    arr = np.asarray(x, dtype=np.float32)
    if expected_len is not None and arr.shape[0] != int(expected_len):
        raise ValueError(f"Expected length {expected_len}, got {arr.shape[0]}.")
    arr = np.where(np.isfinite(arr), arr, fill_value).astype(np.float32)
    return arr


def _to_float32_array_allow_inf(x, expected_len=None):
    arr = np.asarray(x, dtype=np.float32)
    if expected_len is not None and arr.shape[0] != int(expected_len):
        raise ValueError(f"Expected length {expected_len}, got {arr.shape[0]}.")
    bad = np.isnan(arr) | (arr == -np.inf)
    arr[bad] = np.inf
    return arr.astype(np.float32)


def _to_int64_array(x, expected_len=None, fill_value=-1):
    arr = np.asarray(x, dtype=np.int64)
    if expected_len is not None and arr.shape[0] != int(expected_len):
        raise ValueError(f"Expected length {expected_len}, got {arr.shape[0]}.")
    return arr.astype(np.int64)


def _safe_float(x, default=np.nan):
    try:
        value = float(x)
    except Exception:
        return float(default)
    if np.isfinite(value):
        return float(value)
    if np.isposinf(value):
        return float(np.inf)
    return float(default)


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return int(default)


def _extract_genuine_prelandmark_history(row, split_name, row_index):
    """
    Return aligned candidate events after enforcing the input boundary.

    The canonical sample may contain the artificial HYBRID_INDEX landmark.
    It is removed here. CLS and padding do not occur in canonical sample arrays,
    but are excluded defensively. Unknown clinical tokens remain valid inputs.
    """
    event_ids = np.asarray(row["history_event_ids"], dtype=np.int64)
    type_ids = np.asarray(row["history_event_type_ids"], dtype=np.int64)
    event_times = np.asarray(row["history_event_times_months"], dtype=np.float64)
    event_values = np.asarray(row["history_event_values"], dtype=np.float32)
    value_masks = np.asarray(row["history_event_value_masks"], dtype=np.float32)

    lengths = {
        int(event_ids.size),
        int(type_ids.size),
        int(event_times.size),
        int(event_values.size),
        int(value_masks.size),
    }
    if len(lengths) != 1:
        raise ValueError(
            f"{split_name} row={row_index}: canonical history arrays are misaligned."
        )

    index_months = float(row["index_age_months"])
    finite = np.isfinite(event_times)
    prelandmark = event_times <= index_months + AGE_TOL_MONTHS + TIME_EPS_MONTHS
    genuine = (
        (event_ids != int(EV_NEAR_RECRUITMENT_INDEX_ID))
        & (event_ids != int(EV_CLS_ID))
        & (event_ids != int(EV_PAD_ID))
    )
    keep = finite & prelandmark & genuine

    # The canonical builder is already required to exclude future inputs. Raise
    # rather than silently correcting a genuine event that crosses the boundary.
    future_genuine = finite & (~prelandmark) & genuine
    if bool(future_genuine.any()):
        first = int(np.flatnonzero(future_genuine)[0])
        raise ValueError(
            f"{split_name} row={row_index}: a genuine predictor event occurs "
            f"after the locked landmark (event_time={event_times[first]}, "
            f"index_time={index_months})."
        )

    order = np.lexsort((type_ids[keep], event_ids[keep], event_times[keep]))
    kept_event_times = event_times[keep][order]
    kept_event_times = np.minimum(kept_event_times, index_months)

    return {
        "event_ids": event_ids[keep][order].astype(np.int64),
        "event_type_ids": type_ids[keep][order].astype(np.int64),
        "event_times_months": kept_event_times.astype(np.float32),
        "event_values": np.nan_to_num(
            event_values[keep][order], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32),
        "event_value_masks": (value_masks[keep][order] >= 0.5).astype(np.float32),
    }


# =============================================================================
# Dataset
# =============================================================================

# The canonical data provides at most one timed CRP event per participant.
# Retain it in the 25-event input if it falls outside the recent window.
if bool(cfg.EXCLUDE_CRP_FROM_INPUTS):
    raise ValueError("This is the CRP-enabled experiment; CRP exclusion must be disabled.")
if not bool(cfg.INCLUDE_CRP_AS_EVENT_VALUE):
    raise ValueError("INCLUDE_CRP_AS_EVENT_VALUE must be True for the CRP-enabled experiment.")
if not bool(cfg.USE_EVENT_VALUE_CHANNEL):
    raise ValueError("USE_EVENT_VALUE_CHANNEL must be True for numeric CRP input.")
if "crp" not in TYPE_TO_ID:
    raise ValueError("No CRP type in the canonical A10 vocabulary; check the source artifacts.")
CRP_TYPE_ID = int(TYPE_TO_ID["crp"])
CRP_TOKEN_IDS = np.asarray([
    int(token_id) for token, token_id in EV_TOKEN_TO_ID.items()
    if str(token).strip().upper() == "CRP_MEASURED"
    or str(token).strip().upper().startswith("CRP_D")
], dtype=np.int64)
if any("crp" in str(name).lower() for name in cfg.SAMPLE_CONTINUOUS_COLS_RAW):
    raise ValueError("CRP must be represented only as a timed event, not a static covariate.")


def _crp_event_mask(event_ids, event_type_ids):
    return ((np.asarray(event_type_ids) == CRP_TYPE_ID)
            | np.isin(np.asarray(event_ids), CRP_TOKEN_IDS))


def _select_recent_history_with_crp(history, max_events):
    """Select recent events, retaining CRP when present."""
    n_events = int(history["event_ids"].size)
    max_events = int(max_events)
    if max_events < 1:
        raise ValueError("max_events must be positive.")
    if n_events <= max_events:
        return {name: values.copy() for name, values in history.items()}

    selected = np.arange(n_events - max_events, n_events, dtype=np.int64)
    crp_positions = np.flatnonzero(
        _crp_event_mask(history["event_ids"], history["event_type_ids"])
    )
    if crp_positions.size > 0:
        crp_position = int(crp_positions[-1])
        if not bool(np.isin(crp_position, selected)):
            selected[0] = crp_position
            selected.sort()

    return {name: values[selected].copy() for name, values in history.items()}


class FixedLandmarkSequenceDataset(Dataset):
    """
    One immutable landmark sample per participant. The Transformer receives
    up to 25 selected events and a CLS token.
    """

    def __init__(self, samples, split_name="train", full_sequence_max_events=None):
        self.samples = samples.reset_index(drop=True).copy()
        self.split_name = str(split_name)
        self.full_sequence_max_events = int(
            FULL_SEQUENCE_MAX_EVENTS if full_sequence_max_events is None else full_sequence_max_events
        )

        required_cols = [
            "eid",
            "index_age_months",
            "index_age_years",
            "history_event_ids",
            "history_event_type_ids",
            "history_event_times_months",
            "history_event_values",
            "history_event_value_masks",
            "gender_idx",
            "ethnicity_idx",
            "townsend_q",
            "continuous_static",
            "continuous_static_raw",
            "at_risk",
            "event_obs",
            "event_bin",
            "C_bin",
            "C_months",
            "t_tilde",
            "death_observed",
            "death_age",
            "death_observed_post_index",
            "death_time_months",
            "death_bin",
            "cr_at_risk",
            "cr_event_type",
            "cr_event_bin",
            "cr_event_time_months",
        ]
        missing = [name for name in required_cols if name not in self.samples.columns]
        if missing:
            raise ValueError(f"{self.split_name}: missing required sample columns: {missing}")
        if len(self.samples) == 0:
            raise ValueError(f"{self.split_name}: sample dataframe is empty.")
        if self.samples["eid"].astype(str).duplicated().any():
            raise ValueError(
                f"{self.split_name}: expected exactly one locked-landmark row per eid."
            )

    def __len__(self):
        return int(len(self.samples))

    def __getitem__(self, idx):
        row = self.samples.iloc[int(idx)]
        history = _extract_genuine_prelandmark_history(
            row=row,
            split_name=self.split_name,
            row_index=int(idx),
        )

        sequence = _select_recent_history_with_crp(
            history,
            max_events=self.full_sequence_max_events,
        )

        item = {
            # CLS is appended in the collator.
            "sequence_event_ids": torch.as_tensor(sequence["event_ids"], dtype=torch.long),
            "sequence_event_type_ids": torch.as_tensor(sequence["event_type_ids"], dtype=torch.long),
            "sequence_event_times_months": torch.as_tensor(sequence["event_times_months"], dtype=torch.float32),
            "sequence_event_values": torch.as_tensor(sequence["event_values"], dtype=torch.float32),
            "sequence_event_value_masks": torch.as_tensor(sequence["event_value_masks"], dtype=torch.float32),
            "index_age_months": torch.tensor(
                _safe_float(row["index_age_months"], default=0.0),
                dtype=torch.float32,
            ),
            "index_age_years": torch.tensor(
                _safe_float(row["index_age_years"], default=0.0),
                dtype=torch.float32,
            ),
            "gender_idx": torch.tensor(_safe_int(row["gender_idx"], default=2), dtype=torch.long),
            "ethnicity_idx": torch.tensor(_safe_int(row["ethnicity_idx"], default=0), dtype=torch.long),
            "townsend_q": torch.tensor(_safe_int(row["townsend_q"], default=UNKNOWN_TOWNSEND_Q), dtype=torch.long),
            "continuous_static": torch.tensor(
                _to_float32_array(
                    row["continuous_static"],
                    expected_len=N_CONTINUOUS_STATIC,
                    fill_value=0.0,
                ),
                dtype=torch.float32,
            ),
            "continuous_static_raw": torch.tensor(
                np.asarray(row["continuous_static_raw"], dtype=np.float32),
                dtype=torch.float32,
            ),
            # Existing disease targets.
            "at_risk": torch.tensor(
                _to_float32_array(row["at_risk"], expected_len=K, fill_value=0.0),
                dtype=torch.float32,
            ),
            "event_obs": torch.tensor(
                _to_float32_array(row["event_obs"], expected_len=K, fill_value=0.0),
                dtype=torch.float32,
            ),
            "event_bin": torch.tensor(
                _to_int64_array(row["event_bin"], expected_len=K, fill_value=-1),
                dtype=torch.long,
            ),
            "C_bin": torch.tensor(_safe_int(row["C_bin"], default=0), dtype=torch.long),
            "C_months": torch.tensor(_safe_float(row["C_months"], default=0.0), dtype=torch.float32),
            "t_tilde": torch.tensor(
                _to_float32_array_allow_inf(row["t_tilde"], expected_len=K),
                dtype=torch.float32,
            ),
            # Existing shared death targets.
            "death_observed": torch.tensor(_safe_int(row["death_observed"], default=0), dtype=torch.float32),
            "death_age": torch.tensor(_safe_float(row["death_age"], default=np.nan), dtype=torch.float32),
            "death_observed_post_index": torch.tensor(
                _safe_int(row["death_observed_post_index"], default=0),
                dtype=torch.float32,
            ),
            "death_time_months": torch.tensor(
                _safe_float(row["death_time_months"], default=np.inf),
                dtype=torch.float32,
            ),
            "death_bin": torch.tensor(_safe_int(row["death_bin"], default=-1), dtype=torch.long),
            # Existing per-condition competing-risk targets.
            "cr_at_risk": torch.tensor(
                _to_float32_array(row["cr_at_risk"], expected_len=K, fill_value=0.0),
                dtype=torch.float32,
            ),
            "cr_event_type": torch.tensor(
                _to_int64_array(
                    row["cr_event_type"],
                    expected_len=K,
                    fill_value=CR_TARGET_CENSOR,
                ),
                dtype=torch.long,
            ),
            "cr_event_bin": torch.tensor(
                _to_int64_array(row["cr_event_bin"], expected_len=K, fill_value=-1),
                dtype=torch.long,
            ),
            "cr_event_time_months": torch.tensor(
                _to_float32_array_allow_inf(
                    row["cr_event_time_months"], expected_len=K
                ),
                dtype=torch.float32,
            ),
            "eid": str(row["eid"]),
        }
        return item


# =============================================================================
# Collate
# =============================================================================

def fixed_landmark_sequence_collate(batch, training=False):
    B = len(batch)
    if B == 0:
        raise ValueError("Empty batch received by collate.")

    eids = [item["eid"] for item in batch]
    index_age_months = torch.stack([item["index_age_months"] for item in batch]).float()
    index_age_years = torch.stack([item["index_age_years"] for item in batch]).float()

    # Selected event sequence with one CLS at the landmark.
    sequence_ids = []
    sequence_types = []
    sequence_times = []
    sequence_values = []
    sequence_masks = []
    sequence_real_counts = []

    for i, item in enumerate(batch):
        real_count = int(item["sequence_event_ids"].numel())
        sequence_real_counts.append(real_count)
        sequence_ids.append(
            torch.cat(
                [item["sequence_event_ids"], torch.tensor([int(EV_CLS_ID)], dtype=torch.long)]
            )
        )
        sequence_types.append(
            torch.cat(
                [item["sequence_event_type_ids"], torch.tensor([int(TYPE_CLS_ID)], dtype=torch.long)]
            )
        )
        sequence_times.append(
            torch.cat(
                [
                    item["sequence_event_times_months"],
                    index_age_months[i].view(1).to(torch.float32),
                ]
            )
        )
        sequence_values.append(
            torch.cat(
                [item["sequence_event_values"], torch.tensor([0.0], dtype=torch.float32)]
            )
        )
        sequence_masks.append(
            torch.cat(
                [item["sequence_event_value_masks"], torch.tensor([0.0], dtype=torch.float32)]
            )
        )

    lengths = torch.tensor([int(x.numel()) for x in sequence_ids], dtype=torch.long)
    sequence_real_counts = torch.tensor(sequence_real_counts, dtype=torch.long)
    Lmax = int(lengths.max().item())
    if Lmax > int(FULL_SEQUENCE_MAX_LEN):
        raise RuntimeError(
            f"Full sequence exceeded FULL_SEQUENCE_MAX_LEN={FULL_SEQUENCE_MAX_LEN}: {Lmax}."
        )

    event_ids = torch.full((B, Lmax), int(EV_PAD_ID), dtype=torch.long)
    event_type_ids = torch.full((B, Lmax), int(TYPE_PAD_ID), dtype=torch.long)
    event_times_months = torch.zeros((B, Lmax), dtype=torch.float32)
    event_values = torch.zeros((B, Lmax), dtype=torch.float32)
    event_value_masks = torch.zeros((B, Lmax), dtype=torch.float32)
    attn_mask = torch.zeros((B, Lmax), dtype=torch.float32)

    for i in range(B):
        L = int(lengths[i].item())
        event_ids[i, :L] = sequence_ids[i]
        event_type_ids[i, :L] = sequence_types[i]
        event_times_months[i, :L] = sequence_times[i]
        event_values[i, :L] = sequence_values[i]
        event_value_masks[i, :L] = sequence_masks[i]
        attn_mask[i, :L] = 1.0

    # Static covariates.
    gender_idx = torch.stack([item["gender_idx"] for item in batch]).long()
    ethnicity_idx = torch.stack([item["ethnicity_idx"] for item in batch]).long()
    townsend_q = torch.stack([item["townsend_q"] for item in batch]).long()
    continuous_static = torch.stack([item["continuous_static"] for item in batch]).float()
    continuous_static_raw = torch.stack([item["continuous_static_raw"] for item in batch]).float()

    gender_idx = gender_idx.clamp(0, int(N_GENDER_MODEL) - 1)
    bad_eth = (ethnicity_idx < 0) | (ethnicity_idx >= int(N_ETHNICITY_KNOWN))
    ethnicity_idx[bad_eth] = int(UNKNOWN_ETHNICITY_IDX)
    bad_town = (townsend_q < 0) | (townsend_q >= int(N_TOWNSEND_KNOWN))
    townsend_q[bad_town] = int(UNKNOWN_TOWNSEND_Q)

    # Optional sequence token masking.
    if bool(training) and float(getattr(cfg, "EVENT_MASK_PROB", 0.0)) > 0.0:
        probability = float(getattr(cfg, "EVENT_MASK_PROB", 0.0))
        valid = attn_mask > 0.5
        cls_positions = torch.arange(Lmax).view(1, -1) == (lengths.view(-1, 1) - 1)
        can_mask = valid & (~cls_positions)
        random_draw = torch.rand_like(attn_mask)
        event_ids[(random_draw < probability) & can_mask] = int(EV_UNK_ID)

    if bool(training) and float(getattr(cfg, "COVARIATE_DROP_PROB", 0.0)) > 0.0:
        probability = float(getattr(cfg, "COVARIATE_DROP_PROB", 0.0))
        drop = torch.rand((B, 1), dtype=torch.float32) < probability
        continuous_static = continuous_static.masked_fill(drop, 0.0)

    out = {
        "eid": eids,
        # Padded input sequence for the Transformer.
        "event_ids": event_ids,
        "event_type_ids": event_type_ids,
        "event_times_months": event_times_months,
        "event_values": event_values,
        "event_value_masks": event_value_masks,
        "attn_mask": attn_mask,
        "lengths": lengths,
        "sequence_real_counts": sequence_real_counts,
        "index_age_months": index_age_months,
        "index_age_years": index_age_years,
        "gender_idx": gender_idx,
        "ethnicity_idx": ethnicity_idx,
        "townsend_q": townsend_q,
        "continuous_static": continuous_static,
        "continuous_static_raw": continuous_static_raw,
        # Existing targets.
        "at_risk": torch.stack([item["at_risk"] for item in batch]).float(),
        "event_obs": torch.stack([item["event_obs"] for item in batch]).float(),
        "event_bin": torch.stack([item["event_bin"] for item in batch]).long(),
        "C_bin": torch.stack([item["C_bin"] for item in batch]).long(),
        "C_months": torch.stack([item["C_months"] for item in batch]).float(),
        "t_tilde": torch.stack([item["t_tilde"] for item in batch]).float(),
        "death_observed": torch.stack([item["death_observed"] for item in batch]).float(),
        "death_age": torch.stack([item["death_age"] for item in batch]).float(),
        "death_observed_post_index": torch.stack(
            [item["death_observed_post_index"] for item in batch]
        ).float(),
        "death_time_months": torch.stack([item["death_time_months"] for item in batch]).float(),
        "death_bin": torch.stack([item["death_bin"] for item in batch]).long(),
        "cr_at_risk": torch.stack([item["cr_at_risk"] for item in batch]).float(),
        "cr_event_type": torch.stack([item["cr_event_type"] for item in batch]).long(),
        "cr_event_bin": torch.stack([item["cr_event_bin"] for item in batch]).long(),
        "cr_event_time_months": torch.stack(
            [item["cr_event_time_months"] for item in batch]
        ).float(),
    }
    return out


# =============================================================================
# History coverage audits
# =============================================================================

def summarise_history_coverage(samples, split_name):
    counts = []
    sequence_counts = []
    crp_available = 0
    crp_sequence = 0
    crp_numeric_values = 0
    crp_missing_values = 0
    participants_with_crp = 0
    post_landmark_count = 0
    landmark_count = 0
    for index, row in samples.reset_index(drop=True).iterrows():
        history = _extract_genuine_prelandmark_history(row, split_name, index)
        source_crp = _crp_event_mask(history["event_ids"], history["event_type_ids"])
        if int(source_crp.sum()) > 1:
            raise RuntimeError(
                f"{split_name}: participant {row['eid']} has more than one CRP event."
            )
        sequence = _select_recent_history_with_crp(
            history,
            max_events=FULL_SEQUENCE_MAX_EVENTS,
        )
        sequence_crp = _crp_event_mask(
            sequence["event_ids"], sequence["event_type_ids"]
        )
        if bool(source_crp.any()) and not bool(sequence_crp.any()):
            raise RuntimeError(
                f"{split_name}: CRP was not retained in the input sequence "
                f"for participant {row['eid']}."
            )
        crp_available += int(source_crp.sum())
        crp_sequence += int(sequence_crp.sum())
        participants_with_crp += int(source_crp.any())
        if bool(source_crp.any()):
            crp_masks = history["event_value_masks"][source_crp] >= 0.5
            crp_values = history["event_values"][source_crp]
            crp_numeric_values += int(crp_masks.sum())
            crp_missing_values += int((~crp_masks).sum())
            if bool((crp_masks & (crp_values < 0.0)).any()):
                raise RuntimeError(
                    f"{split_name}: participant {row['eid']} has a negative CRP value."
                )
        count = int(history["event_ids"].size)
        counts.append(count)
        sequence_counts.append(int(sequence["event_ids"].size))
        raw_ids = np.asarray(row["history_event_ids"], dtype=np.int64)
        landmark_count += int((raw_ids == int(EV_NEAR_RECRUITMENT_INDEX_ID)).sum())
        raw_times = np.asarray(row["history_event_times_months"], dtype=np.float64)
        post_landmark_count += int(
            (
                np.isfinite(raw_times)
                & (raw_times > float(row["index_age_months"]) + AGE_TOL_MONTHS + TIME_EPS_MONTHS)
                & (raw_ids != int(EV_NEAR_RECRUITMENT_INDEX_ID))
            ).sum()
        )
    values = np.asarray(counts, dtype=np.int64)
    sequence_values = np.asarray(sequence_counts, dtype=np.int64)
    return {
        "split": str(split_name),
        "participants": int(len(samples)),
        "available_events_total": int(values.sum()),
        "available_events_mean": float(values.mean()),
        "available_events_median": float(np.median(values)),
        "available_events_p90": float(np.quantile(values, 0.90)),
        "available_events_p99": float(np.quantile(values, 0.99)),
        "available_events_max": int(values.max()),
        "participants_with_zero_genuine_events": int((values == 0).sum()),
        "input_events_total": int(sequence_values.sum()),
        "available_events_outside_input": int(
            (values - sequence_values).sum()
        ),
        "participants_exceeding_input_limit": int((values > FULL_SEQUENCE_MAX_EVENTS).sum()),
        "artificial_landmark_tokens_in_canonical_rows": int(landmark_count),
        "artificial_landmark_tokens_used_as_clinical_inputs": 0,
        "genuine_post_landmark_inputs": int(post_landmark_count),
        "crp_events_available": int(crp_available),
        "crp_events_in_input": int(crp_sequence),
        "crp_events_with_numeric_value": int(crp_numeric_values),
        "crp_events_with_missing_value": int(crp_missing_values),
        "participants_with_crp_input": int(participants_with_crp),
        "participants_dropped_for_crp": 0,
    }



train_ds = FixedLandmarkSequenceDataset(train_samples, "train")
val_ds = FixedLandmarkSequenceDataset(val_samples, "val")

from functools import partial
train_generator = torch.Generator().manual_seed(int(cfg.TRAINING_SEED))

def make_loader(dataset, training=False):
    return DataLoader(dataset, batch_size=int(cfg.BATCH_SIZE), shuffle=training,
                      generator=train_generator if training else None, num_workers=int(cfg.NUM_WORKERS),
                      pin_memory=DEVICE.type == "cuda", collate_fn=partial(fixed_landmark_sequence_collate, training=training))

train_loader = make_loader(train_ds, True)
val_loader = make_loader(val_ds)
print(f"Training on {len(train_ds):,} participants; validating on {len(val_ds):,}.")
