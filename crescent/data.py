"""Stage 2: load and audit the prepared splits, write provenance, build datasets and dataloaders.

Extracted from the CRESCENT_Final notebook (cells 4, 6, 8). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 4
# =============================================================================

# =============================================================================
# CELL 2 — LOAD AND AUDIT THE EXACT CANONICAL TRAIN/VALIDATION/TEST SPLITS
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# CELL 2 — BUILD THE MODEL SAMPLES FROM THE NEWLY PREPARED TRAIN / VAL / TEST FILES
# =============================================================================
# Each prepared row is one participant at the one-month-cap index. Predictors are the
# stored event history (all on or before the index). Targets are built here from the
# stored diagnosis first-occurrence ages, the administrative end of follow-up
# (end_age) and the post-index death fields; nothing after the index is a predictor.

import json
import numpy as np
import pandas as pd
from collections import Counter

PREPARED_JSON_COLUMNS = [
    "event_tokens", "event_ages", "event_types", "event_values", "event_value_masks",
    "unique_diagnosis", "unique_diagnosis_ages",
]
PREPARED_REQUIRED_COLUMNS = [
    "eid", "split", "index_dt", "index_age_years", "index_offset_days_from_recruitment",
    "gender_idx", "ethnicity_idx", "townsend_q", "townsend_z", "townsend_missing",
    "end_age", "death_observed", "death_age", "model_followup_months",
] + PREPARED_JSON_COLUMNS
MAX_INDEX_OFFSET_DAYS = 31.0   # one-calendar-month cap around recruitment
LANDMARK_TOKEN_NAME = "HYBRID_INDEX"


def read_prepared_split(name):
    frame = pd.read_csv(PREPARED_PATHS[name], dtype={"eid": "string"}, low_memory=False)
    missing = [c for c in PREPARED_REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(
            f"{PREPARED_PATHS[name].name} lacks {missing}. Re-run the updated data-preparation "
            "notebook (it writes townsend_q and the readable categorical columns)."
        )
    if not frame["split"].astype(str).eq(name).all():
        raise ValueError(f"{PREPARED_PATHS[name].name} contains rows from another split.")
    for column in PREPARED_JSON_COLUMNS:
        frame[column] = frame[column].map(lambda s: json.loads(s) if isinstance(s, str) else [])
    frame["eid"] = frame["eid"].astype(str)
    return frame.reset_index(drop=True)


prepared = {name: read_prepared_split(name) for name in ("train", "val", "test")}
_all_eids = pd.concat([prepared[n]["eid"] for n in prepared])
if _all_eids.duplicated().any():
    raise RuntimeError("Participant appears in more than one prepared split (or twice in one).")
for _name, _frame in prepared.items():
    _offset = pd.to_numeric(_frame["index_offset_days_from_recruitment"], errors="coerce")
    if (~np.isfinite(_offset)).any() or (_offset.abs() > MAX_INDEX_OFFSET_DAYS).any():
        raise RuntimeError(f"{_name}: index offset from recruitment exceeds the one-calendar-month cap.")

# ---- static categorical vocabulary written by the preparation notebook ------------------------
with PREPARED_PATHS["static_vocabulary"].open(encoding="utf-8") as handle:
    STATIC_META_PREPARED = json.load(handle)
for _key in ("N_GENDER_MODEL", "UNKNOWN_GENDER_IDX", "N_ETHNICITY_KNOWN", "UNKNOWN_ETHNICITY_IDX",
             "N_ETHNICITY_MODEL", "N_TOWNSEND_KNOWN", "UNKNOWN_TOWNSEND_Q", "N_TOWNSEND_MODEL"):
    if _key not in STATIC_META_PREPARED:
        raise KeyError(f"static_covariate_vocabulary.json is missing {_key}")
for _name, _frame in prepared.items():
    for _col, _n in (("gender_idx", "N_GENDER_MODEL"), ("ethnicity_idx", "N_ETHNICITY_MODEL"),
                     ("townsend_q", "N_TOWNSEND_MODEL")):
        _v = pd.to_numeric(_frame[_col], errors="coerce")
        if _v.isna().any() or (_v < 0).any() or (_v >= int(STATIC_META_PREPARED[_n])).any():
            raise RuntimeError(f"{_name}.{_col} has ids outside the declared vocabulary {_n}.")

# ---- target conditions -------------------------------------------------------------------------
# Default: every adult diagnosis code carried by at least MIN_TRAIN_PARTICIPANTS_PER_CONDITION
# training participants (counted over the full diagnosis record), in alphabetical order.
# To keep the exact condition list (and order) of an earlier run, point
# CRESCENT_CONDITION_LIST at its canonical_target_metadata.json, or at a csv whose first column
# (or a column called condition_name / code) lists the codes.
CONDITION_LIST_PATH = os.getenv("CRESCENT_CONDITION_LIST", "").strip()
MIN_TRAIN_PARTICIPANTS_PER_CONDITION = int(os.getenv("CRESCENT_MIN_TRAIN_PARTICIPANTS_PER_CONDITION", "100"))

train_dx_participants = Counter(
    code for codes in prepared["train"]["unique_diagnosis"] for code in set(map(str, codes))
)
if CONDITION_LIST_PATH:
    _p = Path(CONDITION_LIST_PATH).expanduser()
    if _p.suffix.lower() == ".json":
        _obj = json.loads(_p.read_text(encoding="utf-8"))
        condition_names = [str(x) for x in (_obj["condition_names"] if isinstance(_obj, dict) else _obj)]
    else:
        _tbl = pd.read_csv(_p)
        _col = next((c for c in ("condition_name", "code", "condition") if c in _tbl.columns), _tbl.columns[0])
        condition_names = [str(x) for x in _tbl[_col].tolist()]
    if len(set(condition_names)) != len(condition_names):
        raise ValueError("The condition list contains duplicates.")
    _seen_any = set(code for n in prepared for codes in prepared[n]["unique_diagnosis"] for code in map(str, codes))
    _unmatched = [c for c in condition_names if c not in _seen_any]
    if len(_unmatched) > 0.01 * len(condition_names):
        raise RuntimeError(
            f"{len(_unmatched)} of {len(condition_names)} listed conditions never occur in the prepared "
            f"diagnosis codes (for example {_unmatched[:5]}). Check the code format."
        )
    CONDITION_SELECTION = {"source": str(_p), "unmatched_codes": _unmatched}
else:
    condition_names = sorted(
        code for code, n in train_dx_participants.items() if n >= MIN_TRAIN_PARTICIPANTS_PER_CONDITION
    )
    CONDITION_SELECTION = {
        "source": "derived from the training split",
        "min_train_participants_per_condition": MIN_TRAIN_PARTICIPANTS_PER_CONDITION,
        "n_codes_in_training_diagnoses": int(len(train_dx_participants)),
    }
if not condition_names:
    raise RuntimeError("No target conditions selected; lower CRESCENT_MIN_TRAIN_PARTICIPANTS_PER_CONDITION.")
K_BUILD = len(condition_names)
COND_INDEX = {code: i for i, code in enumerate(condition_names)}
print(f"Target conditions: K = {K_BUILD:,}  ({CONDITION_SELECTION['source']})")

# ---- event and event-type vocabularies (training tokens only; anything unseen -> UNK) ----------
_special = ["<PAD>", "<UNK>", "<CLS>", LANDMARK_TOKEN_NAME]
_train_tokens = sorted({str(t) for toks in prepared["train"]["event_tokens"] for t in toks} - set(_special))
EV_TOKEN_TO_ID_BUILD = {tok: i for i, tok in enumerate(_special + _train_tokens)}
_type_special = ["<PAD>", "<UNK>", "<CLS>"]
_train_types = sorted({str(t).lower() for ts in prepared["train"]["event_types"] for t in ts} - set(_type_special))
TYPE_TO_ID_BUILD = {t: i for i, t in enumerate(_type_special + _train_types)}
if "crp" not in TYPE_TO_ID_BUILD:
    raise RuntimeError("No CRP events in the prepared training data.")

# ---- continuous static covariates and their training-only scaler --------------------------------
_cont_cols = tuple(cfg.SAMPLE_CONTINUOUS_COLS_RAW)
_age_train = prepared["train"]["index_age_years"].astype(float)
AGE_MEAN = float(_age_train.mean()); AGE_STD = float(_age_train.std(ddof=0))
if not np.isfinite(AGE_STD) or AGE_STD <= 1e-8:
    raise RuntimeError("Degenerate training index-age scale.")

def months_to_bin(months):
    """Same bin rule as the model: bin b covers (6b, 6b+6] months; clipped to the axis."""
    m = np.asarray(months, dtype=np.float64)
    b = np.ceil(np.maximum(m, 0.0) / float(haz.BIN_WIDTH_MONTHS)) - 1.0
    return np.clip(b, 0, int(haz.N_BINS) - 1)


def build_samples(frame, split):
    unk_ev = EV_TOKEN_TO_ID_BUILD["<UNK>"]; unk_ty = TYPE_TO_ID_BUILD["<UNK>"]
    tol_years = float(cfg.AGE_TOL_YEARS)
    records = []
    for r in frame.itertuples(index=False):
        ia = float(r.index_age_years); im = 12.0 * ia
        end_age = pd.to_numeric(r.end_age, errors="coerce")
        C = 12.0 * (float(end_age) - ia) if np.isfinite(end_age) else np.nan
        if not np.isfinite(C) or C <= 0:
            raise ValueError(f"{split} eid={r.eid}: non-positive administrative follow-up (end_age/index age).")
        d_age = pd.to_numeric(r.death_age, errors="coerce")
        death_obs = bool(int(r.death_observed) == 1 and np.isfinite(d_age))
        td = 12.0 * (float(d_age) - ia) if death_obs else np.inf
        if death_obs and td <= 0:
            raise ValueError(f"{split} eid={r.eid}: death on or before the index.")

        at_risk = np.ones(K_BUILD, dtype=np.uint8)
        t64 = np.full(K_BUILD, np.inf, dtype=np.float64)
        for code, age in zip(r.unique_diagnosis, r.unique_diagnosis_ages):
            k = COND_INDEX.get(str(code))
            if k is None:
                continue
            if float(age) <= ia + tol_years:          # already present at the index -> not at risk
                at_risk[k] = 0
            else:
                tm = 12.0 * (float(age) - ia)
                if tm <= C + 1e-6:                    # beyond administrative follow-up -> censored
                    t64[k] = tm
        has_c = np.isfinite(t64) & (at_risk == 1)
        cond_wins = has_c & ((t64 <= td + 1e-8) if death_obs else True)
        cr_type = np.zeros(K_BUILD, dtype=np.int8)
        cr_time = np.full(K_BUILD, np.inf, dtype=np.float64)
        cr_type[cond_wins] = int(cfg.CR_TARGET_CONDITION)
        cr_time[cond_wins] = t64[cond_wins]
        if death_obs:
            dmask = (at_risk == 1) & (~cond_wins)
            cr_type[dmask] = int(cfg.CR_TARGET_DEATH)
            cr_time[dmask] = td
        cr_bin = np.where(cr_type > 0, months_to_bin(cr_time), -1).astype(np.int16)
        event_bin = np.where(has_c, months_to_bin(t64), -1).astype(np.int16)

        toks = [str(t) for t in r.event_tokens]
        ev_ids = np.asarray([EV_TOKEN_TO_ID_BUILD.get(t, unk_ev) for t in toks], dtype=np.int64)
        ty_ids = np.asarray([TYPE_TO_ID_BUILD.get(str(t).lower(), unk_ty) for t in r.event_types], dtype=np.int64)
        townsend_observed = 1.0 - float(int(r.townsend_missing))
        raw_static = np.asarray([ia, float(r.townsend_z), townsend_observed], dtype=np.float32)
        scaled_static = np.asarray([(ia - AGE_MEAN) / AGE_STD, float(r.townsend_z), townsend_observed], dtype=np.float32)

        records.append({
            "eid": r.eid, "index_dt": str(r.index_dt),
            "index_age_years": ia, "index_age_months": im,
            "index_offset_days_from_recruitment": float(r.index_offset_days_from_recruitment),
            "history_event_ids": ev_ids, "history_event_type_ids": ty_ids,
            "history_event_times_months": 12.0 * np.asarray(r.event_ages, dtype=np.float64),
            "history_event_values": np.nan_to_num(np.asarray(r.event_values, dtype=np.float32)),
            "history_event_value_masks": np.asarray(r.event_value_masks, dtype=np.float32),
            "continuous_static": scaled_static, "continuous_static_raw": raw_static,
            "gender_idx": int(r.gender_idx), "ethnicity_idx": int(r.ethnicity_idx),
            "townsend_q": int(r.townsend_q),
            "at_risk": at_risk, "event_obs": has_c.astype(np.uint8), "event_bin": event_bin,
            "t_tilde": t64.astype(np.float32),
            "C_months": float(C), "C_bin": int(months_to_bin(C)),
            "death_observed": int(death_obs), "death_age": float(d_age) if death_obs else np.nan,
            "death_observed_post_index": int(death_obs),
            "death_time_months": float(td), "death_bin": int(months_to_bin(td)) if death_obs else -1,
            "cr_at_risk": at_risk, "cr_event_type": cr_type, "cr_event_bin": cr_bin,
            "cr_event_time_months": cr_time.astype(np.float32),
        })
    return pd.DataFrame.from_records(records)


SAMPLES = {name: build_samples(prepared[name], name) for name in ("train", "val", "test")}
for _name, _f in SAMPLES.items():
    _t = np.stack(_f["cr_event_type"].to_numpy())
    print(f"{_name}: {len(_f):,} participants | at-risk pairs {int(np.stack(_f['at_risk'].to_numpy()).sum()):,} | "
          f"condition events {int((_t == int(cfg.CR_TARGET_CONDITION)).sum()):,} | "
          f"death-competing pairs {int((_t == int(cfg.CR_TARGET_DEATH)).sum()):,}")

# ---- metadata objects used by the rest of the notebook ----------------------------------------
VOCAB_METADATA = {
    "schema_version": "crescent.prepared_vocabulary.v1",
    "condition_vocab": {
        "K": K_BUILD,
        "code_to_id": {c: i for c, i in COND_INDEX.items()},
        "id_to_code": {str(i): c for c, i in COND_INDEX.items()},
        "selection": CONDITION_SELECTION,
    },
    "event_vocab": {
        "EV_VOCAB_SIZE_MODEL": len(EV_TOKEN_TO_ID_BUILD),
        "EV_PAD_ID": EV_TOKEN_TO_ID_BUILD["<PAD>"], "EV_UNK_ID": EV_TOKEN_TO_ID_BUILD["<UNK>"],
        "EV_CLS_ID": EV_TOKEN_TO_ID_BUILD["<CLS>"],
        "EV_NEAR_RECRUITMENT_INDEX_ID": EV_TOKEN_TO_ID_BUILD[LANDMARK_TOKEN_NAME],
        "EV_TOKEN_TO_ID": EV_TOKEN_TO_ID_BUILD,
        "EV_ID_TO_TOKEN": {str(i): t for t, i in EV_TOKEN_TO_ID_BUILD.items()},
    },
    "event_type_vocab": {
        "TYPE_VOCAB_SIZE_MODEL": len(TYPE_TO_ID_BUILD),
        "TYPE_PAD_ID": TYPE_TO_ID_BUILD["<PAD>"], "TYPE_UNK_ID": TYPE_TO_ID_BUILD["<UNK>"],
        "TYPE_CLS_ID": TYPE_TO_ID_BUILD["<CLS>"],
        "TYPE_TO_ID": TYPE_TO_ID_BUILD,
        "TYPE_ID_TO_TYPE": {str(i): t for t, i in TYPE_TO_ID_BUILD.items()},
        "CRP_TYPE_ID_FOR_AUDIT": TYPE_TO_ID_BUILD["crp"],
    },
    "static_categories": {k: v for k, v in STATIC_META_PREPARED.items()},
    "static_continuous_scaler": {
        "columns": list(_cont_cols),
        "scaled_columns": {"index_age_years": {"mean": AGE_MEAN, "std": AGE_STD, "fit_split": "train"}},
        "unscaled_columns": ["townsend_z (already train-standardised)", "townsend_observed (0/1)"],
    },
    "sample_continuous_cols_raw": list(_cont_cols),
}
TARGET_METADATA = {
    "K": K_BUILD,
    "condition_names": list(condition_names),
    "hazard_binning": {
        "BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS),
        "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS),
        "N_BINS": int(haz.N_BINS),
    },
    "death_adjusted_design": {
        "cr_target_labels": {
            str(int(cfg.CR_TARGET_CENSOR)): "censored_or_neither",
            str(int(cfg.CR_TARGET_CONDITION)): "incident_condition_k",
            str(int(cfg.CR_TARGET_DEATH)): "death_before_condition_k",
        },
        "tie_policy": str(cfg.DEATH_TIE_POLICY),
    },
    "definitions": {
        "at_risk": "condition k absent from the diagnosis history on or before the index (one-day tolerance)",
        "t_tilde": "months from the index to the first post-index occurrence of k within administrative follow-up; inf if none",
        "C_months": "months from the index to administrative end of follow-up (end_age), not shortened by death",
        "death_time_months": "months from the index to death when observed after the index within the window; inf otherwise",
        "cr_event_type": "1 = condition k first; 2 = death first (condition not observed before it); 0 = neither",
    },
}
SOURCE_TARGET_CONFIG = {
    "index_design": "one-calendar-month-cap hybrid index (+/-1 month around recruitment)",
    "maximum_calendar_months_from_recruitment": 1,
    "indices_sampled": False,
    "indices_shifted_in_model": False,
}


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
test_samples = SAMPLES["test"].copy()


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


def compare_with_source_index_audit(frame, split):
    """The model rows must carry exactly the index fields of the prepared table."""
    source = prepared[split]
    if list(frame["eid"].astype(str)) != list(source["eid"].astype(str)):
        raise RuntimeError(f"{split}: sample eids differ from the prepared table.")
    if not np.array_equal(frame["index_age_years"].to_numpy(float), source["index_age_years"].to_numpy(float)):
        raise RuntimeError(f"{split}: index ages differ from the prepared table.")
    if not np.array_equal(
        frame["index_offset_days_from_recruitment"].to_numpy(float),
        pd.to_numeric(source["index_offset_days_from_recruitment"]).to_numpy(float),
    ):
        raise RuntimeError(f"{split}: index offsets differ from the prepared table.")
    if (pd.to_datetime(frame["index_dt"]) != pd.to_datetime(source["index_dt"])).any():
        raise RuntimeError(f"{split}: index dates differ from the prepared table.")


for split_name, frame in (
    ("train", train_samples),
    ("val", val_samples),
    ("test", test_samples),
):
    validate_canonical_samples(frame, split_name)
    compare_with_source_index_audit(frame, split_name)

train_eids = set(train_samples["eid"].astype(str))
val_eids = set(val_samples["eid"].astype(str))
test_eids = set(test_samples["eid"].astype(str))
if train_eids & val_eids or train_eids & test_eids or val_eids & test_eids:
    raise RuntimeError("Participant leakage across canonical train/validation/test splits.")

N_CONTINUOUS_STATIC = int(np.asarray(train_samples.iloc[0]["continuous_static"]).size)
if N_CONTINUOUS_STATIC < 1:
    raise ValueError("Canonical continuous_static vectors are empty.")
for split_name, frame in (
    ("train", train_samples),
    ("val", val_samples),
    ("test", test_samples),
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

MODEL_INPUT_HASHES = dict(INPUT_HASHES)
# The sample-file and vocabulary hashes are added when those files are written (next cell).

print(pd.DataFrame([
    {"split": "train", "rows": len(train_samples), "participants": len(train_eids)},
    {"split": "val", "rows": len(val_samples), "participants": len(val_eids)},
    {"split": "test", "rows": len(test_samples), "participants": len(test_eids)},
]).to_string(index=False))
print("Index fields of the model samples match the prepared tables for train, validation, and test.")
print("Inputs read: the prepared train/val/test files only (hash-verified against splits_manifest.json).")


# ## Vocabulary, scalers and targets
#
# The condition list, event/type vocabularies and the index-age scaler are derived from the training split of the newly prepared data only; validation and test tokens that are not in the training vocabulary map to UNK. Targets are built from the prepared diagnosis first-occurrence ages, the administrative end of follow-up and the post-index death fields (definitions are stored in `canonical_target_metadata.json`). Predictors never include anything after the index.
#


# =============================================================================
# Notebook cell 6
# =============================================================================

# =============================================================================
# CELL 4B — WRITE THE MODEL SAMPLES AND PROVENANCE TO THE OUTPUT DIRECTORY
# =============================================================================

# Downstream result/CI cells read canonical_{train,val,test}_samples.pkl from RUN_DIR.
# These files are built from the prepared data in the previous cell and hashed here.
for split_name, frame in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
    output_path = RUN_DIR / f"canonical_{split_name}_samples.pkl"
    frame.to_pickle(output_path)
    MODEL_INPUT_HASHES[f"canonical_{split_name}_samples_sha256"] = sha256_file(output_path)
    _back = pd.read_pickle(output_path)
    if len(_back) != len(frame) or list(_back["eid"]) != list(frame["eid"]):
        raise RuntimeError(f"{split_name} sample pickle did not round-trip.")

def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, default=str)

_write_json(RUN_DIR / "cell3_vocab_scalers_deathsafe.json", VOCAB_METADATA)
_write_json(RUN_DIR / "canonical_target_metadata.json", TARGET_METADATA)
_write_json(RUN_DIR / "cell4_death_adjusted_target_config.json", SOURCE_TARGET_CONFIG)
MODEL_INPUT_HASHES["vocabulary_and_scalers_sha256"] = sha256_file(RUN_DIR / "cell3_vocab_scalers_deathsafe.json")

for split_name, frame in (("train", train_samples), ("val", val_samples), ("test", test_samples)):
    frame[["eid", "index_dt", "index_age_years", "index_offset_days_from_recruitment"]].to_csv(
        RUN_DIR / f"cell4_{split_name}_sample_index_audit.csv", index=False
    )
shutil.copy2(PREPARED_PATHS["manifest"], RUN_DIR / "prepared_splits_manifest.json")
shutil.copy2(PREPARED_PATHS["static_vocabulary"], RUN_DIR / "static_covariate_vocabulary.json")

model_input_manifest = {
    "model_variant": str(cfg.MODEL_VARIANT),
    "input_hashes": dict(MODEL_INPUT_HASHES),
    "paths": {
        "prepared_train": str(PREPARED_PATHS["train"]),
        "prepared_val": str(PREPARED_PATHS["val"]),
        "prepared_test": str(PREPARED_PATHS["test"]),
        "prepared_splits_manifest": str(PREPARED_PATHS["manifest"]),
        "static_covariate_vocabulary": str(PREPARED_PATHS["static_vocabulary"]),
        "canonical_train_samples": str(RUN_DIR / "canonical_train_samples.pkl"),
        "canonical_val_samples": str(RUN_DIR / "canonical_val_samples.pkl"),
        "canonical_test_samples": str(RUN_DIR / "canonical_test_samples.pkl"),
        "vocabulary_and_scalers": str(RUN_DIR / "cell3_vocab_scalers_deathsafe.json"),
        "target_metadata": str(RUN_DIR / "canonical_target_metadata.json"),
    },
    "condition_selection": CONDITION_SELECTION,
    "K": int(K),
    "index_design": str(SOURCE_TARGET_CONFIG["index_design"]),
    "indices_sampled": False,
    "indices_shifted_in_model": False,
    "maximum_calendar_months_from_recruitment": 1,
    "samples_built_from_prepared_tables": True,
}
_write_json(RUN_DIR / "model_input_manifest.json", model_input_manifest)

canonical_source_audit = pd.DataFrame([
    {
        "split": split_name,
        "prepared_file": str(PREPARED_PATHS[split_name]),
        "prepared_file_sha256": INPUT_HASHES[f"prepared_{split_name}_csv_sha256"],
        "sample_pickle": str(RUN_DIR / f"canonical_{split_name}_samples.pkl"),
        "sample_pickle_sha256": MODEL_INPUT_HASHES[f"canonical_{split_name}_samples_sha256"],
        "rows": int(len(frame)),
        "participants": int(frame["eid"].astype(str).nunique()),
        "index_fields_match_prepared_table": True,
    }
    for split_name, frame in (("train", train_samples), ("val", val_samples), ("test", test_samples))
])
canonical_source_audit.to_csv(RUN_DIR / "prepared_data_source_audit.csv", index=False)

print("Prepared inputs locked for CRESCENT:")
print(canonical_source_audit.to_string(index=False))


# =============================================================================
# Notebook cell 8
# =============================================================================

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

assert "cfg" in globals() and "haz" in globals() and "RUN_DIR" in globals(), "Run Cell 1 first."
assert "train_samples" in globals() and "val_samples" in globals() and "test_samples" in globals(), "Run canonical loader Cell 2 first."
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


history_coverage_df = pd.DataFrame(
    [
        summarise_history_coverage(train_samples, "train"),
        summarise_history_coverage(val_samples, "val"),
        summarise_history_coverage(test_samples, "test"),
    ]
)
history_coverage_df.to_csv(
    RUN_DIR / "cell4.5_full_history_coverage_by_split.csv",
    index=False,
)
print("\nFixed-landmark input coverage:")
print(history_coverage_df.to_string(index=False))
if int(history_coverage_df["crp_events_available"].sum()) == 0:
    raise RuntimeError("No CRP events reached the model inputs.")
if int(history_coverage_df["crp_events_with_numeric_value"].sum()) == 0:
    raise RuntimeError("CRP events exist, but none carries an observed numeric value.")
if not history_coverage_df["crp_events_available"].eq(
    history_coverage_df["crp_events_in_input"]
).all():
    raise RuntimeError("At least one prepared CRP event was lost from the model sequence.")
history_coverage_df[[
    "split",
    "participants",
    "crp_events_available",
    "crp_events_in_input",
    "crp_events_with_numeric_value",
    "crp_events_with_missing_value",
    "participants_with_crp_input",
    "participants_dropped_for_crp",
]].to_csv(RUN_DIR / "crp_input_audit.csv", index=False)
print("CRP: timed events, prepared numeric values and masks retained in model inputs.")
if int(history_coverage_df["genuine_post_landmark_inputs"].sum()) != 0:
    raise RuntimeError("A genuine predictor event with model time after the landmark was detected.")
print("Input uses up to 25 selected events plus CLS per participant.")


# =============================================================================
# Build datasets/loaders
# =============================================================================

train_ds = FixedLandmarkSequenceDataset(
    train_samples, split_name="train", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS
)
val_ds = FixedLandmarkSequenceDataset(
    val_samples, split_name="val", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS
)
test_ds = FixedLandmarkSequenceDataset(
    test_samples, split_name="test", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS
)


def _seed_worker(worker_id):
    worker_seed = int(getattr(cfg, "TRAINING_SEED", 101)) + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


train_generator = torch.Generator()
train_generator.manual_seed(int(getattr(cfg, "TRAINING_SEED", 101)))

train_loader = DataLoader(
    train_ds,
    batch_size=int(cfg.BATCH_SIZE),
    shuffle=True,
    num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=True),
    worker_init_fn=_seed_worker if int(cfg.NUM_WORKERS) > 0 else None,
    generator=train_generator,
)
val_loader = DataLoader(
    val_ds,
    batch_size=int(cfg.BATCH_SIZE),
    shuffle=False,
    num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=False),
)
test_loader = DataLoader(
    test_ds,
    batch_size=int(cfg.BATCH_SIZE),
    shuffle=False,
    num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=False),
)


# =============================================================================
# Batch validation
# =============================================================================

def peek_batch(loader, name):
    batch = next(iter(loader))
    B = int(batch["event_ids"].shape[0])
    L = int(batch["event_ids"].shape[1])
    crp_mask = batch["event_type_ids"] == int(CRP_TYPE_ID)
    for token_id in CRP_TOKEN_IDS:
        crp_mask = crp_mask | (batch["event_ids"] == int(token_id))
    observed_crp = crp_mask & (batch["event_value_masks"] >= 0.5)
    if bool(observed_crp.any()):
        assert torch.isfinite(batch["event_values"][observed_crp]).all()
        assert torch.all(batch["event_values"][observed_crp] >= 0.0)

    assert not bool((batch["event_ids"] == int(EV_NEAR_RECRUITMENT_INDEX_ID)).any())

    assert batch["event_ids"].shape == batch["event_type_ids"].shape
    assert batch["event_ids"].shape == batch["event_times_months"].shape
    assert batch["event_ids"].shape == batch["event_values"].shape
    assert batch["event_ids"].shape == batch["event_value_masks"].shape
    assert batch["event_ids"].shape == batch["attn_mask"].shape
    assert torch.all(batch["lengths"] == batch["sequence_real_counts"] + 1)
    assert torch.all(batch["sequence_real_counts"] <= int(FULL_SEQUENCE_MAX_EVENTS))

    for i in range(B):
        length = int(batch["lengths"][i].item())
        assert int(batch["event_ids"][i, length - 1].item()) == int(EV_CLS_ID)
        assert int(batch["event_type_ids"][i, length - 1].item()) == int(TYPE_CLS_ID)
        assert abs(
            float(batch["event_times_months"][i, length - 1].item())
            - float(batch["index_age_months"][i].item())
        ) < 1e-4
        if length > 1:
            assert torch.all(
                batch["event_times_months"][i, : length - 1]
                <= batch["index_age_months"][i] + AGE_TOL_MONTHS + 1e-5
            )

    assert batch["at_risk"].shape == (B, int(K))
    assert batch["cr_at_risk"].shape == (B, int(K))
    assert batch["cr_event_type"].shape == (B, int(K))
    assert torch.all(batch["cr_at_risk"] == batch["at_risk"])
    assert torch.all(torch.isfinite(batch["C_months"])) and torch.all(
        batch["C_months"] > 0
    )

    print(
        f"[{name}] B={B} | input Lmax={L} | zero-event="
        f"{int((batch['sequence_real_counts'] == 0).sum())} | "
        f"post-index deaths={int(batch['death_observed_post_index'].sum())}"
    )
    return batch


_ = peek_batch(train_loader, "train")
_ = peek_batch(val_loader, "val")
_ = peek_batch(test_loader, "test")


# =============================================================================
# Save metadata
# =============================================================================

cell45_meta = {
    "dataset_class": "FixedLandmarkSequenceDataset",
    "collate": "fixed_landmark_sequence_collate",
    "participant_split_unchanged": True,
    "one_locked_landmark_per_participant": True,
    "landmark_source": str(PREPARED_DATA_DIR),
    "canonical_train_samples": str(RUN_DIR / "canonical_train_samples.pkl"),
    "canonical_val_samples": str(RUN_DIR / "canonical_val_samples.pkl"),
    "canonical_test_samples": str(RUN_DIR / "canonical_test_samples.pkl"),
    "landmark_sha256": INPUT_HASHES["near_index_key_sha256"],
    "samples_built_from_prepared_tables": True,
    "raw_sources_opened": False,
    "post_landmark_predictors_allowed": False,
    "post_landmark_predictor_semantics": (
        "no model or actual event time exceeds the canonical participant-specific index"
    ),
    "post_fixed_index_observations_allowed": False,
    "backdating_allowed": False,
    "context_cutoff": "recruitment",
    "crp_in_model_inputs": True,
    "crp_representation": "timed event with prepared numeric value and observedness mask",
    "input_ablation": None,
    "crp_retained_in_input_when_present": True,
    "interaction_sequence_policy": "deterministic_most_recent_events",
    "full_sequence_max_events": int(FULL_SEQUENCE_MAX_EVENTS),
    "full_sequence_max_len_with_cls": int(FULL_SEQUENCE_MAX_LEN),
    "artificial_landmark_used_as_clinical_event": False,
    "BATCH_SIZE": int(cfg.BATCH_SIZE),
    "NUM_WORKERS": int(cfg.NUM_WORKERS),
    "K": int(K),
    "N_CONTINUOUS_STATIC": int(N_CONTINUOUS_STATIC),
    "EV_PAD_ID": int(EV_PAD_ID),
    "EV_CLS_ID": int(EV_CLS_ID),
    "EV_NEAR_RECRUITMENT_INDEX_ID": int(EV_NEAR_RECRUITMENT_INDEX_ID),
    "TYPE_PAD_ID": int(TYPE_PAD_ID),
    "TYPE_CLS_ID": int(TYPE_CLS_ID),
    "history_coverage": history_coverage_df.to_dict("records"),
    "death_adjusted_targets": {
        "enabled": bool(getattr(cfg, "USE_DEATH_HEAD", True)),
        "cr_event_type": {
            str(CR_TARGET_CENSOR): "censored_or_neither",
            str(CR_TARGET_CONDITION): "incident_condition_k",
            str(CR_TARGET_DEATH): "death_before_condition_k",
        },
        "conditions_compete_with_each_other": False,
        "death_shared_across_conditions": True,
    },
}
with open(RUN_DIR / "cell4.5_full_history_dataloader_metadata.json", "w") as handle:
    json.dump(cell45_meta, handle, indent=2, default=str)
# Preserve the historical metadata filename for downstream tooling.
with open(RUN_DIR / "cell4.5_dataloader_metadata_death_adjusted.json", "w") as handle:
    json.dump(cell45_meta, handle, indent=2, default=str)

print("\nSaved Cell 4.5 metadata:")
print(" ", RUN_DIR / "cell4.5_full_history_dataloader_metadata.json")
print(" ", RUN_DIR / "cell4.5_full_history_coverage_by_split.csv")
print(
    "\n✅ Cell 4.5 complete — canonical landmarks and targets retained; "
    "up to 25 pre-index events, including CRP when present, form each input."
)


# ## Model architecture
# 
# The 25-event sequence feeds the three-layer Transformer and condition reader. Train-only global event/exposure estimates initialize the condition and death rate intercepts. The optional condition identity path retains each disease query in the reader output and supplies a learnable disease-specific baseline input.
#
