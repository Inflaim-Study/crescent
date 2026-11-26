# Gemma 4 E4B (frozen encoder) — Canonical train/validation loading, datasets, and cached frozen Gemma state loaders
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 6 ---
# CELL 2 — LOAD AND AUDIT THE CRESCENT MODEL SAMPLES (hash-verified, byte-identical copies)
# Reads only files written by the completed revised CRESCENT run. No raw CSV, split JSON or other source is opened.
# The test pickle is hashed and copied (bytes only); it is parsed only in the `evaluate` phase.

MANIFEST_PATH = CRESCENT_RUN_DIR / "model_input_manifest.json"
REQUIRED_CRESCENT_FILES = [
    "canonical_train_samples.pkl", "canonical_val_samples.pkl", "canonical_test_samples.pkl",
    "model_input_manifest.json", "cell3_vocab_scalers_deathsafe.json", "canonical_target_metadata.json",
    "config.json", "prepared_splits_manifest.json", "static_covariate_vocabulary.json",
    "cell4_death_adjusted_target_config.json", "prepared_data_source_audit.csv",
    "cell4_train_sample_index_audit.csv", "cell4_val_sample_index_audit.csv", "cell4_test_sample_index_audit.csv",
]
_missing = [name for name in REQUIRED_CRESCENT_FILES if not (CRESCENT_RUN_DIR / name).is_file()]
if _missing:
    raise FileNotFoundError("The CRESCENT run directory is missing: " + ", ".join(_missing))

CRESCENT_MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
MODEL_INPUT_HASHES = dict(CRESCENT_MANIFEST["input_hashes"])
_HASH_FILES = {
    "canonical_train_samples_sha256": "canonical_train_samples.pkl",
    "canonical_val_samples_sha256": "canonical_val_samples.pkl",
    "canonical_test_samples_sha256": "canonical_test_samples.pkl",
    "vocabulary_and_scalers_sha256": "cell3_vocab_scalers_deathsafe.json",
}
# Verify every pickle and the vocabulary file against the manifest BEFORE parsing anything.
OBSERVED_INPUT_HASHES = {}
for _key, _name in _HASH_FILES.items():
    OBSERVED_INPUT_HASHES[_key] = sha256_file(CRESCENT_RUN_DIR / _name)
    if str(MODEL_INPUT_HASHES.get(_key)) != OBSERVED_INPUT_HASHES[_key]:
        raise RuntimeError(f"{_name} does not match model_input_manifest.json (expected {MODEL_INPUT_HASHES.get(_key)}, observed {OBSERVED_INPUT_HASHES[_key]}).")

# Byte-identical copies in the comparator run directory (CRESCENT's section 9-11 code reads them from RUN_DIR).
COPIED_FILES = [name for name in REQUIRED_CRESCENT_FILES if name != "canonical_test_samples.pkl"]   # test bytes are copied only for evaluation
COPY_HASH_AUDIT = []
for _name in COPIED_FILES:
    _source, _target = CRESCENT_RUN_DIR / _name, RUN_DIR / _name
    if not _target.exists() or sha256_file(_target) != sha256_file(_source):
        shutil.copy2(_source, _target)
    _ok = sha256_file(_target) == sha256_file(_source)
    COPY_HASH_AUDIT.append({"file": _name, "sha256": sha256_file(_source), "copy_identical": bool(_ok)})
    if not _ok:
        raise RuntimeError(f"Copy of {_name} is not byte-identical.")
pd.DataFrame(COPY_HASH_AUDIT).to_csv(RUN_DIR / "gemma_input_copy_audit.csv", index=False)

VOCAB_METADATA = json.loads((RUN_DIR / "cell3_vocab_scalers_deathsafe.json").read_text(encoding="utf-8"))
TARGET_METADATA = json.loads((RUN_DIR / "canonical_target_metadata.json").read_text(encoding="utf-8"))
SOURCE_TARGET_CONFIG = json.loads((RUN_DIR / "cell4_death_adjusted_target_config.json").read_text(encoding="utf-8"))

# K and the condition order are read from the metadata, never hard-coded.
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



if TARGET_METADATA["hazard_binning"] != {"BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS), "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS), "N_BINS": int(haz.N_BINS)}:
    raise RuntimeError("Hazard binning in the CRESCENT metadata differs from the shared evaluation axis.")

train_samples = pd.read_pickle(RUN_DIR / "canonical_train_samples.pkl")
val_samples = pd.read_pickle(RUN_DIR / "canonical_val_samples.pkl")
test_samples = None  # test access is deferred until both full arm locks have been verified

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





def compare_with_crescent_index_audit(frame, split):
    """The comparator rows must carry exactly the index fields written by the CRESCENT run."""
    audit = pd.read_csv(CRESCENT_RUN_DIR / f"cell4_{split}_sample_index_audit.csv", dtype={"eid": str})
    if list(frame["eid"].astype(str)) != list(audit["eid"].astype(str)):
        raise RuntimeError(f"{split}: sample eids differ from the CRESCENT index audit.")
    if not np.allclose(frame["index_age_years"].to_numpy(float), audit["index_age_years"].to_numpy(float), rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{split}: index ages differ from the CRESCENT index audit.")
    if not np.allclose(frame["index_offset_days_from_recruitment"].to_numpy(float),
                       audit["index_offset_days_from_recruitment"].to_numpy(float), rtol=0.0, atol=1e-9):
        raise RuntimeError(f"{split}: index offsets differ from the CRESCENT index audit.")
    if (pd.to_datetime(frame["index_dt"]) != pd.to_datetime(audit["index_dt"])).any():
        raise RuntimeError(f"{split}: index dates differ from the CRESCENT index audit.")


_loaded = [("train", train_samples), ("val", val_samples)] + ([("test", test_samples)] if test_samples is not None else [])
for split_name, frame in _loaded:
    validate_canonical_samples(frame, split_name)
    compare_with_crescent_index_audit(frame, split_name)

train_eids = set(train_samples["eid"].astype(str))
val_eids = set(val_samples["eid"].astype(str))
test_eids = set(test_samples["eid"].astype(str)) if test_samples is not None else set()
if train_eids & val_eids or train_eids & test_eids or val_eids & test_eids:
    raise RuntimeError("Participant leakage across canonical train/validation/test splits.")

N_CONTINUOUS_STATIC = int(np.asarray(train_samples.iloc[0]["continuous_static"]).size)
if N_CONTINUOUS_STATIC < 1:
    raise ValueError("Canonical continuous_static vectors are empty.")
for split_name, frame in _loaded:
    if frame["continuous_static"].map(lambda value: np.asarray(value).size != N_CONTINUOUS_STATIC).any():
        raise ValueError(f"{split_name}: inconsistent continuous_static width.")

CR_TARGET_LABELS = TARGET_METADATA["death_adjusted_design"]["cr_target_labels"]
if (int(next(k for k, v in CR_TARGET_LABELS.items() if "censor" in v or "neither" in v)) != int(cfg.CR_TARGET_CENSOR)
        or int(next(k for k, v in CR_TARGET_LABELS.items() if "condition" in v)) != int(cfg.CR_TARGET_CONDITION)
        or int(next(k for k, v in CR_TARGET_LABELS.items() if "death" in v)) != int(cfg.CR_TARGET_DEATH)):
    raise RuntimeError("Canonical competing-risk labels differ from the shared configuration.")

AGE_MEAN = float(STATIC_CONTINUOUS_SCALER["scaled_columns"]["index_age_years"]["mean"])
AGE_STD = float(STATIC_CONTINUOUS_SCALER["scaled_columns"]["index_age_years"]["std"])

# Acceptance test A1 (data identity): hashes, eid order, K and condition order equal CRESCENT's.
ACCEPTANCE = {}
ACCEPTANCE["A1_data_identity"] = {
    "pass": True,
    "input_hashes_verified": OBSERVED_INPUT_HASHES,
    "K": int(K),
    "condition_names_sha256": hashlib.sha256("\n".join(condition_names).encode()).hexdigest(),
    "n_train": int(len(train_samples)), "n_val": int(len(val_samples)),
    "n_test": int(len(test_samples)) if test_samples is not None else None,
    "test_pickle_parsed": bool(test_samples is not None),
}

print(pd.DataFrame([
    {"split": s, "rows": len(f), "participants": f["eid"].astype(str).nunique()} for s, f in _loaded
]).to_string(index=False))
print(f"K = {K} conditions (read from canonical_target_metadata.json).")
print("Inputs: model samples copied byte-for-byte from the CRESCENT run and verified against model_input_manifest.json.")

initialize_lora_budget()

# --- Notebook cell 7 ---
GEMMA_ENCODER_IDENTITY={}
ADAPTATION_REGISTER={'schema':'gemma.adaptation.v7','arm':ARM,'source':'user supplied V6 notebooks; revised decoder classes retained',
 'pipeline_code_sha256':PIPELINE_CODE_SHA256,
 'changes':['two matched CRESCENT-decoder arms only','input masking disabled in both Gemma arms; report a matched no-masking CRESCENT control',
            'high-precision canonical text with explicit category mapping','cache bound before extraction to data, encoder, tokenizer, code and framework',
            'effective-batch count normalization','raw last checkpoint plus matching optimizer, EMA, RNG and sampler state',
            'complete per-arm grid/final selection inside the training notebook','test rows inaccessible before both arm locks',
            'full validation every 5000 updates independent of NLL cadence','single-device all-layer LoRA; base Gemma remains frozen'],
 'interpretation':'secondary encoder/adaptation comparison; neither arm is an independent published survival architecture'}
atomic_json(RUN_DIR/'gemma_adaptation_register.json',ADAPTATION_REGISTER)
ADAPTATION_REGISTER_SHA256=sha256_file(RUN_DIR/'gemma_adaptation_register.json')
SHARED_CODE_MANIFEST={'pipeline_code_sha256':PIPELINE_CODE_SHA256,'decoder_reference_notebook_sha256':CRESCENT_NOTEBOOK_SHA256,
 'task':'source-matched six-month entry','training':'V7 stateful accumulation and selection; not claimed verbatim',
 'evaluation':'legacy CRESCENT evaluation functions retained with phase guards and V7 checkpoint loader'}
atomic_json(RUN_DIR/'shared_code_manifest.json',SHARED_CODE_MANIFEST)

def gemma_settings_block():
    return {'study_version':'V7','arm':ARM,'phase':PHASE,'seed':SEED,'hpid':HPID,'encoder':dict(GEMMA_ENCODER_IDENTITY),
            'layer_choice':LAYER_CHOICE,'parent_arm':'B','lora_rank':LORA_RANK if ARM=='C' else None,
            'lora_alpha':LORA_ALPHA if ARM=='C' else None,'lora_dropout':LORA_DROPOUT if ARM=='C' else None,
            'adapter_lr':ADAPTER_LR if ARM=='C' else None,'effective_batch':EFFECTIVE_BATCH if ARM=='C' else cfg.BATCH_SIZE,
            'pipeline_code_sha256':PIPELINE_CODE_SHA256,'text_contract':TEXT_CONTRACT_VERSION,
            'crescent_run_dir':str(CRESCENT_RUN_DIR),'selected_hyperparameters':_plain(asdict(cfg))}

def write_config_json():
    atomic_json(RUN_DIR/'config.json',{'model_variant':cfg.MODEL_VARIANT,'config':asdict(cfg),
                'prepared_data_source':{'crescent_run_dir':str(CRESCENT_RUN_DIR),'model_input_hashes':dict(MODEL_INPUT_HASHES)},
                'fixed_index_contract':CRESCENT_CONFIG.get('fixed_index_contract',{}),
                'hazard_binning':{'BIN_WIDTH_MONTHS':haz.BIN_WIDTH_MONTHS,'MAX_HORIZON_MONTHS':haz.MAX_HORIZON_MONTHS,'N_BINS':haz.N_BINS},
                'delayed_entry_contract':delayed_entry_contract(),'gemma':gemma_settings_block()})
write_config_json()

# --- Notebook cell 9 ---
# CELL 4 — DATASETS AND DATALOADERS (CRESCENT's own dataset, collate and history-selection code, verbatim)
from torch.utils.data import Dataset, DataLoader

CR_TARGET_CENSOR = int(getattr(cfg, "CR_TARGET_CENSOR", 0))
CR_TARGET_CONDITION = int(getattr(cfg, "CR_TARGET_CONDITION", 1))
CR_TARGET_DEATH = int(getattr(cfg, "CR_TARGET_DEATH", 2))
FULL_SEQUENCE_MAX_EVENTS = int(getattr(cfg, "FULL_SEQUENCE_MAX_EVENTS", 25))
FULL_SEQUENCE_MAX_LEN = FULL_SEQUENCE_MAX_EVENTS + 1  # one appended CLS token
AGE_TOL_MONTHS = float(getattr(cfg, "AGE_TOL_YEARS", 1.0 / 365.25)) * 12.0
TIME_EPS_MONTHS = 1e-6

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
    [summarise_history_coverage(frame, name)
     for name, frame in (("train", train_samples), ("val", val_samples), ("test", test_samples)) if frame is not None]
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


# ---- datasets and loaders (as in CRESCENT; the test loader exists only in the evaluate phase) ---------------------
train_ds = FixedLandmarkSequenceDataset(train_samples, split_name="train", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS)
val_ds = FixedLandmarkSequenceDataset(val_samples, split_name="val", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS)
test_ds = (FixedLandmarkSequenceDataset(test_samples, split_name="test", full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS)
           if test_samples is not None else None)

def _seed_worker(worker_id):
    worker_seed = int(getattr(cfg, "TRAINING_SEED", 101)) + int(worker_id)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

train_generator = torch.Generator()
train_generator.manual_seed(int(getattr(cfg, "TRAINING_SEED", 101)))

train_loader = DataLoader(
    train_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=True, num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=True),
    worker_init_fn=_seed_worker if int(cfg.NUM_WORKERS) > 0 else None, generator=train_generator,
)
val_loader = DataLoader(
    val_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=False),
)
test_loader = (DataLoader(
    test_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=int(cfg.NUM_WORKERS),
    pin_memory=torch.cuda.is_available(),
    collate_fn=lambda rows: fixed_landmark_sequence_collate(rows, training=False),
) if test_ds is not None else None)

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
if test_loader is not None:
    _ = peek_batch(test_loader, "test")

STEPS_PER_EPOCH = int(len(train_loader))
if STEPS_PER_EPOCH < 1:
    raise RuntimeError("Training loader is empty.")
write_config_json()

cell45_meta = {
    "dataset_class": "FixedLandmarkSequenceDataset",
    "collate": "fixed_landmark_sequence_collate",
    "participant_split_unchanged": True,
    "one_locked_landmark_per_participant": True,
    "landmark_source": str(CRESCENT_RUN_DIR),
    "canonical_train_samples": str(RUN_DIR / "canonical_train_samples.pkl"),
    "canonical_val_samples": str(RUN_DIR / "canonical_val_samples.pkl"),
    "canonical_test_samples": str(RUN_DIR / "canonical_test_samples.pkl"),
    "landmark_sha256": MODEL_INPUT_HASHES.get("near_index_key_sha256"),
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




# ---- acceptance tests A2 (input identity, independent re-derivation) and A3 (no leakage) ------------------------
def _independent_sequence(row, max_events=25):
    ids = np.asarray(row["history_event_ids"], dtype=np.int64)
    types = np.asarray(row["history_event_type_ids"], dtype=np.int64)
    times = np.asarray(row["history_event_times_months"], dtype=np.float64)
    values = np.asarray(row["history_event_values"], dtype=np.float32)
    masks = np.asarray(row["history_event_value_masks"], dtype=np.float32)
    keep = np.isfinite(times) & ~np.isin(ids, [EV_NEAR_RECRUITMENT_INDEX_ID, EV_CLS_ID, EV_PAD_ID])
    order = np.lexsort((types[keep], ids[keep], times[keep]))
    ids, types, times = ids[keep][order], types[keep][order], times[keep][order]
    values, masks = np.nan_to_num(values[keep][order]), (masks[keep][order] >= 0.5).astype(np.float32)
    if len(ids) > max_events:
        sel = np.arange(len(ids) - max_events, len(ids))
        crp = np.flatnonzero((types == CRP_TYPE_ID) | np.isin(ids, CRP_TOKEN_IDS))
        if crp.size and int(crp[-1]) not in sel:
            sel[0] = int(crp[-1]); sel.sort()
        ids, types, times, values, masks = ids[sel], types[sel], times[sel], values[sel], masks[sel]
    return ids, np.minimum(times, float(row["index_age_months"])), values, masks


def acceptance_input_identity(samples, split, n=1000):
    frame = samples.reset_index(drop=True).iloc[: int(n)]
    dataset = FixedLandmarkSequenceDataset(frame, split_name=split, full_sequence_max_events=FULL_SEQUENCE_MAX_EVENTS)
    mismatches, max_leak = 0, -np.inf
    for start in range(0, len(frame), 256):
        rows = [dataset[i] for i in range(start, min(start + 256, len(frame)))]
        batch = fixed_landmark_sequence_collate(rows, training=False)
        for local, i in enumerate(range(start, min(start + 256, len(frame)))):
            row = frame.iloc[i]
            ids, times, values, masks = _independent_sequence(row, FULL_SEQUENCE_MAX_EVENTS)
            length = int(batch["lengths"][local])
            ok = length == len(ids) + 1
            ok &= np.array_equal(batch["event_ids"][local, : len(ids)].numpy(), ids)
            ok &= np.allclose(batch["event_times_months"][local, : len(ids)].numpy(), times.astype(np.float32), atol=1e-4)
            ok &= np.allclose(batch["event_values"][local, : len(ids)].numpy(), values, atol=1e-6)
            ok &= np.array_equal(batch["event_value_masks"][local, : len(ids)].numpy(), masks)
            ok &= int(batch["event_ids"][local, length - 1]) == int(EV_CLS_ID)
            ok &= np.allclose(batch["continuous_static"][local].numpy(), np.asarray(row["continuous_static"], dtype=np.float32), atol=1e-7)
            ok &= int(batch["gender_idx"][local]) == int(np.clip(int(row["gender_idx"]), 0, int(N_GENDER_MODEL) - 1))
            mismatches += int(not ok)
            if len(ids):
                max_leak = max(max_leak, float((times - float(row["index_age_months"])).max()))
    return {"participants_checked": int(len(frame)), "mismatches": int(mismatches),
            "max_event_time_minus_index_age_months": None if not np.isfinite(max_leak) else float(max_leak)}


_a2 = {name: acceptance_input_identity(frame, name) for name, frame in _loaded}
ACCEPTANCE["A2_input_identity"] = {"pass": all(v["mismatches"] == 0 for v in _a2.values()), "splits": _a2}
ACCEPTANCE["A3_no_leakage"] = {
    "pass": bool(all((v["max_event_time_minus_index_age_months"] or -1.0) <= AGE_TOL_MONTHS + 1e-5 for v in _a2.values())
                 and (PHASE == "evaluate" or test_samples is None)),
    "age_tolerance_months": float(AGE_TOL_MONTHS), "test_pickle_parsed_before_lock": bool(PHASE == "train" and test_samples is not None),
}
for _name in ("A2_input_identity", "A3_no_leakage"):
    if not ACCEPTANCE[_name]["pass"]:
        raise RuntimeError(f"Acceptance test {_name} failed: {ACCEPTANCE[_name]}")
print("A2 input identity and A3 no-leakage passed:", {k: v["participants_checked"] for k, v in _a2.items()})
print(f"Steps per epoch = {STEPS_PER_EPOCH:,} (batch {cfg.BATCH_SIZE}); validation every {cfg.VALIDATION_INTERVAL:,} steps; maximum {cfg.MAX_ITERATIONS:,} steps (CRESCENT recipe).")

# --- Notebook cell 12 ---
# CELL 5 — CACHED GEMMA STATES (frozen encoder) AND LOADERS
# The extraction notebook wrote, per split, one cached state per CRESCENT sequence position (every event line, then the index line, which plays the
# role of CRESCENT's CLS token) and one mean-pooled vector, for two layers. Here they are attached to CRESCENT's own dataset items and collate.
from torch.utils.data import Dataset as _Dataset

_identity_path = CACHE_DIR / "encoder_identity.json"
if not _identity_path.is_file():
    raise FileNotFoundError(f"Run notebook 01 first: {_identity_path} is missing.")
CACHE_IDENTITY = json.loads(_identity_path.read_text(encoding="utf-8"))
GEMMA_ENCODER_IDENTITY.update({k: v for k, v in CACHE_IDENTITY.items() if k != "file_sha256"})
GEMMA_ENCODER_IDENTITY["file_sha256_table_in_cache_identity"] = str(_identity_path)
D_GEMMA = int(CACHE_IDENTITY["d_model"])
CACHE_BINDING_SHA256=bind_or_check_cache(CACHE_IDENTITY)
_needed_splits = ["train", "val"]
for _split in _needed_splits:
    if not (CACHE_DIR / f"{_split}_cache_meta.json").is_file():
        raise FileNotFoundError(f"Missing completed cache for {_split}: extract it with notebook 01 (GEMMA_EXTRACT_SPLITS includes {_split}).")
CACHE_META = {s: json.loads((CACHE_DIR / f"{s}_cache_meta.json").read_text()) for s in _needed_splits}
for _split, _meta in CACHE_META.items():
    if (_meta["encoder_fingerprint_sha256"] != CACHE_IDENTITY["fingerprint_sha256"]
            or _meta["text_serialisation_sha256"] != CACHE_IDENTITY["text_serialisation_sha256"]):
        raise RuntimeError(f"The {_split} cache was written with a different encoder or text serialisation than the other splits.")
for _s,_m in CACHE_META.items():
    if _m.get("cache_binding_sha256")!=CACHE_BINDING_SHA256:raise ValueError("Split cache binding differs.")
if len({m["text_serialisation_sha256"] for m in CACHE_META.values()}) != 1:
    raise RuntimeError("Splits were serialised by different contracts.")
write_config_json()


# GEMMA_CACHE_IN_RAM=1 (default) reads the chosen layer's cache files into memory once, instead of one random filesystem read per
# participant per step. The values are byte-identical to the memmap; only the access path changes. Set 0 to use the memmap.
CACHE_IN_RAM = os.getenv("GEMMA_CACHE_IN_RAM", "1").strip().lower() in {"1", "true", "yes"}


class CacheReader:
    """float16 cache access: loaded into RAM once (default) or lazily memory-mapped (safe with DataLoader worker processes)."""

    def __init__(self, split, layer_tag):
        self.split, self.tag = split, layer_tag
        self.meta = CACHE_META[split]
        self.offsets = np.load(CACHE_DIR / f"{split}_row_offsets.npy")
        self.eids = [str(e) for e in np.load(CACHE_DIR / f"{split}_tokens.npz")["eids"]]
        self._states = self._pool = None

    def _open(self):
        if self._states is None:
            mode = None if CACHE_IN_RAM else "r"
            t0 = time.time()
            self._states = np.load(CACHE_DIR / self.meta["files"][f"event_states_{self.tag}"], mmap_mode=mode)
            self._pool = np.load(CACHE_DIR / self.meta["files"][f"mean_pool_{self.tag}"], mmap_mode=mode)
            if CACHE_IN_RAM:
                print(f"[{self.split}] {self.tag}-layer cache loaded into RAM: {(self._states.nbytes + self._pool.nbytes) / 1e9:.1f} GB "
                      f"in {time.time() - t0:.0f} s", flush=True)

    def rows(self, i):
        self._open()
        return np.asarray(self._states[self.offsets[i]: self.offsets[i + 1]], dtype=np.float32)

    def pooled(self, i):
        self._open()
        return np.asarray(self._pool[i], dtype=np.float32)

    def __getstate__(self):
        state = dict(self.__dict__); state["_states"] = state["_pool"] = None
        return state


class CachedDataset(_Dataset):
    """CRESCENT's FixedLandmarkSequenceDataset item + the frozen-Gemma states of the same participant."""

    def __init__(self, base, split):
        self.base, self.split = base, split
        self.reader = CacheReader(split, LAYER_CHOICE)
        base_eids = [str(e) for e in base.samples["eid"]]
        if base_eids != self.reader.eids:
            raise RuntimeError(f"{split}: participant order in the cache differs from the CRESCENT samples.")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        rows = self.reader.rows(int(idx))
        n = int(item["sequence_event_ids"].numel())
        if rows.shape[0] != n + 1:
            raise RuntimeError(f"{self.split} participant {item['eid']}: cache has {rows.shape[0]} rows, CRESCENT sequence has {n}+1.")
        item["gemma_rows"] = torch.from_numpy(rows)
        item["gemma_mean_pool"] = torch.from_numpy(self.reader.pooled(int(idx)))
        return item


def cached_collate(rows, training=False):
    """CRESCENT's collate, plus padded Gemma states aligned with its positions and a mask of positions replaced by UNK."""
    batch = fixed_landmark_sequence_collate(rows, training=training)
    B, L = batch["event_ids"].shape
    states = torch.zeros(B, L, D_GEMMA, dtype=torch.float32)
    for b, item in enumerate(rows):
        states[b, : item["gemma_rows"].shape[0]] = item["gemma_rows"]
    batch["gemma_states"] = states
    batch["gemma_index_state"] = states[torch.arange(B), batch["lengths"] - 1]
    batch["gemma_mean_pool"] = torch.stack([item["gemma_mean_pool"] for item in rows])
    if training and float(cfg.EVENT_MASK_PROB) > 0.0:
        reference = fixed_landmark_sequence_collate(rows, training=False)["event_ids"]
        batch["event_masked"] = ((batch["event_ids"] != reference) & (batch["attn_mask"] > 0.5)).float()
    else:
        batch["event_masked"] = torch.zeros(B, L, dtype=torch.float32)
    return batch


cached_train_ds = CachedDataset(train_ds, "train")
cached_val_ds = CachedDataset(val_ds, "val")
cached_test_ds = CachedDataset(test_ds, "test") if test_ds is not None else None
train_loader = DataLoader(cached_train_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=True, num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                          collate_fn=lambda rows: cached_collate(rows, training=True), worker_init_fn=_seed_worker if int(cfg.NUM_WORKERS) > 0 else None,
                          generator=train_generator)
val_loader = DataLoader(cached_val_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                        collate_fn=lambda rows: cached_collate(rows, training=False))
test_loader = (DataLoader(cached_test_ds, batch_size=int(cfg.BATCH_SIZE), shuffle=False, num_workers=int(cfg.NUM_WORKERS), pin_memory=torch.cuda.is_available(),
                          collate_fn=lambda rows: cached_collate(rows, training=False)) if cached_test_ds is not None else None)

# ---- acceptance G4: cache alignment (every position, mask handling, pad rows) ----------------------------------------------
_probe = next(iter(val_loader))
_ok = True
for _b in range(int(_probe["lengths"].shape[0])):
    _L = int(_probe["lengths"][_b])
    _ok &= bool(torch.count_nonzero(_probe["gemma_states"][_b, _L:]) == 0)
    _ok &= bool(torch.equal(_probe["gemma_index_state"][_b], _probe["gemma_states"][_b, _L - 1]))
    _ok &= bool(torch.isfinite(_probe["gemma_states"][_b, :_L]).all())
_train_probe = next(iter(train_loader))
ACCEPTANCE["G4_cache_alignment"] = {
    "pass": bool(_ok and _probe["gemma_states"].shape[-1] == D_GEMMA),
    "d_gemma": D_GEMMA, "layer": LAYER_CHOICE, "val_probe_batch": int(_probe["lengths"].shape[0]),
    "masked_positions_in_train_probe": int(_train_probe["event_masked"].sum()), "event_mask_prob": float(cfg.EVENT_MASK_PROB),
}
if not ACCEPTANCE["G4_cache_alignment"]["pass"]:
    raise RuntimeError(f"G4 failed: {ACCEPTANCE['G4_cache_alignment']}")
print("G4 cache alignment passed:", ACCEPTANCE["G4_cache_alignment"])
