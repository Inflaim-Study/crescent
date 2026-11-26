# Delphi-2M-inspired — Canonical contract loading, event-stream sequences, and autoregressive dataset
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 8 ---
CANONICAL_REQUIRED_NOW = (
    CANONICAL_PATHS["train"],
    CANONICAL_PATHS["val"],
    CANONICAL_PATHS["test"],
    CANONICAL_PATHS["target_metadata"],
    CANONICAL_PATHS["target_config"],
    CANONICAL_PATHS["vocab_scalers"],
    CANONICAL_PATHS["input_manifest"],
    CANONICAL_PATHS["main_config"],
)
require_files(CANONICAL_REQUIRED_NOW)

with CANONICAL_PATHS["target_metadata"].open(encoding="utf-8") as handle:
    TARGET_METADATA = json.load(handle)
with CANONICAL_PATHS["vocab_scalers"].open(encoding="utf-8") as handle:
    VOCAB_METADATA = json.load(handle)
with CANONICAL_PATHS["input_manifest"].open(encoding="utf-8") as handle:
    INPUT_MANIFEST = json.load(handle)
if CANONICAL_PATHS["dataloader_metadata"].is_file():
    with CANONICAL_PATHS["dataloader_metadata"].open(encoding="utf-8") as handle:
        DATALOADER_METADATA = json.load(handle)
else:
    DATALOADER_METADATA = {"full_sequence_max_events": 25, "artificial_landmark_used_as_clinical_event": False}
with CANONICAL_PATHS["target_config"].open(encoding="utf-8") as handle:
    TARGET_CONFIG = json.load(handle)
with CANONICAL_PATHS["main_config"].open(encoding="utf-8") as handle:
    MAIN_CONFIG = json.load(handle)

for label, payload in (
    ("target metadata", TARGET_METADATA),
    ("vocabulary/scaler metadata", VOCAB_METADATA),
    ("input manifest", INPUT_MANIFEST),
    ("dataloader metadata", DATALOADER_METADATA),
    ("target config", TARGET_CONFIG),
    ("main config", MAIN_CONFIG),
):
    if not isinstance(payload, dict):
        raise RuntimeError(f"Canonical {label} must be a JSON object.")


def manifest_index_binding(manifest: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Resolve the native index path/hash names without changing the index."""
    paths = manifest.get("paths", {})
    hashes = manifest.get("input_hashes", {})
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise RuntimeError("The main manifest has invalid paths/input_hashes objects.")

    preferred = (
        ("latest_covariate_index_key", ("latest_covariate_index_key_sha256",)),
        (
            "fixed_six_month_index_key",
            (
                "fixed_index_key_sha256",
                "fixed_six_month_index_key_sha256",
                "near_index_key_sha256",
            ),
        ),
        (
            "hybrid_index_key",
            ("hybrid_index_key_sha256", "near_index_key_sha256"),
        ),
        (
            "hybrid_landmark_index_key",
            ("hybrid_landmark_index_key_sha256", "near_index_key_sha256"),
        ),
        ("near_index_key", ("near_index_key_sha256",)),
        ("index_key", ("index_key_sha256",)),
    )
    for path_key, hash_keys in preferred:
        if not paths.get(path_key):
            continue
        for hash_key in hash_keys:
            if hashes.get(hash_key):
                return path_key, hash_key

    path_keys = sorted(
        key
        for key, value in paths.items()
        if value and "index" in str(key).lower() and "audit" not in str(key).lower()
    )
    hash_keys = sorted(
        key
        for key, value in hashes.items()
        if value and "index" in str(key).lower() and str(key).endswith("_sha256")
    )
    if not path_keys or not hash_keys:
        return None, None
    return path_keys[0], hash_keys[0]


EXPECTED_HASHES = INPUT_MANIFEST.get("input_hashes", {})
if not isinstance(EXPECTED_HASHES, dict):
    raise RuntimeError("The main manifest input_hashes field must be an object.")

# The target config and top-level run config contain the corrected semantic
# index contract. The input manifest's path/hash binding remains authoritative
# for byte identity, but this trained run retained stale legacy names/text such
# as `fixed_six_month_index_key`; those labels must not redefine the index.
MANIFEST_INDEX_DESIGN = str(INPUT_MANIFEST.get("index_design", "")).strip()
TARGET_INDEX_DESIGN = str(TARGET_CONFIG.get("index_design", "")).strip()
MAIN_CONFIG_INDEX_DESIGN = str(
    MAIN_CONFIG.get("fixed_index_contract", {}).get("design", "")
).strip()
CANONICAL_INDEX_DESIGN = TARGET_INDEX_DESIGN or MAIN_CONFIG_INDEX_DESIGN
if not CANONICAL_INDEX_DESIGN or not MAIN_CONFIG_INDEX_DESIGN:
    raise RuntimeError(
        "The target config and main config must both declare the canonical index design."
    )
if int(TARGET_CONFIG.get("maximum_calendar_months_from_recruitment", 1)) != 1:
    raise RuntimeError(
        "The target config must declare maximum_calendar_months_from_recruitment=1."
    )
if int(
    MAIN_CONFIG.get("fixed_index_contract", {}).get(
        "calendar_months_from_recruitment", -1
    )
) != 1:
    raise RuntimeError(
        "The main config must declare calendar_months_from_recruitment=1."
    )
if TARGET_CONFIG.get("indices_sampled") is not False:
    raise RuntimeError("The target config must declare indices_sampled=false.")
if TARGET_CONFIG.get("indices_shifted_in_model") is not False:
    raise RuntimeError(
        "The target config must declare indices_shifted_in_model=false."
    )
_loaded_entry = int(
    MAIN_CONFIG.get("config", {}).get(
        "ENTRY_DELAY_MONTHS",
        MAIN_CONFIG.get("config", {}).get("GAP_WINDOW_MONTHS", CANONICAL_ENTRY_DELAY_MONTHS),
    )
)
if _loaded_entry != CANONICAL_ENTRY_DELAY_MONTHS:
    raise RuntimeError("The loaded source config entry delay changed from the locked export.")
if tuple(
    int(value)
    for value in MAIN_CONFIG.get("config", {}).get("EVAL_HORIZONS_MONTHS", ())
) != CFG.report_horizons_months:
    raise RuntimeError("The loaded main config horizons differ from the comparator horizons.")
if INPUT_MANIFEST.get("indices_sampled") is not False:
    raise RuntimeError(
        "The main manifest must explicitly record indices_sampled=false."
    )
if INPUT_MANIFEST.get("indices_shifted_in_model") is not False:
    raise RuntimeError(
        "The main manifest must explicitly record indices_shifted_in_model=false."
    )
CANONICAL_INDEX_PATH_KEY, CANONICAL_INDEX_HASH_KEY = manifest_index_binding(
    INPUT_MANIFEST
)

# These are the artifacts the comparator actually consumes. Their hashes must
# be present and their bytes must match. Upstream CSV/index files are not opened.
REQUIRED_CONSUMED_HASHES = {
    "vocabulary_and_scalers_sha256",
    "canonical_train_samples_sha256",
    "canonical_val_samples_sha256",
    "canonical_test_samples_sha256",
}
missing_consumed_hashes = sorted(
    key for key in REQUIRED_CONSUMED_HASHES if not EXPECTED_HASHES.get(key)
)
if missing_consumed_hashes:
    raise RuntimeError(
        "The main manifest is missing hashes for consumed canonical artifacts: "
        + ", ".join(missing_consumed_hashes)
    )

CONDITION_NAMES = [str(x) for x in TARGET_METADATA["condition_names"]]
K = int(TARGET_METADATA["K"])
if K != len(CONDITION_NAMES):
    raise ValueError("Canonical condition metadata has inconsistent K and condition_names.")

HAZARD_METADATA = TARGET_METADATA["hazard_binning"]
BIN_WIDTH_MONTHS = int(HAZARD_METADATA["BIN_WIDTH_MONTHS"])
MAX_HORIZON_MONTHS = int(HAZARD_METADATA["MAX_HORIZON_MONTHS"])
N_BINS = int(HAZARD_METADATA["N_BINS"])
if N_BINS * BIN_WIDTH_MONTHS != MAX_HORIZON_MONTHS:
    raise ValueError("Canonical hazard axis is not a complete fixed-width grid.")
if CFG.gap_months + max(CFG.report_horizons_months) > MAX_HORIZON_MONTHS:
    raise ValueError("Lead-time gap plus reporting horizon exceeds the canonical axis.")
if any(h <= 0 or h % BIN_WIDTH_MONTHS for h in CFG.report_horizons_months):
    raise ValueError("All report horizons must be positive canonical bin boundaries.")
if CFG.integration_start_months % BIN_WIDTH_MONTHS:
    raise ValueError("Integration start must lie on a canonical bin boundary.")

DENSE_HORIZONS_MONTHS = tuple(
    range(
        CFG.integration_start_months,
        max(CFG.report_horizons_months) + 1,
        BIN_WIDTH_MONTHS,
    )
)

EVENT_META = VOCAB_METADATA["event_vocab"]
TYPE_META = VOCAB_METADATA["event_type_vocab"]
STATIC_META = VOCAB_METADATA["static_categories"]
EVENT_VOCAB_SIZE = int(EVENT_META["EV_VOCAB_SIZE_MODEL"])
EVENT_PAD_ID = int(EVENT_META["EV_PAD_ID"])
EVENT_UNK_ID = int(EVENT_META["EV_UNK_ID"])
EVENT_CLS_ID = int(EVENT_META["EV_CLS_ID"])
TYPE_VOCAB_SIZE = int(TYPE_META["TYPE_VOCAB_SIZE_MODEL"])
TYPE_PAD_ID = int(TYPE_META["TYPE_PAD_ID"])
TYPE_UNK_ID = int(TYPE_META["TYPE_UNK_ID"])
TYPE_CLS_ID = int(TYPE_META["TYPE_CLS_ID"])
N_GENDER = int(STATIC_META["N_GENDER_MODEL"])
N_ETHNICITY = int(STATIC_META["N_ETHNICITY_MODEL"])
N_TOWNSEND = int(STATIC_META["N_TOWNSEND_MODEL"])
UNKNOWN_GENDER = int(STATIC_META["UNKNOWN_GENDER_IDX"])
UNKNOWN_ETHNICITY = int(STATIC_META["UNKNOWN_ETHNICITY_IDX"])
UNKNOWN_TOWNSEND = int(STATIC_META["UNKNOWN_TOWNSEND_Q"])

CR_LABELS = TARGET_METADATA["death_adjusted_design"]["cr_target_labels"]
CR_TARGET_CENSOR = int(
    next(k for k, v in CR_LABELS.items() if "censor" in v or "neither" in v)
)
CR_TARGET_CONDITION = int(
    next(k for k, v in CR_LABELS.items() if "condition" in v)
)
CR_TARGET_DEATH = int(next(k for k, v in CR_LABELS.items() if "death" in v))

MAIN_CLINICAL_HISTORY_MAX_EVENTS = int(
    DATALOADER_METADATA.get("full_sequence_max_events", -1)
)
if MAIN_CLINICAL_HISTORY_MAX_EVENTS < 1:
    raise RuntimeError(
        "The A10 dataloader metadata has no valid full_sequence_max_events contract."
    )
if DATALOADER_METADATA.get("artificial_landmark_used_as_clinical_event") is True:
    raise RuntimeError(
        "The A10 dataloader metadata says the artificial landmark was used clinically."
    )

# The raw-CRP flag is optional in older manifests. The comparator contract is
# instead enforced from canonical values/masks and the vocabulary/scaler file:
# consume the prepared values exactly once, with no comparator-side transform.
MAIN_MANIFEST_RAW_CRP_NO_TRANSFORMATION = INPUT_MANIFEST.get(
    "raw_crp_no_transformation"
)

# Preserve upstream provenance exactly as strings. Relative paths are not
# resolved against this notebook's working directory, and raw files are never
# opened or hashed because they are not model inputs here.
manifest_paths = INPUT_MANIFEST.get("paths", {})
if not isinstance(manifest_paths, dict):
    raise RuntimeError("The main manifest paths field must be an object.")
PROVENANCE_HASH_BINDINGS = {
    "prepared_model_table": "prepared_model_table_sha256",
    "patient_split": "patient_split_sha256",
    "cohort_lock": "cohort_lock_sha256",
    "preparation_metadata": "preparation_metadata_sha256",
}
if CANONICAL_INDEX_PATH_KEY and CANONICAL_INDEX_HASH_KEY:
    PROVENANCE_HASH_BINDINGS[CANONICAL_INDEX_PATH_KEY] = CANONICAL_INDEX_HASH_KEY
MAIN_SOURCE_CONTRACT: Dict[str, Dict[str, Any]] = {}
for source_name, hash_key in PROVENANCE_HASH_BINDINGS.items():
    declared_path = manifest_paths.get(source_name)
    expected_hash = EXPECTED_HASHES.get(hash_key)
    if declared_path is None and expected_hash is None:
        continue
    MAIN_SOURCE_CONTRACT[source_name] = {
        "manifest_path": declared_path,
        "expected_sha256": expected_hash,
        "observed_sha256": None,
        "opened_by_comparator": False,
        "status": "provenance_only_not_reopened",
    }

consumed_artifacts = (
    (
        "vocabulary/scalers",
        CANONICAL_PATHS["vocab_scalers"],
        "vocabulary_and_scalers_sha256",
    ),
    ("train samples", CANONICAL_PATHS["train"], "canonical_train_samples_sha256"),
    ("validation samples", CANONICAL_PATHS["val"], "canonical_val_samples_sha256"),
)
for label, path, hash_key in consumed_artifacts:
    expected = str(EXPECTED_HASHES[hash_key])
    observed = sha256_file(path)
    if expected != observed:
        raise RuntimeError(
            f"Canonical {label} hash differs from the main-model manifest: {path}."
        )

TRAIN_SAMPLES = pd.read_pickle(CANONICAL_PATHS["train"]).copy()
VAL_SAMPLES = pd.read_pickle(CANONICAL_PATHS["val"]).copy()
TRAIN_SAMPLES["split"] = "train"
VAL_SAMPLES["split"] = "val"

print({
    "resolved_canonical_main_run": str(MAIN_RUN_DIR),
    "canonical_index_design": CANONICAL_INDEX_DESIGN,
    "main_config_index_design": MAIN_CONFIG_INDEX_DESIGN,
    "manifest_legacy_index_design": MANIFEST_INDEX_DESIGN,
    "canonical_index_path_key": CANONICAL_INDEX_PATH_KEY,
    "canonical_index_hash_key": CANONICAL_INDEX_HASH_KEY,
    "canonical_train_rows": len(TRAIN_SAMPLES),
    "canonical_val_rows": len(VAL_SAMPLES),
    "canonical_test_loaded": False,
    "raw_sources_opened": False,
    "main_manifest_raw_crp_no_transformation": (
        MAIN_MANIFEST_RAW_CRP_NO_TRANSFORMATION
    ),
    "comparator_crp_contract": (
        "canonical prepared value/mask; no second transformation; "
        "absence is no CRP_MEASURED event"
    ),
    "conditions": K,
    "bin_width_months": BIN_WIDTH_MONTHS,
    "max_horizon_months": MAX_HORIZON_MONTHS,
    "n_bins": N_BINS,
    "clinical_history_max_events": MAIN_CLINICAL_HISTORY_MAX_EVENTS,
    "gap_months": CFG.gap_months,
    "report_horizons_months": CFG.report_horizons_months,
})

# --- Notebook cell 10 ---
REQUIRED_CANONICAL_COLUMNS = {
    "eid", "index_age_years", "index_age_months",
    "history_event_ids", "history_event_type_ids", "history_event_times_months",
    "history_event_values", "history_event_value_masks",
    "gender_idx", "ethnicity_idx", "townsend_q", "continuous_static",
    "at_risk", "event_obs", "event_bin", "t_tilde", "C_months", "C_bin",
    "death_observed_post_index", "death_time_months", "death_bin",
    "cr_at_risk", "cr_event_type", "cr_event_time_months",
}

AGE_TOL_YEARS = float(TARGET_CONFIG.get("age_tolerance_years", 1.0 / 365.25))
TIME_EPS_MONTHS = 1e-6
HISTORY_TIME_TOL_MONTHS = AGE_TOL_YEARS * 12.0 + TIME_EPS_MONTHS
# cr_event_time_months is persisted by the canonical main notebook as
# float32, whereas death_time_months/t_tilde can retain higher precision.
# Compare duplicated target times at a tolerance tied to that storage format.
TARGET_TIME_STORAGE_TOL_MONTHS = max(
    TIME_EPS_MONTHS,
    4.0 * float(np.spacing(np.float32(MAX_HORIZON_MONTHS))),
)
if not np.isfinite(AGE_TOL_YEARS) or AGE_TOL_YEARS < 0:
    raise ValueError(f"Invalid canonical age tolerance: {AGE_TOL_YEARS}")


def _array_length(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return -1


def _validate_history_axis(values: Any, index_months: float, split: str, row_index: Any, eid: str, label: str) -> None:
    times = np.asarray(values, dtype=float)
    if times.size == 0:
        return
    if not np.all(np.isfinite(times)):
        raise ValueError(f"{split} row {row_index} eid={eid}: non-finite {label} history time.")
    maximum = float(times.max())
    if maximum > index_months + HISTORY_TIME_TOL_MONTHS:
        excess_days = (maximum - index_months) * 365.25 / 12.0
        raise ValueError(
            f"{split} row {row_index} eid={eid}: genuine post-index {label}; "
            f"excess={excess_days:.6f} days, allowed={AGE_TOL_YEARS * 365.25:.6f} days."
        )


def validate_canonical_samples(frame: pd.DataFrame, split: str) -> None:
    missing = sorted(REQUIRED_CANONICAL_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{split}: canonical sample columns missing: {missing}")
    if frame.empty or frame["eid"].astype(str).duplicated().any():
        raise ValueError(f"{split}: expected exactly one row per participant.")

    for column in (
        "at_risk", "event_obs", "event_bin", "t_tilde",
        "cr_at_risk", "cr_event_type", "cr_event_bin", "cr_event_time_months",
    ):
        if column not in frame.columns:
            continue
        bad = frame[column].map(_array_length).ne(K)
        if bad.any():
            raise ValueError(f"{split}: {column} has {int(bad.sum())} rows not length K={K}.")

    event_columns = (
        "history_event_ids", "history_event_type_ids", "history_event_times_months",
        "history_event_values", "history_event_value_masks",
    )
    for row_index, row in frame.iterrows():
        eid = str(row["eid"])
        lengths = [_array_length(row[column]) for column in event_columns]
        if min(lengths) < 1 or len(set(lengths)) != 1:
            raise ValueError(f"{split} row {row_index} eid={eid}: misaligned event arrays {lengths}.")
        if "history_diag_ids" in frame.columns and "history_diag_times_months" in frame.columns:
            diag_lengths = [_array_length(row["history_diag_ids"]), _array_length(row["history_diag_times_months"])]
            if min(diag_lengths) < 0 or len(set(diag_lengths)) != 1:
                raise ValueError(f"{split} row {row_index} eid={eid}: misaligned diagnosis arrays.")

        index_months = float(row["index_age_months"])
        index_years = float(row["index_age_years"])
        if not np.isfinite(index_months) or not np.isfinite(index_years):
            raise ValueError(f"{split} row {row_index} eid={eid}: non-finite index age.")
        if not np.isclose(index_months, index_years * 12.0, rtol=0.0, atol=1e-4):
            raise ValueError(f"{split} row {row_index} eid={eid}: inconsistent index age units.")

        _validate_history_axis(row["history_event_times_months"], index_months, split, row_index, eid, "event")
        if "history_diag_times_months" in frame.columns:
            _validate_history_axis(row["history_diag_times_months"], index_months, split, row_index, eid, "diagnosis")

        event_ids = np.asarray(row["history_event_ids"], dtype=int)
        type_ids = np.asarray(row["history_event_type_ids"], dtype=int)
        if np.any((event_ids < 0) | (event_ids >= EVENT_VOCAB_SIZE)):
            raise ValueError(f"{split} row {row_index} eid={eid}: event ID outside canonical vocabulary.")
        if np.any((type_ids < 0) | (type_ids >= TYPE_VOCAB_SIZE)):
            raise ValueError(f"{split} row {row_index} eid={eid}: event-type ID outside canonical vocabulary.")

        cr_types = set(np.unique(np.asarray(row["cr_event_type"], dtype=int)).tolist())
        if not cr_types.issubset({CR_TARGET_CENSOR, CR_TARGET_CONDITION, CR_TARGET_DEATH}):
            raise ValueError(f"{split} row {row_index} eid={eid}: invalid competing-risk labels.")
        at_risk = np.asarray(row["at_risk"], dtype=float) > 0.5
        cr_at_risk = np.asarray(row["cr_at_risk"], dtype=float) > 0.5
        if not np.array_equal(at_risk, cr_at_risk):
            raise ValueError(
                f"{split} row {row_index} eid={eid}: legacy and competing-risk "
                "at-risk axes differ."
            )
        cr_type = np.asarray(row["cr_event_type"], dtype=int)
        cr_time = np.asarray(row["cr_event_time_months"], dtype=float)
        condition_time = np.asarray(row["t_tilde"], dtype=float)
        condition_observed = np.asarray(row["event_obs"], dtype=float) > 0.5
        condition_rows = cr_at_risk & (cr_type == CR_TARGET_CONDITION)
        if np.any(
            condition_rows
            & (
                ~condition_observed
                | ~np.isclose(
                    cr_time,
                    condition_time,
                    rtol=0.0,
                    atol=TARGET_TIME_STORAGE_TOL_MONTHS,
                    equal_nan=False,
                )
            )
        ):
            raise ValueError(
                f"{split} row {row_index} eid={eid}: condition target fields "
                "are not schema-equivalent."
            )
        death_rows = cr_at_risk & (cr_type == CR_TARGET_DEATH)
        if np.any(death_rows):
            death_observed = float(row["death_observed_post_index"]) > 0.5
            death_time = float(row["death_time_months"])
            if (
                not death_observed
                or not np.all(np.isclose(
                    cr_time[death_rows],
                    death_time,
                    rtol=0.0,
                    atol=TARGET_TIME_STORAGE_TOL_MONTHS,
                ))
            ):
                maximum_difference = (
                    float(np.max(np.abs(cr_time[death_rows] - death_time)))
                    if np.any(death_rows)
                    else np.nan
                )
                raise ValueError(
                    f"{split} row {row_index} eid={eid}: death target fields "
                    "are not schema-equivalent; "
                    f"death_observed={death_observed}, "
                    f"death_time_months={death_time}, "
                    f"maximum_cr_time_difference={maximum_difference}, "
                    f"allowed_storage_tolerance="
                    f"{TARGET_TIME_STORAGE_TOL_MONTHS}."
                )


def compare_with_main_index_audit(frame: pd.DataFrame, split: str) -> None:
    audit_path = CANONICAL_PATHS.get(f"{split}_index_audit")
    if audit_path is None or not Path(audit_path).is_file():
        return
    audit = pd.read_csv(audit_path, dtype={"eid": str})
    required = {"eid", "index_dt", "index_age_years", "index_offset_days_from_recruitment"}
    if not required.issubset(audit.columns) or not required.issubset(frame.columns):
        return
    left = frame[["eid", "index_dt", "index_age_years", "index_offset_days_from_recruitment"]].copy()
    left["eid"] = left["eid"].astype(str)
    right = audit[list(required)].copy()
    merged = left.merge(right, on="eid", how="outer", suffixes=("_sample", "_audit"), indicator=True, validate="one_to_one")
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{split}: canonical samples and main-model index audit have different participants.")
    date_bad = pd.to_datetime(merged["index_dt_sample"]) != pd.to_datetime(merged["index_dt_audit"])
    age_bad = ~np.isclose(
        pd.to_numeric(merged["index_age_years_sample"]).to_numpy(float),
        pd.to_numeric(merged["index_age_years_audit"]).to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    )
    offset_bad = pd.to_numeric(merged["index_offset_days_from_recruitment_sample"]).to_numpy(float) != pd.to_numeric(merged["index_offset_days_from_recruitment_audit"]).to_numpy(float)
    if bool(np.any(date_bad | age_bad | offset_bad)):
        raise RuntimeError(f"{split}: canonical sample index fields differ from the main-model index audit.")


validate_canonical_samples(TRAIN_SAMPLES, "train")
validate_canonical_samples(VAL_SAMPLES, "val")
compare_with_main_index_audit(TRAIN_SAMPLES, "train")
compare_with_main_index_audit(VAL_SAMPLES, "val")
if set(TRAIN_SAMPLES["eid"].astype(str)) & set(VAL_SAMPLES["eid"].astype(str)):
    raise RuntimeError("Participant leakage between canonical train and validation splits.")

INDEX_AUDIT_COLUMNS = [
    col for col in [
        "eid", "split", "index_source", "index_dt", "index_age_years",
        "recruitment_dt", "age_at_recruitment", "base_random_index_age_years",
        "base_random_offset_days", "reduced_offset_days_unclipped",
        "index_offset_days_from_recruitment",
    ]
    if col in TRAIN_SAMPLES.columns and col in VAL_SAMPLES.columns
]
pd.concat([
    TRAIN_SAMPLES[INDEX_AUDIT_COLUMNS],
    VAL_SAMPLES[INDEX_AUDIT_COLUMNS],
], ignore_index=True).to_csv(RUN_DIR / "canonical_index_audit_train_val.csv", index=False)

N_CONTINUOUS_STATIC = int(np.asarray(TRAIN_SAMPLES.iloc[0]["continuous_static"]).size)
if N_CONTINUOUS_STATIC < 1:
    raise ValueError("Canonical continuous_static vectors are empty.")
for split, frame in (("train", TRAIN_SAMPLES), ("val", VAL_SAMPLES)):
    bad = frame["continuous_static"].map(lambda x: np.asarray(x).size != N_CONTINUOUS_STATIC)
    if bad.any():
        raise ValueError(f"{split}: inconsistent continuous_static vector length.")

print({
    "canonical_contract_validated": True,
    "age_tolerance_days": AGE_TOL_YEARS * 365.25,
    "train_index_audit_checked": CANONICAL_PATHS["train_index_audit"].is_file(),
    "val_index_audit_checked": CANONICAL_PATHS["val_index_audit"].is_file(),
})

# --- Notebook cell 12 ---
DAYS_PER_YEAR = 365.25
DAYS_PER_MONTH = DAYS_PER_YEAR / 12.0

# Delphi uses one autoregressive vocabulary. Canonical event-stream IDs are
# preserved through a fixed +2 offset; input-only and terminal tokens follow.
DELPHI_PAD_ID = 0
DELPHI_NO_EVENT_ID = 1
DELPHI_EVENT_OFFSET = 2
DELPHI_STATIC_ID = DELPHI_EVENT_OFFSET + EVENT_VOCAB_SIZE
DELPHI_DEATH_ID = DELPHI_STATIC_ID + 1
DELPHI_VOCAB_SIZE = DELPHI_DEATH_ID + 1

EVENT_TOKEN_TO_SOURCE_ID = {
    str(token): int(identifier)
    for token, identifier in EVENT_META["EV_TOKEN_TO_ID"].items()
}
EVENT_SOURCE_ID_TO_TOKEN = {
    int(identifier): str(token)
    for identifier, token in EVENT_META["EV_ID_TO_TOKEN"].items()
}
TYPE_TO_SOURCE_ID = {
    str(token).strip().lower(): int(identifier)
    for token, identifier in TYPE_META["TYPE_TO_ID"].items()
}
DX_TYPE_SOURCE_ID = TYPE_TO_SOURCE_ID.get("dx")
if DX_TYPE_SOURCE_ID is None:
    raise ValueError("Canonical event-type vocabulary does not contain 'dx'.")

if "EV_NEAR_RECRUITMENT_INDEX_ID" in EVENT_META:
    CANONICAL_LANDMARK_SOURCE_ID = int(EVENT_META["EV_NEAR_RECRUITMENT_INDEX_ID"])
    CANONICAL_LANDMARK_TOKEN = str(
        EVENT_SOURCE_ID_TO_TOKEN.get(CANONICAL_LANDMARK_SOURCE_ID, "EV_NEAR_RECRUITMENT_INDEX_ID")
    )
else:
    LANDMARK_TOKEN_CANDIDATES = (
        "HYBRID_INDEX",
        "FIXED_SIX_MONTH_INDEX",
        "LATEST_COVARIATE_INDEX",
        "INDEX_LANDMARK",
    )
    available_landmark_tokens = [
        token for token in LANDMARK_TOKEN_CANDIDATES
        if token in EVENT_TOKEN_TO_SOURCE_ID
    ]
    if len(available_landmark_tokens) != 1:
        raise ValueError(
            "Expected exactly one canonical compatibility landmark token; found "
            f"{available_landmark_tokens}."
        )
    CANONICAL_LANDMARK_TOKEN = available_landmark_tokens[0]
    CANONICAL_LANDMARK_SOURCE_ID = int(
        EVENT_TOKEN_TO_SOURCE_ID[CANONICAL_LANDMARK_TOKEN]
    )
CRP_TYPE_SOURCE_ID = TYPE_TO_SOURCE_ID.get("crp")
CRP_SOURCE_IDS = {
    int(v)
    for k, v in EVENT_TOKEN_TO_SOURCE_ID.items()
    if k.upper() == "CRP_MEASURED" or k.upper().startswith("CRP_D")
}
CRP_MEASURED_SOURCE_ID = EVENT_TOKEN_TO_SOURCE_ID.get("CRP_MEASURED")


def _condition_source_event_id(condition_name: str) -> int:
    raw = str(condition_name)
    candidates = [raw]
    if raw.startswith("DX:"):
        candidates.append(raw[3:])
    else:
        candidates.append(f"DX:{raw}")
    for candidate in candidates:
        if candidate in EVENT_TOKEN_TO_SOURCE_ID:
            return int(EVENT_TOKEN_TO_SOURCE_ID[candidate])
    raise KeyError(
        "Condition is absent from the canonical event vocabulary: "
        f"{condition_name!r}; attempted {candidates}."
    )


CONDITION_SOURCE_EVENT_IDS = np.asarray(
    [_condition_source_event_id(name) for name in CONDITION_NAMES],
    dtype=np.int64,
)
if len(np.unique(CONDITION_SOURCE_EVENT_IDS)) != K:
    raise ValueError("Condition-to-event mapping is not one-to-one.")
DELPHI_CONDITION_IDS = CONDITION_SOURCE_EVENT_IDS + DELPHI_EVENT_OFFSET

PREDICTABLE_SOURCE_EVENT_IDS: set[int] = set()
_has_preindex_dx_repeats = False
for event_ids, type_ids in zip(
    TRAIN_SAMPLES["history_event_ids"],
    TRAIN_SAMPLES["history_event_type_ids"],
):
    seen_dx: set[int] = set()
    for event_id, type_id in zip(event_ids, type_ids):
        if int(type_id) == int(DX_TYPE_SOURCE_ID):
            eid_int = int(event_id)
            if eid_int in seen_dx:
                _has_preindex_dx_repeats = True
            seen_dx.add(eid_int)
            PREDICTABLE_SOURCE_EVENT_IDS.add(eid_int)
PREDICTABLE_SOURCE_EVENT_IDS.update(
    int(value) for value in CONDITION_SOURCE_EVENT_IDS
)

ALLOWED_OUTPUT_MASK = np.zeros(DELPHI_VOCAB_SIZE, dtype=bool)
ALLOWED_OUTPUT_MASK[DELPHI_NO_EVENT_ID] = True
ALLOWED_OUTPUT_MASK[DELPHI_DEATH_ID] = True
for source_event_id in PREDICTABLE_SOURCE_EVENT_IDS:
    if 0 <= source_event_id < EVENT_VOCAB_SIZE:
        ALLOWED_OUTPUT_MASK[
            DELPHI_EVENT_OFFSET + source_event_id
        ] = True
ALLOWED_OUTPUT_MASK[DELPHI_PAD_ID] = False
ALLOWED_OUTPUT_MASK[DELPHI_STATIC_ID] = False

POST_INDEX_ALLOWED_OUTPUT_MASK = np.zeros(DELPHI_VOCAB_SIZE, dtype=bool)
POST_INDEX_ALLOWED_OUTPUT_MASK[DELPHI_NO_EVENT_ID] = True
POST_INDEX_ALLOWED_OUTPUT_MASK[DELPHI_CONDITION_IDS] = True
POST_INDEX_ALLOWED_OUTPUT_MASK[DELPHI_DEATH_ID] = True

POST_INDEX_TARGET_OUTPUT_MASK = POST_INDEX_ALLOWED_OUTPUT_MASK.copy()
POST_INDEX_TARGET_OUTPUT_MASK[DELPHI_NO_EVENT_ID] = False

# Tokens governed by first-occurrence survival semantics (cannot recur once observed).
FIRST_OCCURRENCE_OUTPUT_MASK = np.zeros(DELPHI_VOCAB_SIZE, dtype=bool)
if not _has_preindex_dx_repeats:
    FIRST_OCCURRENCE_OUTPUT_MASK = ALLOWED_OUTPUT_MASK.copy()
    FIRST_OCCURRENCE_OUTPUT_MASK[DELPHI_NO_EVENT_ID] = False
else:
    FIRST_OCCURRENCE_OUTPUT_MASK[DELPHI_CONDITION_IDS] = True
    FIRST_OCCURRENCE_OUTPUT_MASK[DELPHI_DEATH_ID] = True


def _canonical_real_events(
    row: pd.Series,
    include_future: bool,
) -> List[Tuple[float, int, int, float, float]]:
    """Return A10-matched pre-index inputs plus optional incident outcomes."""
    pre_index: List[Tuple[float, int, int, int, float, float]] = []
    index_age = float(row["index_age_months"])
    min_event_age_months = CFG.query_epsilon_days / DAYS_PER_MONTH
    event_axes = zip(
        row["history_event_ids"],
        row["history_event_type_ids"],
        row["history_event_times_months"],
        row["history_event_values"],
        row["history_event_value_masks"],
    )
    for order, (source_event, source_type, age_months, event_value, value_mask) in enumerate(event_axes):
        source_event = int(source_event)
        source_type = int(source_type)
        age_months = float(age_months)
        genuine = source_event not in {
            int(CANONICAL_LANDMARK_SOURCE_ID),
            int(EVENT_CLS_ID),
            int(EVENT_PAD_ID),
        }
        if not genuine:
            continue
        if not (0 <= source_event < EVENT_VOCAB_SIZE) or not np.isfinite(age_months):
            continue
        if age_months > index_age + HISTORY_TIME_TOL_MONTHS:
            raise RuntimeError(
                f"Post-index predictor in canonical history for eid={row['eid']}: "
                f"event_age={age_months}, index_age={index_age}."
            )
        # Keep strictly > 0.0 so DELPHI_STATIC_ID at 0.0 is never masked by mask_ties.
        age_months = max(min(age_months, index_age), min_event_age_months)
        prepared_value = float(
            np.nan_to_num(event_value, nan=0.0, posinf=0.0, neginf=0.0)
        )
        prepared_mask = float(
            np.isfinite(value_mask) and float(value_mask) >= 0.5
        )
        is_crp_event = (
            (CRP_TYPE_SOURCE_ID is not None and source_type == int(CRP_TYPE_SOURCE_ID))
            or (source_event in CRP_SOURCE_IDS)
        )
        if prepared_mask > 0.5 and not is_crp_event:
            raise RuntimeError(
                f"Unexpected numeric event outside CRP for eid={row['eid']}."
            )
        pre_index.append(
            (
                age_months,
                source_event,
                source_type,
                order,
                prepared_value,
                prepared_mask,
            )
        )

    pre_index.sort(key=lambda item: (item[0], item[1], item[2]))
    if len(pre_index) > MAIN_CLINICAL_HISTORY_MAX_EVENTS:
        recent = list(pre_index[-MAIN_CLINICAL_HISTORY_MAX_EVENTS:])
        crp_items = [
            item for item in pre_index
            if (CRP_TYPE_SOURCE_ID is not None and item[2] == int(CRP_TYPE_SOURCE_ID))
            or (item[1] in CRP_SOURCE_IDS)
        ]
        if crp_items and crp_items[-1] not in recent:
            recent[0] = crp_items[-1]
            recent.sort(key=lambda item: (item[0], item[1], item[2]))
        pre_index = recent
    events: List[Tuple[float, int, int, float, float]] = [
        (
            age_months,
            DELPHI_EVENT_OFFSET + source_event,
            rank,
            prepared_value,
            prepared_mask,
        )
        for rank, (
            age_months,
            source_event,
            _source_type,
            _source_order,
            prepared_value,
            prepared_mask,
        ) in enumerate(pre_index)
    ]

    if not include_future:
        return sorted(events, key=lambda item: (item[0], item[2]))

    followup = min(
        max(float(row["C_months"]), 0.0),
        float(MAX_HORIZON_MONTHS),
    )
    cr_at_risk = np.asarray(row["cr_at_risk"], dtype=float) > 0.5
    cr_type = np.asarray(row["cr_event_type"], dtype=int)
    cr_time = np.asarray(row["cr_event_time_months"], dtype=float)

    future_order = len(events) + 10
    for condition in range(K):
        if (
            cr_at_risk[condition]
            and cr_type[condition] == CR_TARGET_CONDITION
            and np.isfinite(cr_time[condition])
            and 0.0 < cr_time[condition] <= followup + TIME_EPS_MONTHS
        ):
            events.append(
                (
                    index_age + float(cr_time[condition]),
                    int(DELPHI_CONDITION_IDS[condition]),
                    future_order + condition,
                    0.0,
                    0.0,
                )
            )

    death_observed = float(row["death_observed_post_index"]) > 0.5
    death_time = float(row["death_time_months"])
    if (
        death_observed
        and np.isfinite(death_time)
        and 0.0 < death_time <= followup + TIME_EPS_MONTHS
    ):
        events.append((
            index_age + death_time,
            DELPHI_DEATH_ID,
            future_order + K,
            0.0,
            0.0,
        ))
    return sorted(events, key=lambda item: (item[0], item[2]))


def _observation_start_age_months(
    row: pd.Series,
    real_events: Sequence[Tuple[float, int, int, float, float]],
) -> float:
    """Consistent adult observation origin (default 35y), or strictly before earliest event."""
    origin_months = float(CFG.origin_age_years) * 12.0
    eps_months = float(CFG.query_epsilon_days) / DAYS_PER_MONTH
    index_age = float(row["index_age_months"])
    upper = index_age - eps_months
    if real_events:
        upper = min(upper, float(real_events[0][0]) - eps_months)
    return float(max(0.0, min(origin_months, upper)))


def _no_event_ages_months(
    start_age: float,
    end_age: float,
    padding_mode: str,
) -> List[float]:
    """Return paper-style lifetime NO_EVENT probes intersected with observation."""
    if not np.isfinite(start_age) or not np.isfinite(end_age):
        return []
    if float(end_age) - float(start_age) <= TIME_EPS_MONTHS:
        return []
    if padding_mode == "none":
        return []

    lifetime_months = 100.0 * 12.0
    epsilon_months = CFG.query_epsilon_days / DAYS_PER_MONTH
    if padding_mode == "random":
        candidate_count = max(
            1,
            int(round(lifetime_months / CFG.no_event_interval_months)),
        )
        candidates = np.random.uniform(
            epsilon_months,
            lifetime_months,
            size=candidate_count,
        )
    elif padding_mode == "regular":
        candidates = np.arange(
            epsilon_months,
            lifetime_months,
            CFG.no_event_interval_months,
            dtype=float,
        )
    else:
        raise ValueError(f"Unsupported no-event padding mode: {padding_mode!r}.")

    return sorted(
        float(value)
        for value in candidates
        if float(start_age) < float(value) < float(end_age)
    )


def _observation_end_age_months(row: pd.Series) -> float:
    """Return the locked follow-up boundary, truncated at observed death."""
    index_age = float(row["index_age_months"])
    followup = min(
        max(float(row["C_months"]), 0.0),
        float(MAX_HORIZON_MONTHS),
    )
    observation_end = index_age + followup
    death_observed = float(row["death_observed_post_index"]) > 0.5
    death_time = float(row["death_time_months"])
    if (
        death_observed
        and np.isfinite(death_time)
        and 0.0 < death_time <= followup + TIME_EPS_MONTHS
    ):
        observation_end = min(observation_end, index_age + death_time)
    return float(observation_end)


def build_training_sequence(
    row: pd.Series,
    padding_mode: str = "regular",
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Build a birth-anchored AR sequence with DELPHI_STATIC_ID at age 0.0."""
    observation_end = _observation_end_age_months(row)
    real_events = [
        event
        for event in _canonical_real_events(row, include_future=True)
        if float(event[0]) <= observation_end + TIME_EPS_MONTHS
    ]
    start_age = _observation_start_age_months(row, real_events)
    index_age = float(row["index_age_months"])

    serialized: List[Tuple[float, int, int, float, float]] = [
        (start_age, DELPHI_STATIC_ID, -20, 0.0, 0.0)
    ]
    serialized.extend(real_events)
    serialized.extend(
        (age, DELPHI_NO_EVENT_ID, -10, 0.0, 0.0)
        for age in _no_event_ages_months(
            start_age,
            observation_end,
            padding_mode,
        )
    )

    final_real_age = (
        float(real_events[-1][0]) if real_events else float(index_age)
    )
    if observation_end > final_real_age + TIME_EPS_MONTHS:
        serialized.append(
            (observation_end, DELPHI_NO_EVENT_ID, 10**9, 0.0, 0.0)
        )
    serialized.sort(key=lambda item: (item[0], item[2]))

    token_ids = [int(item[1]) for item in serialized]
    ages_days = [float(item[0]) * DAYS_PER_MONTH for item in serialized]
    event_values = [float(item[3]) for item in serialized]
    event_value_masks = [float(item[4]) for item in serialized]
    if len(token_ids) < 2:
        raise RuntimeError(
            f"Delphi sequence has fewer than two tokens for eid={row['eid']}."
        )
    return token_ids, ages_days, event_values, event_value_masks


def build_prediction_sequence(
    row: pd.Series,
    query_offset_months: float,
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Build a birth-anchored history-only sequence plus one rate query."""
    real_events = _canonical_real_events(row, include_future=False)
    start_age = _observation_start_age_months(row, real_events)
    index_age = float(row["index_age_months"])
    query_age_months = index_age + float(query_offset_months)

    serialized: List[Tuple[float, int, int, float, float]] = [
        (start_age, DELPHI_STATIC_ID, -20, 0.0, 0.0)
    ]
    serialized.extend(real_events)
    # Stop regular NO_EVENT probes at index_age so future bin queries do not
    # condition on synthetic post-index all-disease-free survival.
    serialized.extend(
        (age, DELPHI_NO_EVENT_ID, -10, 0.0, 0.0)
        for age in _no_event_ages_months(
            start_age,
            index_age,
            "regular",
        )
    )
    serialized.sort(key=lambda item: (item[0], item[2]))

    query_age_days = query_age_months * DAYS_PER_MONTH + CFG.query_epsilon_days
    token_ids = [int(item[1]) for item in serialized] + [DELPHI_NO_EVENT_ID]
    ages_days = [float(item[0]) * DAYS_PER_MONTH for item in serialized] + [
        query_age_days
    ]
    event_values = [float(item[3]) for item in serialized] + [0.0]
    event_value_masks = [float(item[4]) for item in serialized] + [0.0]

    if len(token_ids) > CFG.block_size:
        keep = [0] + list(range(len(token_ids) - (CFG.block_size - 1), len(token_ids)))
        token_ids = [token_ids[j] for j in keep]
        ages_days = [ages_days[j] for j in keep]
        event_values = [event_values[j] for j in keep]
        event_value_masks = [event_value_masks[j] for j in keep]
    return token_ids, ages_days, event_values, event_value_masks


def add_delphi_sequences(
    frame: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """Persist regular-padding sequences for audits; train padding is dynamic."""
    frame = frame.copy()
    sequences = [
        build_training_sequence(row, padding_mode="regular")
        for _, row in frame.iterrows()
    ]
    frame["delphi_token_ids"] = [tokens for tokens, _, _, _ in sequences]
    frame["delphi_age_days"] = [ages for _, ages, _, _ in sequences]
    frame["delphi_event_values"] = [values for _, _, values, _ in sequences]
    frame["delphi_event_value_masks"] = [masks for _, _, _, masks in sequences]
    frame["delphi_padding_mode_for_audit"] = "regular"
    frame["delphi_split"] = str(split)
    return frame


TRAIN_SAMPLES = add_delphi_sequences(TRAIN_SAMPLES, "train")
VAL_SAMPLES = add_delphi_sequences(VAL_SAMPLES, "val")

CANONICAL_SOURCE_HASHES = {
    key: sha256_file(path)
    for key, path in CANONICAL_PATHS.items()
    if (
        not key.startswith("test")
        and key != "main_per_condition_metrics"
        and Path(path).is_file()
    )
}

MODEL_SCIENTIFIC_CONFIG_KEYS = (
    "block_size", "d_model", "n_heads", "n_layers", "d_ff", "dropout",
    "token_dropout", "bias", "use_input_norm", "use_output_bias",
    "mask_ties", "t_min_days", "origin_age_years", "ema_decay",
    "no_event_interval_months", "query_epsilon_days", "paper_rate_exp_logits",
    "post_index_target_output_restriction", "dynamic_first_event_risk_sets",
    "selection_post_index_targets_only", "full_vocabulary_ar_likelihood",
    "landmark_auxiliary_weight", "landmark_brier_weight",
    "validation_risk_subset_n", "validation_risk_horizons_months",
    "validation_macro_brier_weight", "validation_macro_auc_weight",
    "enable_validation_rate_scaling", "rate_scaling_prior_events",
    "rate_scaling_log_limit",
    "max_iterations", "seed", "training_seed",
    "batch_size", "validation_interval", "validation_batches",
    "patience_validations", "min_delta", "max_gradient_norm", "precision",
    "learning_rate", "min_learning_rate", "weight_decay", "beta1", "beta2",
    "warmup_iterations", "gap_months", "report_horizons_months",
    "integration_start_months", "include_right_censor_survival",
    "censor_aware_no_event_padding",
    "selection_exclude_no_event", "selection_include_censor_survival",
)
MODEL_SCIENTIFIC_CONFIG = {
    key: getattr(CFG, key) for key in MODEL_SCIENTIFIC_CONFIG_KEYS
}
MODEL_SCIENTIFIC_CONFIG["resolved_precision"] = PRECISION
MODEL_SCIENTIFIC_CONFIG["deterministic_algorithms"] = True
PREPROCESSING_CONTRACT = {
    "schema_version": CFG.schema_version,
    "model_name": "Delphi2M_Canonical_AR_RateGrid",
    "main_model_run_dir": str(MAIN_RUN_DIR),
    "main_model_input_manifest_sha256": sha256_file(
        CANONICAL_PATHS["input_manifest"]
    ),
    "canonical_index_contract": {
        "index_design": CANONICAL_INDEX_DESIGN,
        "semantic_authority": str(CANONICAL_PATHS["target_config"]),
        "main_config_design": MAIN_CONFIG_INDEX_DESIGN,
        "manifest_legacy_design": MANIFEST_INDEX_DESIGN,
        "maximum_calendar_months_from_recruitment": 1,
        "indices_sampled": False,
        "indices_shifted_in_model": False,
        "manifest_path_key": CANONICAL_INDEX_PATH_KEY,
        "manifest_hash_key": CANONICAL_INDEX_HASH_KEY,
        "landmark_token": CANONICAL_LANDMARK_TOKEN,
    },
    "main_source_contract": MAIN_SOURCE_CONTRACT,
    "canonical_source_hashes": CANONICAL_SOURCE_HASHES,
    "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
    "condition_names": CONDITION_NAMES,
    "K": K,
    "hazard_axis": {
        "bin_width_months": BIN_WIDTH_MONTHS,
        "max_horizon_months": MAX_HORIZON_MONTHS,
        "n_bins": N_BINS,
        "gap_months": CFG.gap_months,
        "report_horizons_months": list(CFG.report_horizons_months),
        "dense_horizons_months": list(DENSE_HORIZONS_MONTHS),
    },
    "delphi_vocabulary": {
        "pad_id": DELPHI_PAD_ID,
        "no_event_id": DELPHI_NO_EVENT_ID,
        "canonical_event_offset": DELPHI_EVENT_OFFSET,
        "canonical_event_vocab_size": EVENT_VOCAB_SIZE,
        "static_input_only_id": DELPHI_STATIC_ID,
        "death_id": DELPHI_DEATH_ID,
        "vocab_size": DELPHI_VOCAB_SIZE,
        "condition_token_ids": DELPHI_CONDITION_IDS.tolist(),
        "input_only_output_masking": True,
        "preindex_dx_repeats_detected": bool(_has_preindex_dx_repeats),
    },
    "sequence_contract": {
        "origin": f"adult_origin_{CFG.origin_age_years:.1f}y_with_static_demographics_excluding_index_age",
        "clinical_history_max_events": MAIN_CLINICAL_HISTORY_MAX_EVENTS,
        "artificial_landmark_removed": True,
        "numeric_event_value_contract": "log1p(prepared_crp_value) * value_mask",
        "exogenous_interval_survival": "disjoint_full_interval_survival_retained",
        "post_index_output_support": "at_risk_canonical_targets_plus_death_and_no_event",
        "dynamic_first_event_risk_sets": bool(CFG.dynamic_first_event_risk_sets),
        "pre_index_death_masked_for_left_truncation": True,
        "prediction_no_event_probes": "pre_index_only_plus_single_target_age_query",
        "test_future_used": False,
        "same_time_masking": bool(CFG.mask_ties),
        "age_unit": "days",
    },
    "static_extension": {
        "gender_categories": N_GENDER,
        "ethnicity_categories": N_ETHNICITY,
        "townsend_categories": N_TOWNSEND,
        "continuous_static_features": N_CONTINUOUS_STATIC,
        "method": (
            "combined learned demographic context injected once at birth (age 0.0), "
            "with standardized index_age_years zeroed to prevent survival-age leakage"
        ),
    },
    "model_scientific_config": MODEL_SCIENTIFIC_CONFIG,
    "test_loaded": False,
}
PREPROCESSING_CONTRACT_HASH = canonical_json_hash(PREPROCESSING_CONTRACT)
PREPROCESSING_CONTRACT_PATH = (
    RUN_DIR / "canonical_comparator_preprocessing_contract.json"
)
if PREPROCESSING_CONTRACT_PATH.exists():
    with PREPROCESSING_CONTRACT_PATH.open(encoding="utf-8") as handle:
        existing = json.load(handle)
    if canonical_json_hash(existing) != PREPROCESSING_CONTRACT_HASH:
        raise RuntimeError(
            "Output directory contains a different preprocessing contract. "
            "Use a new EHR_DELPHI_OUTPUT."
        )
else:
    atomic_json_dump(PREPROCESSING_CONTRACT, PREPROCESSING_CONTRACT_PATH)

atomic_json_dump(
    {
        "schema_version": CFG.schema_version,
        "semantic_authority": str(CANONICAL_PATHS["target_config"]),
        "canonical_index_design": CANONICAL_INDEX_DESIGN,
        "main_config_index_design": MAIN_CONFIG_INDEX_DESIGN,
        "manifest_legacy_index_design": MANIFEST_INDEX_DESIGN,
        "manifest_path_key_used_for_hash_binding": CANONICAL_INDEX_PATH_KEY,
        "manifest_hash_key_used_for_hash_binding": CANONICAL_INDEX_HASH_KEY,
        "maximum_calendar_months_from_recruitment": 1,
        "gap_months": CFG.gap_months,
        "lead_time_months": CFG.gap_months,
        "source_export_gap_months": CANONICAL_GAP_MONTHS,
        "source_export_lead_time_months": CANONICAL_LEAD_TIME_MONTHS,
        "indices_sampled": False,
        "indices_shifted_in_model": False,
    },
    RUN_DIR / "canonical_index_contract_sources.json",
)

pd.DataFrame({
    "condition_id": np.arange(K, dtype=int),
    "delphi_token_id": DELPHI_CONDITION_IDS,
    "condition_name": CONDITION_NAMES,
}).to_csv(RUN_DIR / "condition_vocab_canonical.csv", index=False)

pd.DataFrame({
    "eid": pd.concat([TRAIN_SAMPLES["eid"], VAL_SAMPLES["eid"]], ignore_index=True).astype(str),
    "split": ["train"] * len(TRAIN_SAMPLES) + ["val"] * len(VAL_SAMPLES),
    "sequence_length": [
        len(value)
        for value in pd.concat(
            [TRAIN_SAMPLES["delphi_token_ids"], VAL_SAMPLES["delphi_token_ids"]],
            ignore_index=True,
        )
    ],
    "audit_padding_mode": "regular",
}).to_csv(RUN_DIR / "delphi_sequence_audit_train_val.csv", index=False)

atomic_json_dump({
    "implementation": (
        "Paper-faithful Delphi-2M original-loss 50k comparator on CRESCENT's "
        "one-month-cap index and six-month delayed-entry evaluation contract (fixed v3)"
    ),
    "defining_delphi_identity": (
        "autoregressive tied-output transformer with the published unit-weight "
        "next-token and competing-exponential waiting-time likelihood"
    ),
    "published_components_preserved": [
        "causal GPT-style transformer",
        "birth-anchored trajectory starting at age 0.0",
        "absolute-age sinusoidal encoding in days",
        "same-time masking",
        "tied token-embedding/output weights",
        "fixed-candidate random-train and regular-validation NO_EVENT probes",
        "joint token and complete-exposure waiting-time likelihood",
        "dynamic first-occurrence diagnosis risk-set masking",
        "paper-defined exp(logit) cause-specific rates",
        "history-only NO_EVENT risk queries",
    ],
    "not_claimed": (
        "This is not a pretrained 1,000-disease Delphi-2M checkpoint reproduction; "
        "the canonical A10 data interface and output schema are comparison adapters."
    ),
}, RUN_DIR / "scientific_fidelity_manifest.json")

print({
    "delphi_sequences_built": True,
    "train_sequences": len(TRAIN_SAMPLES),
    "validation_sequences": len(VAL_SAMPLES),
    "median_train_sequence_length": float(
        np.median(TRAIN_SAMPLES["delphi_token_ids"].map(len))
    ),
    "max_train_sequence_length": int(
        TRAIN_SAMPLES["delphi_token_ids"].map(len).max()
    ),
    "canonical_event_vocab_size": EVENT_VOCAB_SIZE,
    "delphi_vocab_size": DELPHI_VOCAB_SIZE,
    "input_only_event_tokens": int(
        DELPHI_VOCAB_SIZE - int(ALLOWED_OUTPUT_MASK.sum())
    ),
    "first_occurrence_masked_tokens": int(FIRST_OCCURRENCE_OUTPUT_MASK.sum()),
    "terminal_censor_survival": False,
    "test_loaded": False,
})

# --- Notebook cell 14 ---
class DelphiARDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, split: str):
        self.frame = frame.reset_index(drop=True)
        self.split = str(split)
        if self.split not in {"train", "val"}:
            raise ValueError(f"Unsupported Delphi dataset split: {self.split!r}.")
        self.padding_mode = "random" if self.split == "train" else "regular"

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)
        row = self.frame.iloc[index]
        token_ids, age_days, event_values, event_value_masks = build_training_sequence(
            row, padding_mode=self.padding_mode
        )
        if len(token_ids) < 2:
            raise RuntimeError(
                f"Delphi sequence has no autoregressive target: eid={row['eid']}."
            )
        window = CFG.block_size
        if len(token_ids) > window + 1:
            # Slice token trajectory before splitting input/target so transitions stay contiguous
            # while always retaining position 0 (DELPHI_STATIC_ID).
            keep = [0] + list(range(len(token_ids) - window, len(token_ids)))
            token_ids = [token_ids[j] for j in keep]
            age_days = [age_days[j] for j in keep]
            event_values = [event_values[j] for j in keep]
            event_value_masks = [event_value_masks[j] for j in keep]

        input_ids = token_ids[:-1]
        input_ages = age_days[:-1]
        input_event_values = event_values[:-1]
        input_event_value_masks = event_value_masks[:-1]
        target_ids = token_ids[1:]
        target_ages = age_days[1:]
        censor_mask = [False] * len(input_ids)
        post_index_output_mask = POST_INDEX_ALLOWED_OUTPUT_MASK.copy()
        post_index_output_mask[DELPHI_CONDITION_IDS] = (
            np.asarray(row["cr_at_risk"], dtype=float) > 0.5
        )

        return {
            "eid": str(row["eid"]),
            "token_ids": input_ids,
            "target_ids": target_ids,
            "input_age_days": input_ages,
            "event_values": input_event_values,
            "event_value_masks": input_event_value_masks,
            "target_age_days": target_ages,
            "index_age_days": float(row["index_age_months"]) * DAYS_PER_MONTH,
            "post_index_output_mask": post_index_output_mask,
            "censor_mask": censor_mask,
            "gender_idx": int(row["gender_idx"]),
            "ethnicity_idx": int(row["ethnicity_idx"]),
            "townsend_q": int(row["townsend_q"]),
            "continuous_static": np.asarray(
                row["continuous_static"], dtype=np.float32
            ),
        }


def _sanitize_static_indices(
    gender: torch.Tensor,
    ethnicity: torch.Tensor,
    townsend: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gender[(gender < 0) | (gender >= N_GENDER)] = UNKNOWN_GENDER
    ethnicity[(ethnicity < 0) | (ethnicity >= N_ETHNICITY)] = UNKNOWN_ETHNICITY
    townsend[(townsend < 0) | (townsend >= N_TOWNSEND)] = UNKNOWN_TOWNSEND
    return gender, ethnicity, townsend


def collate_delphi_ar(samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not samples:
        raise ValueError("Cannot collate an empty Delphi batch.")
    lengths = torch.as_tensor(
        [len(sample["token_ids"]) for sample in samples], dtype=torch.long
    )
    max_length = int(lengths.max())
    batch_size = len(samples)
    token_ids = torch.full(
        (batch_size, max_length), DELPHI_PAD_ID, dtype=torch.long
    )
    target_ids = torch.full_like(token_ids, -1)
    input_age_days = torch.zeros(
        (batch_size, max_length), dtype=torch.float32
    )
    event_values = torch.zeros_like(input_age_days)
    event_value_masks = torch.zeros_like(input_age_days)
    target_age_days = torch.zeros_like(input_age_days)
    censor_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool
    )
    post_index_output_mask = torch.zeros(
        (batch_size, DELPHI_VOCAB_SIZE), dtype=torch.bool
    )
    attention_mask = torch.zeros(
        (batch_size, max_length), dtype=torch.bool
    )
    for i, sample in enumerate(samples):
        n = len(sample["token_ids"])
        token_ids[i, :n] = torch.as_tensor(sample["token_ids"], dtype=torch.long)
        target_ids[i, :n] = torch.as_tensor(sample["target_ids"], dtype=torch.long)
        input_age_days[i, :n] = torch.as_tensor(
            sample["input_age_days"], dtype=torch.float32
        )
        event_values[i, :n] = torch.as_tensor(
            sample["event_values"], dtype=torch.float32
        )
        event_value_masks[i, :n] = torch.as_tensor(
            sample["event_value_masks"], dtype=torch.float32
        )
        target_age_days[i, :n] = torch.as_tensor(
            sample["target_age_days"], dtype=torch.float32
        )
        censor_mask[i, :n] = torch.as_tensor(
            sample["censor_mask"], dtype=torch.bool
        )
        post_index_output_mask[i] = torch.as_tensor(
            sample["post_index_output_mask"], dtype=torch.bool
        )
        attention_mask[i, :n] = True

    gender, ethnicity, townsend = _sanitize_static_indices(
        torch.as_tensor([sample["gender_idx"] for sample in samples], dtype=torch.long),
        torch.as_tensor([sample["ethnicity_idx"] for sample in samples], dtype=torch.long),
        torch.as_tensor([sample["townsend_q"] for sample in samples], dtype=torch.long),
    )
    return {
        "eid": [sample["eid"] for sample in samples],
        "token_ids": token_ids,
        "target_ids": target_ids,
        "input_age_days": input_age_days,
        "event_values": event_values,
        "event_value_masks": event_value_masks,
        "target_age_days": target_age_days,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "index_age_days": torch.as_tensor(
            [sample["index_age_days"] for sample in samples], dtype=torch.float32
        ),
        "post_index_output_mask": post_index_output_mask,
        "censor_mask": censor_mask,
        "gender_idx": gender,
        "ethnicity_idx": ethnicity,
        "townsend_q": townsend,
        "continuous_static": torch.as_tensor(
            np.stack([sample["continuous_static"] for sample in samples]),
            dtype=torch.float32,
        ),
    }


def make_ar_loader(
    frame: pd.DataFrame,
    split: str,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(CFG.training_seed)
    return DataLoader(
        DelphiARDataset(frame, split),
        batch_size=CFG.batch_size,
        shuffle=shuffle,
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
        collate_fn=collate_delphi_ar,
        generator=generator if shuffle else None,
        persistent_workers=CFG.num_workers > 0,
    )


def deterministic_training_batch(
    dataset: DelphiARDataset,
    iteration: int,
) -> Dict[str, Any]:
    """Return the seed-42 shuffled-epoch batch assigned to one iteration.

    Both participant order and stochastic no-event augmentation are functions
    of the iteration number. Therefore checkpoint resume does not restart or
    silently change the data stream.
    """
    if dataset.split != "train":
        raise ValueError("deterministic_training_batch requires the train split.")
    participant_count = len(dataset)
    batches_per_epoch = int(math.ceil(participant_count / CFG.batch_size))
    zero_based_iteration = int(iteration) - 1
    epoch = zero_based_iteration // batches_per_epoch
    batch_in_epoch = zero_based_iteration % batches_per_epoch
    permutation_rng = np.random.default_rng(np.random.SeedSequence([
        int(CFG.training_seed), 10, int(epoch)
    ]))
    permutation = permutation_rng.permutation(participant_count)
    start = batch_in_epoch * int(CFG.batch_size)
    stop = min(start + int(CFG.batch_size), participant_count)
    selected_indices = permutation[start:stop]

    numpy_state = np.random.get_state()
    padding_seed = int(np.random.SeedSequence([
        int(CFG.training_seed), 11, int(iteration)
    ]).generate_state(1, dtype=np.uint32)[0])
    try:
        np.random.seed(padding_seed)
        samples = [dataset[int(index)] for index in selected_indices]
    finally:
        np.random.set_state(numpy_state)
    return collate_delphi_ar(samples)


def move_batch(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
