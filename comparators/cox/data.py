# Penalised Cox — Canonical contract validation, full-history features, and competing-risk outcomes
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 9 ---
require_files([
    CANONICAL_PATHS["train"],
    CANONICAL_PATHS["val"],
    CANONICAL_PATHS["target_metadata"],
    CANONICAL_PATHS["target_config"],
    CANONICAL_PATHS["source_config"],
    CANONICAL_PATHS["dataloader_metadata"],
    CANONICAL_PATHS["vocab_scalers"],
    CANONICAL_PATHS["input_manifest"],
    CANONICAL_PATHS["prepared_audit"],
    CANONICAL_PATHS["prepared_manifest"],
])

with CANONICAL_PATHS["target_metadata"].open(encoding="utf-8") as handle:
    TARGET_METADATA = json.load(handle)
with CANONICAL_PATHS["vocab_scalers"].open(encoding="utf-8") as handle:
    VOCAB_METADATA = json.load(handle)
with CANONICAL_PATHS["input_manifest"].open(encoding="utf-8") as handle:
    INPUT_MANIFEST = json.load(handle)
with CANONICAL_PATHS["target_config"].open(encoding="utf-8") as handle:
    TARGET_CONFIG = json.load(handle)
with CANONICAL_PATHS["source_config"].open(encoding="utf-8") as handle:
    SOURCE_MAIN_CONFIG = json.load(handle)
with CANONICAL_PATHS["dataloader_metadata"].open(encoding="utf-8") as handle:
    SOURCE_DATALOADER_METADATA = json.load(handle)

for label, payload in (
    ("target metadata", TARGET_METADATA),
    ("vocabulary/scalers", VOCAB_METADATA),
    ("input manifest", INPUT_MANIFEST),
    ("target config", TARGET_CONFIG),
    ("source config", SOURCE_MAIN_CONFIG),
    ("dataloader metadata", SOURCE_DATALOADER_METADATA),
):
    if not isinstance(payload, dict):
        raise TypeError(f"Canonical {label} must be a JSON object.")

def validate_source_gap_metadata(source_config, target_config, entry_months=6):
    if float(entry_months) != 6.0:
        raise RuntimeError("This comparison requires six-month delayed entry.")
    declared = {}
    for name in ("GAP_WINDOW_MONTHS", "LEAD_TIME_MONTHS", "ENTRY_DELAY_MONTHS"):
        value = source_config.get(name)
        if value is None:
            declared[name] = None
            continue
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Source {name} must be a numeric month count; received {value!r}.") from exc
        if isinstance(value, bool) or number not in (0.0, 6.0):
            raise RuntimeError(f"Source {name}={value!r} is incompatible with this six-month comparison; expected 0 or 6.")
        declared[name] = int(number)
    origins = []
    for metadata in (source_config, target_config):
        for key in ("outcome_time_origin", "target_time_origin"):
            value = metadata.get(key)
            if value is not None:
                normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
                if normalized not in {"index", "original_index", "prediction_index", "canonical_index"}:
                    raise RuntimeError(f"Source {key}={value!r}: supply outcome times relative to the original predictor index. Entry remains month 6.")
                origins.append({"field": key, "value": str(value)})
    return {
        "schema_version": "coxph.source_gap_policy.v7",
        "declared_source_months": declared,
        "source_origin_declarations": origins,
        "required_outcome_time_origin": "original predictor index",
        "experiment_gap_months": 6,
        "experiment_entry_months_from_index": 6,
        "eligibility_application": "intersect canonical risk masks with alive/observed/condition-free status at absolute month 6",
        "event_time_handling": "keep original-index times; subtract 6 once only when forming entry-relative outcomes",
        "additional_gap_added_to_source_gap": False,
        "requires_origin_evidence": any(value == 6 for value in declared.values()) and not origins,
    }


def audit_source_outcome_time_origin(frame, split, tolerance_months=0.1):
    """Corroborate elapsed death times against the persisted absolute ages."""
    required = {"index_age_years", "death_age", "death_time_months", "death_observed_post_index"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{split}: missing original-index time audit fields {sorted(required.difference(frame.columns))}.")
    index_age = pd.to_numeric(frame["index_age_years"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    death_age = pd.to_numeric(frame["death_age"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    times = pd.to_numeric(frame["death_time_months"], errors="raise").to_numpy(dtype=float, na_value=np.nan)
    observed = pd.to_numeric(frame["death_observed_post_index"], errors="raise").to_numpy(dtype=float, na_value=np.nan) > 0.5
    valid = observed & np.isfinite(index_age) & np.isfinite(death_age) & np.isfinite(times)
    expected = 12.0 * (death_age[valid] - index_age[valid])
    offsets = times[valid] - expected
    bad = np.abs(offsets) > float(tolerance_months)
    if np.any(bad):
        median = float(np.median(offsets))
        raise RuntimeError(
            f"{split}: {int(bad.sum())} observed death times disagree with the original index/age coordinates "
            f"(median offset {median:.3f} months). The loader requires original-index outcome times; "
            "a source already shifted to month-six entry must be re-exported with original-index times. "
            "Keep ENTRY_DELAY_MONTHS=6."
        )
    return {"split": split, "observed_death_coordinate_anchors": int(valid.sum()),
        "maximum_absolute_offset_months": float(np.max(np.abs(offsets))) if offsets.size else None,
        "tolerance_months": float(tolerance_months),
        "scope": "death elapsed-time coordinates; condition/censor axes must follow the same source preparation contract"}


def require_source_time_origin_evidence(policy, audits):
    if policy["requires_origin_evidence"] and not any(row["observed_death_coordinate_anchors"] for row in audits):
        raise RuntimeError(
            "The source declares a six-month gap/lead, but its original-index outcome time origin could not be corroborated. "
            "Supply preparation metadata declaring outcome_time_origin='original_index', or canonical original-index labels. "
            "The experiment entry stays at month 6."
        )
    return {**policy, "train_validation_time_origin_audits": audits,
        "test_outcomes_opened_for_this_check": False}


SOURCE_CFG = SOURCE_MAIN_CONFIG.get("config", {})
FIXED_INDEX_CONTRACT = SOURCE_MAIN_CONFIG.get("fixed_index_contract", {})
SOURCE_GAP_POLICY = validate_source_gap_metadata(SOURCE_CFG, TARGET_CONFIG, CFG.gap_months)
if tuple(int(x) for x in SOURCE_CFG.get("EVAL_HORIZONS_MONTHS", ())) != (
    12, 24, 60, 120
):
    raise RuntimeError("Canonical source must report 12/24/60/120-month horizons.")
if int(FIXED_INDEX_CONTRACT.get("calendar_months_from_recruitment", -1)) != 1:
    raise RuntimeError("Canonical source must declare a one-calendar-month cap.")
if int(TARGET_CONFIG.get("maximum_calendar_months_from_recruitment", -1)) != 1:
    raise RuntimeError("Canonical target config must declare a one-calendar-month cap.")
if TARGET_CONFIG.get("indices_sampled") is not False:
    raise RuntimeError("Canonical target config must declare indices_sampled=false.")
if TARGET_CONFIG.get("indices_shifted_in_model") is not False:
    raise RuntimeError("Canonical target config must declare indices_shifted_in_model=false.")

MAIN_METRIC_CONTRACT_EXPLICIT = os.getenv(
    "EHR_MAIN_METRIC_CONTRACT", ""
).strip()
REQUIRE_MAIN_METRIC_CONTRACT = _env_bool(
    "EHR_REQUIRE_MAIN_METRIC_CONTRACT", "NO"
)
metric_contract_candidates = []
if MAIN_METRIC_CONTRACT_EXPLICIT:
    metric_contract_candidates.append(
        Path(MAIN_METRIC_CONTRACT_EXPLICIT).expanduser().resolve()
    )
metric_contract_candidates.extend([
    MAIN_RUN_DIR / "test_metric_definitions.json",
    MAIN_RUN_DIR / "test_evaluation_manifest.json",
    MAIN_RUN_DIR / "metric_definitions.json",
    MAIN_RUN_DIR / "evaluation_manifest.json",
])
MAIN_METRIC_CONTRACT_PATH = next(
    (path for path in metric_contract_candidates if path.is_file()), None
)
if MAIN_METRIC_CONTRACT_EXPLICIT and MAIN_METRIC_CONTRACT_PATH is None:
    raise FileNotFoundError(
        "EHR_MAIN_METRIC_CONTRACT does not identify an existing JSON file."
    )
if REQUIRE_MAIN_METRIC_CONTRACT and MAIN_METRIC_CONTRACT_PATH is None:
    raise FileNotFoundError(
        "No primary-model metric contract was found. Set "
        "EHR_MAIN_METRIC_CONTRACT or disable the strict requirement only "
        "after confirming the first-cell metric settings."
    )
if MAIN_METRIC_CONTRACT_PATH is not None:
    with MAIN_METRIC_CONTRACT_PATH.open(encoding="utf-8") as handle:
        MAIN_METRIC_CONTRACT = json.load(handle)
else:
    MAIN_METRIC_CONTRACT = {}
    print(
        "WARNING: no primary-model metric JSON was found; alignment will be "
        "enforced against canonical target metadata plus the explicit first-cell "
        "metric settings.",
        flush=True,
    )


def _collect_contract_values(value: Any, keys: set[str]) -> List[Any]:
    found: List[Any] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if str(key) in keys:
                found.append(nested)
            found.extend(_collect_contract_values(nested, keys))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found.extend(_collect_contract_values(nested, keys))
    return found


contract_payloads = [
    SOURCE_MAIN_CONFIG,
    SOURCE_DATALOADER_METADATA,
    TARGET_CONFIG,
    TARGET_METADATA,
    INPUT_MANIFEST,
    MAIN_METRIC_CONTRACT,
]


def _unique_int_contract_values(keys: set[str]) -> set[int]:
    result: set[int] = set()
    for payload in contract_payloads:
        for value in _collect_contract_values(payload, keys):
            try:
                result.add(int(value))
            except (TypeError, ValueError):
                continue
    return result


def _unique_tuple_contract_values(keys: set[str]) -> set[Tuple[int, ...]]:
    result: set[Tuple[int, ...]] = set()
    for payload in contract_payloads:
        for value in _collect_contract_values(payload, keys):
            if isinstance(value, (list, tuple)):
                try:
                    result.add(tuple(int(item) for item in value))
                except (TypeError, ValueError):
                    continue
    return result


observed_gap_values = _unique_int_contract_values({
    "gap_months", "prediction_gap_months", "lead_time_gap_months",
    "GAP_WINDOW_MONTHS", "LEAD_TIME_MONTHS",
})
if observed_gap_values.difference({0, 6}):
    raise RuntimeError(f"Source gap/lead declarations must be 0 or 6: {sorted(observed_gap_values)}.")

observed_horizon_values = _unique_tuple_contract_values({
    "report_horizons_months", "reporting_horizons_months",
    "EVAL_HORIZONS_MONTHS", "EVAL_CALIBRATION_HORIZONS_MONTHS",
})
if (
    observed_horizon_values
    and observed_horizon_values != {tuple(CFG.report_horizons_months)}
):
    raise RuntimeError(
        "Primary-model report-horizon contract differs from this comparator: "
        f"observed={sorted(observed_horizon_values)}, "
        f"comparator={tuple(CFG.report_horizons_months)}."
    )

observed_sequence_values = _unique_int_contract_values({
    "max_sequence_len", "MAX_SEQUENCE_LEN"
})
# Sequence length is recorded as evidence only. This comparator deliberately
# summarizes every canonical pre-index event rather than adopting a sequence cap.

ALIGNMENT_CONTRACT_EVIDENCE = {
    "main_metric_contract_path": (
        str(MAIN_METRIC_CONTRACT_PATH)
        if MAIN_METRIC_CONTRACT_PATH is not None else None
    ),
    "main_metric_contract_sha256": (
        sha256_file(MAIN_METRIC_CONTRACT_PATH)
        if MAIN_METRIC_CONTRACT_PATH is not None else None
    ),
    "observed_gap_values": sorted(observed_gap_values),
    "observed_report_horizon_values": [
        list(value) for value in sorted(observed_horizon_values)
    ],
    "observed_sequence_length_values": sorted(observed_sequence_values),
}

CONDITION_NAMES = [str(value) for value in TARGET_METADATA["condition_names"]]
K = int(TARGET_METADATA["K"])
if K != len(CONDITION_NAMES):
    raise ValueError("Inconsistent K and condition_names in canonical metadata.")

HAZARD_METADATA = TARGET_METADATA["hazard_binning"]
BIN_WIDTH_MONTHS = int(HAZARD_METADATA["BIN_WIDTH_MONTHS"])
MAX_HORIZON_MONTHS = int(HAZARD_METADATA["MAX_HORIZON_MONTHS"])
N_BINS = int(HAZARD_METADATA["N_BINS"])
if N_BINS * BIN_WIDTH_MONTHS != MAX_HORIZON_MONTHS:
    raise ValueError("Canonical hazard grid is inconsistent.")
if CFG.gap_months + max(CFG.report_horizons_months) > MAX_HORIZON_MONTHS:
    raise ValueError("Gap plus reporting horizon exceeds the canonical axis.")
if any(horizon % BIN_WIDTH_MONTHS for horizon in CFG.report_horizons_months):
    raise ValueError("Reporting horizons must lie on canonical bin boundaries.")

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
TYPE_VOCAB_SIZE = int(TYPE_META["TYPE_VOCAB_SIZE_MODEL"])
N_GENDER = int(STATIC_META["N_GENDER_MODEL"])
N_ETHNICITY = int(STATIC_META["N_ETHNICITY_MODEL"])
N_TOWNSEND = int(STATIC_META["N_TOWNSEND_MODEL"])

CR_LABELS = TARGET_METADATA["death_adjusted_design"]["cr_target_labels"]
CR_TARGET_CENSOR = int(next(
    key for key, value in CR_LABELS.items()
    if "censor" in str(value).lower() or "neither" in str(value).lower()
))
CR_TARGET_CONDITION = int(next(
    key for key, value in CR_LABELS.items()
    if "condition" in str(value).lower()
))
CR_TARGET_DEATH = int(next(
    key for key, value in CR_LABELS.items()
    if "death" in str(value).lower()
))

EXPECTED_HASHES = INPUT_MANIFEST.get("input_hashes", {})
if not INPUT_MANIFEST.get("samples_built_from_prepared_tables", False):
    raise RuntimeError(
        "model_input_manifest.json does not declare samples_built_from_prepared_tables; "
        "this is not a run of the CRESCENT notebook on the newly prepared data."
    )
if "CRESCENT" not in str(INPUT_MANIFEST.get("model_variant", "")).upper():
    raise RuntimeError("model_input_manifest.json is not from a CRESCENT run.")
PREPARED_AUDIT = pd.read_csv(CANONICAL_PATHS["prepared_audit"]).set_index("split")
with CANONICAL_PATHS["prepared_manifest"].open(encoding="utf-8") as handle:
    PREPARED_SPLITS_MANIFEST = json.load(handle)


def verify_crescent_provenance(split: str, frame=None) -> Dict[str, Any]:
    """Mandatory hash chain: prepared CSV -> manifest -> audit -> sample pickle."""
    observed = sha256_file(CANONICAL_PATHS[split])
    expected = EXPECTED_HASHES.get(f"canonical_{split}_samples_sha256")
    if not expected:
        raise RuntimeError(f"No canonical_{split}_samples_sha256 in model_input_manifest.json.")
    if observed != expected:
        raise RuntimeError(f"Canonical {split} pickle differs from the CRESCENT manifest.")
    if split not in PREPARED_AUDIT.index:
        raise RuntimeError(f"prepared_data_source_audit.csv has no {split} row.")
    audit = PREPARED_AUDIT.loc[split]
    if str(audit["sample_pickle_sha256"]) != observed:
        raise RuntimeError(f"{split} pickle differs from prepared_data_source_audit.csv.")
    prepared_hash = str(PREPARED_SPLITS_MANIFEST["files"][split]["sha256"])
    if str(audit["prepared_file_sha256"]) != prepared_hash:
        raise RuntimeError(
            f"{split}: prepared CSV hash in the audit table differs from "
            "prepared_splits_manifest.json."
        )
    if frame is not None and int(audit["rows"]) != int(len(frame)):
        raise RuntimeError(f"{split} row count differs from the CRESCENT audit table.")
    return {"split": split, "sample_pickle_sha256": observed,
            "prepared_csv_sha256": prepared_hash}


for split in ("train", "val"):
    verify_crescent_provenance(split)

TRAIN_SAMPLES = pd.read_pickle(CANONICAL_PATHS["train"]).copy()
VAL_SAMPLES = pd.read_pickle(CANONICAL_PATHS["val"]).copy()
TRAIN_SAMPLES["split"] = "train"
VAL_SAMPLES["split"] = "val"
CRESCENT_PROVENANCE = [
    verify_crescent_provenance("train", TRAIN_SAMPLES),
    verify_crescent_provenance("val", VAL_SAMPLES),
]

ALIGNMENT_AUDIT = {
    "schema_version": CFG.schema_version,
    "main_run_dir": str(MAIN_RUN_DIR),
    "canonical_files": {
        key: {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for key, path in CANONICAL_PATHS.items()
        if path.is_file() and key not in {"test", "test_index_audit"}
    },
    "train_rows": int(len(TRAIN_SAMPLES)),
    "validation_rows": int(len(VAL_SAMPLES)),
    "train_eid_axis_sha256": canonical_json_hash(
        TRAIN_SAMPLES["eid"].astype(str).tolist()
    ),
    "validation_eid_axis_sha256": canonical_json_hash(
        VAL_SAMPLES["eid"].astype(str).tolist()
    ),
    "condition_axis_sha256": canonical_json_hash(CONDITION_NAMES),
    "gap_months": int(CFG.gap_months),
    "report_horizons_months": list(CFG.report_horizons_months),
    "dense_horizons_months": list(DENSE_HORIZONS_MONTHS),
    "integration_start_months": int(CFG.integration_start_months),
    "cindex_tau_months": int(CFG.ci_cindex_tau_months),
    "ci_point_horizons_months": list(CFG.ci_point_horizons_months),
    "calibration_bins": int(CFG.calibration_bins),
    "minimum_calibration_group_n": int(CFG.min_calibration_group_n),
    "minimum_calibration_bin_n": int(CFG.min_calibration_bin_n),
    "history_contract": "all canonical pre-index events; no truncation",
    "metric_contract_evidence": ALIGNMENT_CONTRACT_EVIDENCE,
    "crescent_provenance_train_val": CRESCENT_PROVENANCE,
    "crescent_model_variant": str(INPUT_MANIFEST.get("model_variant")),
    "test_loaded": False,
}
atomic_json_dump(ALIGNMENT_AUDIT, RUN_DIR / "canonical_alignment_contract.json")

print({
    "canonical_train_rows": len(TRAIN_SAMPLES),
    "canonical_val_rows": len(VAL_SAMPLES),
    "canonical_test_loaded": False,
    "conditions": K,
    "event_vocab_size": EVENT_VOCAB_SIZE,
    "history_contract": "all canonical pre-index events",
    "gap_months": CFG.gap_months,
    "report_horizons_months": CFG.report_horizons_months,
})

# --- Notebook cell 11 ---
# Columns written by the CRESCENT notebook (build_samples in its Cell 4) and used here.
# The old A10 pipeline also stored recruitment_dt, age_at_recruitment, index_source,
# base_random_*, reduced_offset_days_unclipped and history_diag_*; they were only
# validated, never used as predictors or outcomes, and the CRESCENT samples
# (fixed index built from the prepared tables) do not contain them.
REQUIRED_CANONICAL_COLUMNS = {
    "eid", "index_dt", "index_age_years", "index_age_months",
    "index_offset_days_from_recruitment",
    "death_age",
    "history_event_ids", "history_event_type_ids",
    "history_event_times_months", "history_event_values",
    "history_event_value_masks", "gender_idx", "ethnicity_idx",
    "townsend_q", "continuous_static", "at_risk", "event_obs",
    "event_bin", "t_tilde", "C_months", "C_bin",
    "death_observed_post_index", "death_time_months", "death_bin",
    "cr_at_risk", "cr_event_type", "cr_event_bin",
    "cr_event_time_months",
}
AGE_TOL_YEARS = float(
    TARGET_CONFIG.get("age_tolerance_years", 1.0 / 365.25)
)
HISTORY_TIME_TOL_MONTHS = AGE_TOL_YEARS * 12.0 + 1e-6


def _array_length(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return -1


def validate_canonical_samples(frame: pd.DataFrame, split: str) -> None:
    missing = sorted(REQUIRED_CANONICAL_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"{split}: missing canonical columns: {missing}")
    if frame.empty or frame["eid"].astype(str).duplicated().any():
        raise ValueError(f"{split}: expected one non-empty row per participant.")

    for column in (
        "at_risk", "event_obs", "event_bin", "t_tilde",
        "cr_at_risk", "cr_event_type", "cr_event_bin",
        "cr_event_time_months",
    ):
        bad = frame[column].map(_array_length).ne(K)
        if bad.any():
            raise ValueError(
                f"{split}: {column} has rows not length K={K}."
            )

    aligned_columns = (
        "history_event_ids", "history_event_type_ids",
        "history_event_times_months", "history_event_values",
        "history_event_value_masks",
    )
    for row_index, row in frame.iterrows():
        eid = str(row["eid"])
        lengths = [_array_length(row[column]) for column in aligned_columns]
        if min(lengths) < 0 or len(set(lengths)) != 1:
            raise ValueError(
                f"{split} row={row_index} eid={eid}: "
                f"misaligned history arrays {lengths}."
            )
        index_months = float(row["index_age_months"])
        index_years = float(row["index_age_years"])
        if not np.isfinite(index_months) or not np.isfinite(index_years):
            raise ValueError(f"{split} eid={eid}: non-finite index age.")
        if not np.isclose(
            index_months, index_years * 12.0, rtol=0.0, atol=1e-4
        ):
            raise ValueError(f"{split} eid={eid}: inconsistent index age.")

        times = np.asarray(row["history_event_times_months"], dtype=float)
        if not np.all(np.isfinite(times)):
            raise ValueError(f"{split} eid={eid}: non-finite event time.")
        if times.size > 1 and np.any(np.diff(times) < -1e-8):
            raise ValueError(f"{split} eid={eid}: event history is not sorted.")
        if times.size and times.max() > index_months + HISTORY_TIME_TOL_MONTHS:
            raise ValueError(f"{split} eid={eid}: post-index predictor event.")

        event_ids = np.asarray(row["history_event_ids"], dtype=int)
        type_ids = np.asarray(row["history_event_type_ids"], dtype=int)
        if np.any((event_ids < 0) | (event_ids >= EVENT_VOCAB_SIZE)):
            raise ValueError(f"{split} eid={eid}: event ID outside vocabulary.")
        if np.any((type_ids < 0) | (type_ids >= TYPE_VOCAB_SIZE)):
            raise ValueError(
                f"{split} eid={eid}: event type outside vocabulary."
            )


def compare_with_main_index_audit(
    frame: pd.DataFrame, split: str
) -> None:
    audit_path = CANONICAL_PATHS.get(f"{split}_index_audit")
    if audit_path is None or not Path(audit_path).is_file():
        return
    audit = pd.read_csv(audit_path, dtype={"eid": str})
    columns = [
        "eid", "index_dt", "index_age_years",
        "index_offset_days_from_recruitment",
    ]
    if not set(columns).issubset(audit.columns):
        raise ValueError(f"{split}: main index audit is incomplete.")
    left = frame[columns].copy()
    left["eid"] = left["eid"].astype(str)
    right = audit[columns].copy()
    merged = left.merge(
        right, on="eid", how="outer", suffixes=("_sample", "_audit"),
        indicator=True, validate="one_to_one",
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError(f"{split}: index-audit participant mismatch.")
    date_bad = (
        pd.to_datetime(merged["index_dt_sample"])
        != pd.to_datetime(merged["index_dt_audit"])
    )
    age_bad = ~np.isclose(
        merged["index_age_years_sample"].to_numpy(float),
        merged["index_age_years_audit"].to_numpy(float),
        rtol=0.0, atol=1e-12,
    )
    offset_bad = (
        merged["index_offset_days_from_recruitment_sample"].to_numpy(float)
        != merged["index_offset_days_from_recruitment_audit"].to_numpy(float)
    )
    if np.any(date_bad | age_bad | offset_bad):
        raise RuntimeError(f"{split}: locked-index audit mismatch.")


def history_length_audit(frame: pd.DataFrame, split: str) -> Dict[str, Any]:
    lengths = frame["history_event_ids"].map(len).to_numpy(int)
    return {
        "split": split,
        "participants": int(len(frame)),
        "participants_with_zero_events": int(np.sum(lengths == 0)),
        "minimum_events": int(lengths.min()),
        "median_events": float(np.median(lengths)),
        "p95_events": float(np.quantile(lengths, 0.95)),
        "maximum_events": int(lengths.max()),
        "total_events_available": int(lengths.sum()),
        "total_events_used": int(lengths.sum()),
        "events_discarded": 0,
    }


for split_name, frame in (
    ("train", TRAIN_SAMPLES), ("val", VAL_SAMPLES)
):
    validate_canonical_samples(frame, split_name)
    compare_with_main_index_audit(frame, split_name)

if set(TRAIN_SAMPLES["eid"].astype(str)) & set(
    VAL_SAMPLES["eid"].astype(str)
):
    raise RuntimeError("Participant leakage between train and validation.")

SOURCE_GAP_AUDIT = require_source_time_origin_evidence(
    SOURCE_GAP_POLICY,
    [audit_source_outcome_time_origin(TRAIN_SAMPLES, "train"),
     audit_source_outcome_time_origin(VAL_SAMPLES, "val")],
)
atomic_json_dump(SOURCE_GAP_AUDIT, RUN_DIR / "source_gap_time_origin_audit.json")

HISTORY_AUDIT_TRAIN_VAL = pd.DataFrame([
    history_length_audit(TRAIN_SAMPLES, "train"),
    history_length_audit(VAL_SAMPLES, "val"),
])
HISTORY_AUDIT_TRAIN_VAL.to_csv(
    RUN_DIR / "full_history_audit_train_val.csv", index=False
)
print(HISTORY_AUDIT_TRAIN_VAL.to_string(index=False))
print("Full-history audit passed: no canonical events are truncated.")

N_CONTINUOUS_STATIC = int(
    np.asarray(TRAIN_SAMPLES.iloc[0]["continuous_static"]).size
)
if N_CONTINUOUS_STATIC < 1:
    raise ValueError("Canonical continuous_static vectors are empty.")



# -----------------------------------------------------------------------------
# config.json: delayed-entry contract in the layout the CRESCENT notebook reads
# from comparators. The likelihood is stated truthfully: this model is a
# cause-specific Cox partial likelihood on the month-six eligible risk set.
# -----------------------------------------------------------------------------
DELAYED_ENTRY_CONTRACT = {
    "schema_version": DELAYED_ENTRY_SCHEMA_VERSION,
    "entry_delay_months": float(CFG.gap_months),
    "entry_time_eps_months": float(ENTRY_TIME_EPS_MONTHS),
    "entry_rule": "condition-free, alive, and observed strictly beyond entry",
    "events_on_or_before_entry": "excluded from the condition-specific risk set",
    "death_entry_rule": "alive and observed strictly beyond entry",
    "training_likelihood": (
        "cause-specific Cox partial likelihood over the month-six eligible risk set "
        "(common-entry landmark; equivalent to delayed entry at month six)"
    ),
    "predictor_cutoff": "original one-month-cap hybrid index; no entry-delay predictors added",
    "reported_horizons": "months after the delayed-entry landmark",
    "horizons_months_after_entry": list(map(int, CFG.report_horizons_months)),
    "maximum_report_endpoint_months_from_index": int(CFG.gap_months + max(CFG.report_horizons_months)),
    "implementation_status": "implemented in outcomes, risk sets, censoring KMs and evaluation",
}
atomic_json_dump({
    "model_variant": CFG.model_name,
    "config": {
        **asdict(CFG),
        "ENTRY_DELAY_MONTHS": int(CFG.gap_months),
        "EVAL_HORIZONS_MONTHS": list(map(int, CFG.report_horizons_months)),
    },
    "delayed_entry_contract": DELAYED_ENTRY_CONTRACT,
    "crescent_run_dir": str(MAIN_RUN_DIR),
    "crescent_provenance_train_val": CRESCENT_PROVENANCE,
}, RUN_DIR / "config.json")

# --- Notebook cell 13 ---
class CoxFullHistoryFeatureBuilder:
    """Deterministic sparse summaries of every canonical pre-index event."""

    def __init__(
        self,
        event_vocab_size: int,
        type_vocab_size: int,
        n_gender: int,
        n_ethnicity: int,
        n_townsend: int,
        n_continuous: int,
        include_extended: bool,
    ):
        self.event_vocab_size = int(event_vocab_size)
        self.type_vocab_size = int(type_vocab_size)
        self.n_gender = int(n_gender)
        self.n_ethnicity = int(n_ethnicity)
        self.n_townsend = int(n_townsend)
        self.n_continuous = int(n_continuous)
        self.include_extended = bool(include_extended)
        self.fitted = False

        self.global_names = [
            "index_age_years",
            "history_event_count",
            "history_unique_event_count",
            "history_span_months",
            "history_most_recent_recency_months",
            "history_mean_gap_months",
            "history_max_gap_months",
            "history_events_age_lt40",
            "history_events_age_40_to_60",
            "history_events_age_ge60",
            "history_valued_event_count",
        ]

        offset = 0
        self.slices: Dict[str, slice] = {}
        self.slices["continuous"] = slice(offset, offset + self.n_continuous)
        offset += self.n_continuous
        self.slices["gender"] = slice(offset, offset + self.n_gender)
        offset += self.n_gender
        self.slices["ethnicity"] = slice(offset, offset + self.n_ethnicity)
        offset += self.n_ethnicity
        self.slices["townsend"] = slice(offset, offset + self.n_townsend)
        offset += self.n_townsend
        self.slices["global"] = slice(offset, offset + len(self.global_names))
        offset += len(self.global_names)
        self.slices["type_count"] = slice(offset, offset + self.type_vocab_size)
        offset += self.type_vocab_size

        event_blocks = [
            "event_present",
            "event_most_recent_recency_months",
        ]
        if self.include_extended:
            event_blocks.extend([
                "event_log1p_count",
                "event_first_age_years",
                "event_last_age_years",
                "event_span_months",
                "event_last_value",
                "event_last_value_present",
            ])
        self.event_blocks = tuple(event_blocks)
        for block in self.event_blocks:
            self.slices[block] = slice(offset, offset + self.event_vocab_size)
            offset += self.event_vocab_size
        self.n_features = int(offset)

    def feature_names(self) -> List[str]:
        names: List[str] = []
        static_names = VOCAB_METADATA.get(
            "sample_continuous_cols_raw",
            [f"continuous_{j}" for j in range(self.n_continuous)],
        )
        if len(static_names) != self.n_continuous:
            static_names = [f"continuous_{j}" for j in range(self.n_continuous)]
        names.extend([f"static::{value}" for value in static_names])
        names.extend([f"gender::{j}" for j in range(self.n_gender)])
        names.extend([f"ethnicity::{j}" for j in range(self.n_ethnicity)])
        names.extend([f"townsend_q::{j}" for j in range(self.n_townsend)])
        names.extend([f"global::{value}" for value in self.global_names])
        names.extend([
            f"event_type_count::{j}" for j in range(self.type_vocab_size)
        ])
        for block in self.event_blocks:
            names.extend([
                f"{block}::{event_id}"
                for event_id in range(self.event_vocab_size)
            ])
        if len(names) != self.n_features:
            raise AssertionError("Feature-name axis is inconsistent.")
        return names

    @staticmethod
    def _full_history(row: pd.Series) -> Tuple[np.ndarray, ...]:
        return (
            np.asarray(row["history_event_ids"], dtype=np.int64),
            np.asarray(row["history_event_type_ids"], dtype=np.int64),
            np.asarray(row["history_event_times_months"], dtype=np.float64),
            np.asarray(row["history_event_values"], dtype=np.float64),
            np.asarray(row["history_event_value_masks"], dtype=np.float64),
        )

    def _raw_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        rows: List[int] = []
        columns: List[int] = []
        data: List[float] = []

        def add(
            row_index: int,
            column: int,
            value: float,
            keep_zero: bool = False,
        ) -> None:
            if np.isfinite(value) and (keep_zero or value != 0.0):
                rows.append(int(row_index))
                columns.append(int(column))
                data.append(float(value))

        for i, (_, row) in enumerate(frame.iterrows()):
            continuous = np.asarray(row["continuous_static"], dtype=np.float64)
            if continuous.size != self.n_continuous:
                raise ValueError("Inconsistent continuous_static length.")
            if not np.all(np.isfinite(continuous)):
                raise ValueError(
                    "continuous_static contains non-finite values; use the exact "
                    "canonical preprocessed sample files."
                )
            for j, value in enumerate(continuous):
                add(i, self.slices["continuous"].start + j, value, keep_zero=True)

            for key, n_categories, value in (
                ("gender", self.n_gender, int(row["gender_idx"])),
                ("ethnicity", self.n_ethnicity, int(row["ethnicity_idx"])),
                ("townsend", self.n_townsend, int(row["townsend_q"])),
            ):
                if not 0 <= value < n_categories:
                    raise ValueError(f"Categorical index outside {key} vocabulary.")
                add(i, self.slices[key].start + value, 1.0)

            ids, types, times, values, masks = self._full_history(row)
            index_months = float(row["index_age_months"])
            recency_months = np.maximum(index_months - times, 0.0)
            unique_ids, counts = np.unique(ids, return_counts=True)

            if len(times):
                gaps_months = np.maximum(np.diff(times), 0.0)
                global_values = [
                    float(row["index_age_years"]),
                    float(len(ids)),
                    float(len(unique_ids)),
                    max(float(times[-1] - times[0]), 0.0),
                    max(float(recency_months[-1]), 0.0),
                    float(gaps_months.mean()) if gaps_months.size else 0.0,
                    float(gaps_months.max()) if gaps_months.size else 0.0,
                    float(np.sum(times / 12.0 < 40.0)),
                    float(np.sum((times / 12.0 >= 40.0) & (times / 12.0 < 60.0))),
                    float(np.sum(times / 12.0 >= 60.0)),
                    float(np.sum(masks > 0.5)),
                ]
            else:
                global_values = [
                    float(row["index_age_years"]),
                    *([0.0] * (len(self.global_names) - 1)),
                ]
            for j, value in enumerate(global_values):
                add(i, self.slices["global"].start + j, value, keep_zero=True)

            if len(types):
                type_ids, type_counts = np.unique(types, return_counts=True)
                for type_id, count in zip(type_ids, type_counts):
                    add(
                        i,
                        self.slices["type_count"].start + int(type_id),
                        float(count),
                    )

            for event_id, count in zip(unique_ids, counts):
                positions = np.flatnonzero(ids == event_id)
                first_position = int(positions[0])
                last_position = int(positions[-1])
                last_recency_months = float(recency_months[last_position])
                span_months = max(
                    float(times[last_position] - times[first_position]), 0.0
                )
                add(
                    i,
                    self.slices["event_present"].start + int(event_id),
                    1.0,
                )
                add(
                    i,
                    self.slices["event_most_recent_recency_months"].start
                    + int(event_id),
                    last_recency_months,
                    keep_zero=True,
                )
                if self.include_extended:
                    add(
                        i,
                        self.slices["event_log1p_count"].start + int(event_id),
                        float(np.log1p(count)),
                    )
                    add(
                        i,
                        self.slices["event_first_age_years"].start
                        + int(event_id),
                        float(times[first_position] / 12.0),
                        keep_zero=True,
                    )
                    add(
                        i,
                        self.slices["event_last_age_years"].start
                        + int(event_id),
                        float(times[last_position] / 12.0),
                        keep_zero=True,
                    )
                    add(
                        i,
                        self.slices["event_span_months"].start + int(event_id),
                        span_months,
                        keep_zero=True,
                    )
                    measured_positions = positions[masks[positions] > 0.5]
                    if measured_positions.size:
                        last_measured_position = int(measured_positions[-1])
                        last_measured_value = float(values[last_measured_position])
                        if np.isfinite(last_measured_value):
                            add(
                                i,
                                self.slices["event_last_value"].start
                                + int(event_id),
                                last_measured_value,
                                keep_zero=True,
                            )
                            add(
                                i,
                                self.slices["event_last_value_present"].start
                                + int(event_id),
                                1.0,
                            )

        matrix = sparse.coo_matrix(
            (data, (rows, columns)),
            shape=(len(frame), self.n_features),
            dtype=np.float32,
        ).tocsr()
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        return matrix

    def fit_transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        matrix = self._raw_matrix(frame)
        self.fitted = True
        return matrix

    def transform(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        if not self.fitted:
            raise RuntimeError("Feature builder must be fitted on training.")
        return self._raw_matrix(frame)


FEATURE_BUILDER = CoxFullHistoryFeatureBuilder(
    event_vocab_size=EVENT_VOCAB_SIZE,
    type_vocab_size=TYPE_VOCAB_SIZE,
    n_gender=N_GENDER,
    n_ethnicity=N_ETHNICITY,
    n_townsend=N_TOWNSEND,
    n_continuous=N_CONTINUOUS_STATIC,
    include_extended=CFG.include_extended_history_features,
)
X_TRAIN_RAW = FEATURE_BUILDER.fit_transform(TRAIN_SAMPLES)
X_VAL_RAW = FEATURE_BUILDER.transform(VAL_SAMPLES)
RAW_FEATURE_NAMES = FEATURE_BUILDER.feature_names()

if X_TRAIN_RAW.shape[1] != len(RAW_FEATURE_NAMES):
    raise AssertionError("Raw training matrix and feature names are misaligned.")
if not np.isfinite(X_TRAIN_RAW.data).all() or not np.isfinite(X_VAL_RAW.data).all():
    raise ValueError("Non-finite stored CoxPH features.")

# Keep all structural/static/global/type features. Event IDs are retained as a
# group when their training-split presence count reaches the threshold, so all
# summaries for one retained event stay together.
event_present_slice = FEATURE_BUILDER.slices["event_present"]
event_presence = X_TRAIN_RAW[:, event_present_slice].tocsc()
event_training_nonzero = np.diff(event_presence.indptr).astype(np.int64)
EVENT_FEATURE_KEEP = event_training_nonzero >= CFG.min_event_feature_nonzero
FEATURE_MASK = np.zeros(FEATURE_BUILDER.n_features, dtype=bool)
FEATURE_MASK[: event_present_slice.start] = True
for block in FEATURE_BUILDER.event_blocks:
    FEATURE_MASK[FEATURE_BUILDER.slices[block]] = EVENT_FEATURE_KEEP
if not FEATURE_MASK.any():
    raise RuntimeError("Training-only feature filter removed the complete feature axis.")

FEATURE_NAMES = [
    name for name, keep in zip(RAW_FEATURE_NAMES, FEATURE_MASK) if keep
]
SCALER = StandardScaler(with_mean=False, copy=True)
SCALER.fit(X_TRAIN_RAW[:, FEATURE_MASK])
FEATURE_SCALE = np.asarray(SCALER.scale_, dtype=np.float64)
FEATURE_SCALE[~np.isfinite(FEATURE_SCALE) | (FEATURE_SCALE <= 0.0)] = 1.0


def apply_frozen_feature_contract(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    if matrix.shape[1] != FEATURE_MASK.size:
        raise RuntimeError("Raw feature axis differs from the frozen feature contract.")
    transformed = matrix[:, FEATURE_MASK].multiply(1.0 / FEATURE_SCALE)
    result = transformed.tocsr().astype(np.float32)
    result.eliminate_zeros()
    if not np.isfinite(result.data).all():
        raise ValueError("Feature scaling produced non-finite values.")
    return result


X_TRAIN = apply_frozen_feature_contract(X_TRAIN_RAW)
X_VAL = apply_frozen_feature_contract(X_VAL_RAW)
pd.DataFrame({
    "event_id": np.arange(EVENT_VOCAB_SIZE, dtype=int),
    "training_participants_with_event": event_training_nonzero,
    "retained": EVENT_FEATURE_KEEP,
    "minimum_training_nonzero": int(CFG.min_event_feature_nonzero),
}).to_csv(RUN_DIR / "full_history_event_feature_filter.csv", index=False)

FEATURE_CONTRACT = {
    "raw_feature_count": int(len(RAW_FEATURE_NAMES)),
    "selected_feature_count": int(len(FEATURE_NAMES)),
    "retained_event_ids": int(EVENT_FEATURE_KEEP.sum()),
    "event_vocab_size": int(EVENT_VOCAB_SIZE),
    "minimum_event_training_nonzero": int(CFG.min_event_feature_nonzero),
    "raw_feature_axis_sha256": canonical_json_hash(RAW_FEATURE_NAMES),
    "selected_feature_axis_sha256": canonical_json_hash(FEATURE_NAMES),
    "feature_mask_sha256": sha256_array(FEATURE_MASK.astype(np.uint8)),
    "feature_scale_sha256": sha256_array(FEATURE_SCALE),
    "scaling": "training-only StandardScaler(with_mean=False)",
    "history_contract": "all canonical pre-index events; no truncation",
}
atomic_json_dump(FEATURE_CONTRACT, RUN_DIR / "full_history_feature_contract.json")

print({
    "train_feature_shape": X_TRAIN.shape,
    "validation_feature_shape": X_VAL.shape,
    "train_nonzero_values": int(X_TRAIN.nnz),
    "raw_features": len(RAW_FEATURE_NAMES),
    "selected_features": len(FEATURE_NAMES),
    "retained_event_ids": int(EVENT_FEATURE_KEEP.sum()),
    "history_contract": "full history",
    "feature_mode": (
        "full_history_recency_plus_engineered"
        if CFG.include_extended_history_features
        else "full_history_recency"
    ),
})

del X_TRAIN_RAW, X_VAL_RAW, event_presence

# --- Notebook cell 15 ---
def make_condition_outcomes(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    at_risk = np.stack(frame["cr_at_risk"].to_numpy()).astype(float) > 0.5
    event_type = np.stack(frame["cr_event_type"].to_numpy()).astype(np.int8)
    event_time = np.stack(frame["cr_event_time_months"].to_numpy()).astype(float)
    censor_time_from_index = frame["C_months"].to_numpy(float)
    analysis_horizon = float(max(CFG.report_horizons_months))

    finite_event = np.isfinite(event_time)
    event_before_or_in_gap = (
        at_risk
        & finite_event
        & np.isin(event_type, [CR_TARGET_CONDITION, CR_TARGET_DEATH])
        & (event_time <= CFG.gap_months + ENTRY_TIME_EPS_MONTHS)
    )
    # CRESCENT contract: observed strictly beyond month six.
    eligible = (
        at_risk
        & np.isfinite(censor_time_from_index[:, None])
        & (censor_time_from_index[:, None] > CFG.gap_months + ENTRY_TIME_EPS_MONTHS)
        & ~event_before_or_in_gap
    )

    followup_after_gap = np.maximum(
        censor_time_from_index - CFG.gap_months, 0.0
    )
    exit_time_untruncated = np.broadcast_to(
        followup_after_gap[:, None], event_type.shape
    ).copy()
    status_untruncated = np.zeros_like(event_type, dtype=np.int8)

    event_time_after_gap = event_time - CFG.gap_months
    declared_condition = (
        eligible & finite_event & (event_type == CR_TARGET_CONDITION)
        & (event_time_after_gap > ENTRY_TIME_EPS_MONTHS)
    )
    declared_death = (
        eligible & finite_event & (event_type == CR_TARGET_DEATH)
        & (event_time_after_gap > ENTRY_TIME_EPS_MONTHS)
    )
    declared_event = declared_condition | declared_death
    after_observed_followup = declared_event & (
        event_time_after_gap
        > followup_after_gap[:, None] + HISTORY_TIME_TOL_MONTHS
    )
    if after_observed_followup.any():
        first = np.argwhere(after_observed_followup)[0].tolist()
        raise ValueError(
            "Canonical competing-risk event occurs after C_months at "
            f"row/condition {first}."
        )

    status_untruncated[declared_condition] = 1
    status_untruncated[declared_death] = 2
    exit_time_untruncated[declared_event] = event_time_after_gap[declared_event]

    within_analysis_horizon = declared_event & (
        event_time_after_gap <= analysis_horizon + 1e-8
    )
    status = np.where(within_analysis_horizon, status_untruncated, 0).astype(np.int8)
    exit_time = np.minimum(exit_time_untruncated, analysis_horizon)

    invalid = eligible & (
        ~np.isfinite(exit_time) | (exit_time < -1e-6)
    )
    if invalid.any():
        first = np.argwhere(invalid)[0].tolist()
        raise ValueError(
            f"Invalid condition lead-time outcome at row/condition {first}."
        )
    return {
        "eligible": eligible,
        "status": status,
        "status_untruncated": status_untruncated,
        "exit_time": np.maximum(exit_time, 0.0),
        "exit_time_untruncated": np.maximum(exit_time_untruncated, 0.0),
        "event_type": event_type,
        "event_time_from_index": event_time,
    }


def make_death_outcomes(frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    censor_time_from_index = frame["C_months"].to_numpy(float)
    death_observed = frame["death_observed_post_index"].to_numpy(float) > 0.5
    death_time_from_index = frame["death_time_months"].to_numpy(float)
    analysis_horizon = float(max(CFG.report_horizons_months))
    finite_death = death_observed & np.isfinite(death_time_from_index)
    death_before_or_in_gap = finite_death & (
        death_time_from_index <= CFG.gap_months + ENTRY_TIME_EPS_MONTHS
    )
    eligible = (
        np.isfinite(censor_time_from_index)
        & (censor_time_from_index > CFG.gap_months + ENTRY_TIME_EPS_MONTHS)
        & ~death_before_or_in_gap
    )
    followup_after_gap = np.maximum(
        censor_time_from_index - CFG.gap_months, 0.0
    )
    death_time_after_gap = death_time_from_index - CFG.gap_months
    declared_death = (
        eligible & finite_death & (death_time_after_gap > ENTRY_TIME_EPS_MONTHS)
    )
    after_observed_followup = declared_death & (
        death_time_after_gap > followup_after_gap + HISTORY_TIME_TOL_MONTHS
    )
    if after_observed_followup.any():
        first = int(np.flatnonzero(after_observed_followup)[0])
        raise ValueError(
            f"Canonical death occurs after C_months at row {first}."
        )

    status_untruncated = declared_death.astype(np.int8)
    exit_time_untruncated = followup_after_gap.copy()
    exit_time_untruncated[declared_death] = death_time_after_gap[declared_death]
    within_analysis_horizon = declared_death & (
        death_time_after_gap <= analysis_horizon + 1e-8
    )
    status = within_analysis_horizon.astype(np.int8)
    exit_time = np.minimum(exit_time_untruncated, analysis_horizon)
    return {
        "eligible": eligible,
        "status": status,
        "status_untruncated": status_untruncated,
        "exit_time": np.maximum(exit_time, 0.0),
        "exit_time_untruncated": np.maximum(exit_time_untruncated, 0.0),
    }


TRAIN_CONDITION_OUTCOMES = make_condition_outcomes(TRAIN_SAMPLES)
VAL_CONDITION_OUTCOMES = make_condition_outcomes(VAL_SAMPLES)
TRAIN_DEATH_OUTCOMES = make_death_outcomes(TRAIN_SAMPLES)
VAL_DEATH_OUTCOMES = make_death_outcomes(VAL_SAMPLES)

print({
    "training_condition_events_total": int(np.sum(
        TRAIN_CONDITION_OUTCOMES["status"] == 1
    )),
    "validation_condition_events_total": int(np.sum(
        VAL_CONDITION_OUTCOMES["status"] == 1
    )),
    "training_deaths": int(np.sum(TRAIN_DEATH_OUTCOMES["status"] == 1)),
    "validation_deaths": int(np.sum(VAL_DEATH_OUTCOMES["status"] == 1)),
})
