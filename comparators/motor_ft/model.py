# MOTOR-FT — Local-causal Transformer backbone, 22-bin survival head, stage-1 weight transfer, and forward checks
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 8 ---
def estimate_train_global_log_rate_initializers(samples, n_conditions):
    """Train-only event / post-entry exposure MLEs for trainable intercepts."""
    Kloc = int(n_conditions)
    entry = float(cfg.ENTRY_DELAY_MONTHS)
    eps = ENTRY_TIME_EPS_MONTHS
    max_horizon = float(haz.MAX_HORIZON_MONTHS)
    condition_events = np.zeros(Kloc, dtype=np.float64)
    condition_exposure = np.zeros(Kloc, dtype=np.float64)
    death_events = 0.0
    death_exposure = 0.0
    for row in samples.itertuples(index=False):
        raw_censor = float(row.C_months)
        censor = float(np.clip(raw_censor, 0.0, max_horizon))
        active = np.asarray(row.cr_at_risk, dtype=float) > 0.5
        types = np.asarray(row.cr_event_type, dtype=np.int64)
        times = np.asarray(row.cr_event_time_months, dtype=np.float64)
        pair_event = np.isfinite(times) & np.isin(types, [int(cfg.CR_TARGET_CONDITION), int(cfg.CR_TARGET_DEATH)])
        eligible = active & np.isfinite(raw_censor) & (censor > entry + eps) & (~(pair_event & (times <= entry + eps)))
        observed_pair = eligible & pair_event & (times > entry + eps) & (times <= censor + eps)
        stop = np.where(observed_pair, np.minimum(times, censor), censor)
        condition_exposure += np.where(eligible, np.maximum(stop - entry, 0.0), 0.0)
        condition_events += (observed_pair & (types == int(cfg.CR_TARGET_CONDITION))).astype(float)

        death_time = float(row.death_time_months)
        death_flag = int(row.death_observed_post_index) == 1 and np.isfinite(death_time)
        death_eligible = np.isfinite(raw_censor) and censor > entry + eps and not (death_flag and death_time <= entry + eps)
        death_observed = death_eligible and death_flag and entry + eps < death_time <= censor + eps
        death_stop = min(death_time, censor) if death_observed else censor
        death_exposure += max(death_stop - entry, 0.0) if death_eligible else 0.0
        death_events += float(death_observed)

    condition_log_rate = np.full(Kloc, float(cfg.CONDITION_BASELINE_FALLBACK_INITIAL_LOG_RATE), dtype=np.float64)
    estimable = (condition_events > 0.0) & (condition_exposure > 0.0)
    condition_log_rate[estimable] = np.log(np.clip(condition_events[estimable] / condition_exposure[estimable], np.exp(LOG_RATE_MIN), np.exp(LOG_RATE_MAX)))
    death_log_rate = float(cfg.DEATH_BASELINE_FALLBACK_INITIAL_LOG_RATE)
    if death_events > 0.0 and death_exposure > 0.0:
        death_log_rate = float(np.log(np.clip(death_events / death_exposure, np.exp(LOG_RATE_MIN), np.exp(LOG_RATE_MAX))))
    return {
        "condition_log_rate": condition_log_rate.astype(np.float32),
        "death_log_rate": np.float32(death_log_rate),
        "condition_events": condition_events, "condition_exposure_months": condition_exposure,
        "death_events": float(death_events), "death_exposure_months": float(death_exposure),
        "initialization_mode": "train_post_entry_mle", "entry_delay_months": entry,
    }


def conditional_window_cif_from_probs(p_condition, p_death_k, p_neither, start_bin, end_bin):
    T = p_condition.shape[-1]
    s = int(max(-1, min(int(start_bin), T - 1)))
    e = int(max(0, min(int(end_bin), T - 1)))
    if e <= s:
        z = torch.zeros_like(p_condition[:, :, 0])
        return {
            "window_cif_condition": z,
            "window_cif_death_k": z,
            "window_survival_end": torch.ones_like(z),
        }
    pc = p_condition[:, :, s + 1 : e + 1]
    pd = p_death_k[:, :, s + 1 : e + 1]
    pn = p_neither[:, :, s + 1 : e + 1]
    one = torch.ones_like(pn[:, :, :1])
    survival_before = torch.cat([one, torch.cumprod(pn[:, :, :-1], dim=-1)], dim=-1)
    return {
        "window_cif_condition": torch.sum(survival_before * pc, dim=-1),
        "window_cif_death_k": torch.sum(survival_before * pd, dim=-1),
        "window_survival_end": torch.prod(pn, dim=-1),
    }


def competing_risk_outputs(disease, death):
    total = disease + death[:, None, :]
    width = float(haz.BIN_WIDTH_MONTHS)
    any_event = -torch.expm1(-total * width)
    return {"condition_rate": disease, "death_rate": death,
            "p_condition": disease / total * any_event,
            "p_death_k": death[:, None, :] / total * any_event,
            "p_neither": torch.exp(-total * width),
            "p_cond_given_alive": -torch.expm1(-disease * width),
            "p_death": -torch.expm1(-death * width)}

LOG_RATE_MIN = float(getattr(cfg, "LOG_RATE_MIN", -18.0))
LOG_RATE_MAX = float(getattr(cfg, "LOG_RATE_MAX", 2.0))
RATE_EPS = float(cfg.RATE_EPS)
initializers = estimate_train_global_log_rate_initializers(train_samples, K)
N_TIME_BINS = haz.N_BINS
class RMSNorm(nn.Module):
    def __init__(self, dimension, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.eps = eps
    def forward(self, value):
        return value * torch.rsqrt(value.float().pow(2).mean(-1, keepdim=True) + self.eps).to(value.dtype) * self.weight

def rotate_every_two(value):
    first, second = value[..., ::2], value[..., 1::2]
    return torch.stack((-second, first), dim=-1).flatten(-2)

def apply_rope(query, key, times_months):
    half = query.shape[-1] // 2
    inverse = 1.0 / (10000.0 ** torch.linspace(0.0, 2.0, half, device=query.device))
    theta = times_months.float()[:, None, :, None] * (365.25 / 12.0) * inverse[None, None, None, :]
    sine = torch.repeat_interleave(theta.sin(), 2, dim=-1).to(query.dtype)
    cosine = torch.repeat_interleave(theta.cos(), 2, dim=-1).to(query.dtype)
    return query * cosine + rotate_every_two(query) * sine, key * cosine + rotate_every_two(key) * sine

class LocalCausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads = CFG.N_HEADS
        self.head_size = CFG.D_MODEL // CFG.N_HEADS
        self.qkv = nn.Linear(CFG.D_MODEL, 3 * CFG.D_MODEL)
        self.output = nn.Linear(CFG.D_MODEL, CFG.D_MODEL)
        self.dropout = nn.Dropout(CFG.DROPOUT)
    def forward(self, value, mask, times):
        batch, length, dimension = value.shape
        qkv = self.qkv(value).reshape(batch, length, 3, self.heads, self.head_size).permute(2, 0, 3, 1, 4)
        query, key = apply_rope(qkv[0], qkv[1], times)
        score = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        position = torch.arange(length, device=value.device)
        delta = position[:, None] - position[None, :]
        allowed = (delta >= 0) & (delta < CFG.ATTENTION_WIDTH)
        score = score.masked_fill(~allowed[None, None], torch.finfo(score.dtype).min)
        score = score.masked_fill(~mask[:, None, None, :], torch.finfo(score.dtype).min)
        probability = self.dropout(score.softmax(-1))
        output = (probability @ qkv[2]).transpose(1, 2).reshape(batch, length, dimension)
        return self.output(output) * mask.unsqueeze(-1)

class MotorBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = RMSNorm(CFG.D_MODEL)
        self.attention = LocalCausalAttention()
        self.norm2 = RMSNorm(CFG.D_MODEL)
        self.ff = nn.Sequential(
            nn.Linear(CFG.D_MODEL, CFG.D_FF), nn.GELU(), nn.Dropout(CFG.DROPOUT),
            nn.Linear(CFG.D_FF, CFG.D_MODEL), nn.Dropout(CFG.DROPOUT),
        )
    def forward(self, value, mask, times):
        value = value + self.attention(self.norm1(value), mask, times)
        value = value + self.ff(self.norm2(value))
        return value * mask.unsqueeze(-1)

class MotorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.event_embedding = nn.Embedding(EV_VOCAB_SIZE_MODEL, CFG.D_MODEL, padding_idx=EV_PAD_ID)
        self.type_embedding = nn.Embedding(TYPE_VOCAB_SIZE_MODEL, CFG.D_MODEL, padding_idx=TYPE_PAD_ID)
        self.value_projection = nn.Sequential(nn.Linear(2, CFG.VALUE_D_MODEL), nn.GELU(), nn.Linear(CFG.VALUE_D_MODEL, CFG.D_MODEL))
        self.gender_embedding = nn.Embedding(N_GENDER_MODEL, CFG.D_MODEL)
        self.ethnicity_embedding = nn.Embedding(N_ETHNICITY_MODEL, CFG.D_MODEL)
        self.townsend_embedding = nn.Embedding(N_TOWNSEND_MODEL, CFG.D_MODEL)
        self.continuous_projection = nn.Sequential(nn.Linear(N_CONTINUOUS_STATIC, CFG.STATIC_D_MODEL), nn.GELU(), nn.Linear(CFG.STATIC_D_MODEL, CFG.D_MODEL))
        self.input_norm = RMSNorm(CFG.D_MODEL)
        self.dropout = nn.Dropout(CFG.DROPOUT)
        self.blocks = nn.ModuleList([MotorBlock() for _ in range(CFG.N_LAYERS)])
        self.output_norm = RMSNorm(CFG.D_MODEL)
    def forward(self, batch):
        pair = torch.stack((batch["event_values"].float(), batch["event_value_masks"].float()), -1)
        static = (
            self.gender_embedding(batch["gender_idx"].clamp(0, N_GENDER_MODEL - 1))
            + self.ethnicity_embedding(batch["ethnicity_idx"].clamp(0, N_ETHNICITY_MODEL - 1))
            + self.townsend_embedding(batch["townsend_q"].clamp(0, N_TOWNSEND_MODEL - 1))
            + self.continuous_projection(batch["continuous_static"].float())
        )
        hidden = self.event_embedding(batch["event_ids"]) + self.type_embedding(batch["event_type_ids"])
        hidden = hidden + self.value_projection(pair) + static[:, None]
        mask = batch["attn_mask"].bool()
        hidden = self.dropout(self.input_norm(hidden)) * mask.unsqueeze(-1)
        for block in self.blocks:
            hidden = block(hidden, mask, batch["event_times_months"])
        hidden = self.output_norm(hidden)
        row = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row, batch["lengths"] - 1]

class FineTuneHead(nn.Module):
    def __init__(self):
        super().__init__()
        feature_dim = CFG.SURVIVAL_DIM - 1
        self.time_projection = nn.Linear(CFG.D_MODEL, N_TIME_BINS * feature_dim)
        self.task_weight = nn.Parameter(torch.empty(K + 1, feature_dim))
        self.task_bias = nn.Parameter(torch.empty(K + 1))
        nn.init.trunc_normal_(self.task_weight, std=1.0 / math.sqrt(CFG.SURVIVAL_DIM))
    def forward(self, pooled):
        representation = self.time_projection(pooled).reshape(pooled.shape[0], N_TIME_BINS, CFG.SURVIVAL_DIM - 1)
        return torch.einsum("bpr,kr->bkp", representation.float(), self.task_weight.float()) + self.task_bias[None, :, None]

class MotorFineTune(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MotorEncoder()
        self.head = FineTuneHead()
    def forward(self, batch):
        log_rates = self.head(self.encoder(batch)).float() * math.log(2.)
        rates = log_rates.clamp(LOG_RATE_MIN, LOG_RATE_MAX).exp().clamp_min(RATE_EPS)
        return competing_risk_outputs(rates[:, :K], rates[:, K])

model = MotorFineTune()
with torch.no_grad():
    log_rates = np.append(initializers["condition_log_rate"], initializers["death_log_rate"])
    model.head.task_bias.copy_(torch.from_numpy(log_rates / math.log(2.)))
def pretraining_origin():
    if WORKFLOW == "evaluation_only" or RESUME:
        # All fine-tuned encoder weights are contained in our own checkpoint.
        recorded = read_json(RUN_DIR / "config.json")["architecture"]["initialization"]
        requested_mode = os.getenv("MOTOR_INIT")
        if requested_mode and requested_mode != recorded["mode"]:
            raise ValueError("MOTOR_INIT conflicts with this run's recorded initialization.")
        return recorded
    mode = os.getenv("MOTOR_INIT", "pretrained")
    if mode == "scratch":
        return {"mode":mode}
    if mode != "pretrained":
        raise ValueError("MOTOR_INIT must be pretrained or scratch.")
    stage1 = Path(os.getenv("MOTOR_PRETRAIN_RUN_DIR", "runs/motor_pretrain"))
    path = Path(os.getenv("MOTOR_PRETRAIN_CHECKPOINT", str(stage1 / "best_pretrained_model_by_val_nll.pt"))).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Stage-1 encoder checkpoint missing: {path}. Set MOTOR_PRETRAIN_CHECKPOINT for fine-tuning or MOTOR_INIT=scratch for the separate architecture control.")
    # Stage-1 bundles may contain NumPy metadata; use only a trusted study checkpoint.
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    hashes = bundle.get("canonical_split_hashes", {})
    splits = {"train":train_samples, "val":val_samples}
    exact = all(hashes.get(k) == MODEL_INPUT_HASHES[f"canonical_{k}_samples_sha256"] for k in splits)
    verification = "same canonical train and validation file hashes"
    if not exact:
        old_dir = os.getenv("MOTOR_PRETRAIN_SOURCE_DIR", "").strip()
        if not old_dir:
            raise ValueError("Stage-1 split hashes differ. To verify a differently serialized export, set MOTOR_PRETRAIN_SOURCE_DIR to its original canonical run.")
        old_dir = Path(old_dir).expanduser()
        old_vocab = read_json(old_dir / "cell3_vocab_scalers_deathsafe.json")
        for group, field in (("event_vocab", "EV_TOKEN_TO_ID"), ("event_type_vocab", "TYPE_TO_ID")):
            if old_vocab[group][field] != VOCAB_METADATA[group][field]:
                raise ValueError(f"Pretraining vocabulary mapping differs: {field}")
        if old_vocab["static_categories"] != VOCAB_METADATA["static_categories"]:
            raise ValueError("Pretraining categorical input encodings differ.")
        for split, current in splits.items():
            original = old_dir / f"canonical_{split}_samples.pkl"
            if sha256_file(original) != hashes.get(split):
                raise ValueError(f"Original {split} export differs from the stage-1 checkpoint.")
            previous = pd.read_pickle(original)
            def indexed_frame(frame):
                return frame.assign(eid=frame.eid.astype(str)).set_index("eid", verify_integrity=True).sort_index()
            old_axis, new_axis = indexed_frame(previous), indexed_frame(current)
            if not old_axis.index.equals(new_axis.index):
                raise ValueError(f"Stage-1 {split} participants differ from the new split.")
            if not pd.to_datetime(old_axis.index_dt).equals(pd.to_datetime(new_axis.index_dt)):
                raise ValueError(f"Stage-1 {split} index dates differ.")
            np.testing.assert_allclose(old_axis.index_age_months, new_axis.index_age_months, rtol=0, atol=1e-4)
        verification = "source hashes verified; same indexed participants and input vocabulary"
    if bundle.get("test_participants_used_for_pretraining_or_selection") is not False:
        raise ValueError("Stage-1 provenance does not establish exclusion of test participants.")
    # Split hashes bind the ordered token IDs in the encoded sample histories.
    # Check the vocabulary hash as well when recorded by the pretraining pipeline.
    vocab_hash = bundle.get("model_input_hashes", {}).get("vocabulary_and_scalers_sha256")
    if exact and vocab_hash and vocab_hash != MODEL_INPUT_HASHES["vocabulary_and_scalers_sha256"]:
        raise ValueError("Stage-1 vocabulary differs.")
    model.encoder.load_state_dict(bundle["encoder_state_dict"], strict=True)
    return {"mode":mode, "encoder_checkpoint_sha256":sha256_file(path),
            "canonical_train_sha256":hashes["train"], "canonical_val_sha256":hashes["val"],
            "verification":verification, "test_used":False, "transferred":"encoder only"}

origin = pretraining_origin()
cfg.MODEL_VARIANT = "Task-adapted MOTOR (" + origin["mode"] + ")"
model.to(DEVICE)
model_meta = {"architecture": ARCHITECTURE_SOURCE, "backbone": vars(CFG), "parameters":sum(p.numel() for p in model.parameters()),
              "initialization":origin, "head_bins":haz.N_BINS}
cfg.ENCODER_LR_MULTIPLIER = 0.1 if origin["mode"] == "pretrained" else 1.0
contract = {"architecture": model_meta, "config": {k:v for k,v in vars(cfg).items() if k != "DEVICE"},
            "model_input_hashes": MODEL_INPUT_HASHES, "condition_names": condition_names,
            "delayed_entry_contract": delayed_entry_contract(), "hazard_binning": vars(haz)}
if (RUN_DIR / "config.json").exists():
    if read_json(RUN_DIR / "config.json") != contract:
        raise ValueError("Run contract differs from the saved comparator configuration.")
elif WORKFLOW == "evaluation_only" or RESUME:
    raise FileNotFoundError(RUN_DIR / "config.json")
else:
    write_json(RUN_DIR / "config.json", contract)
print(cfg.MODEL_VARIANT, model_meta["parameters"], "parameters.", cfg.MAX_ITERATIONS, "planned updates.")


@torch.no_grad()
def check_landmark_forward():
    rows = [train_ds[i] for i in range(min(3, len(train_ds)))]
    batch = fixed_landmark_sequence_collate(rows, training=False)
    keys = ("event_ids", "event_type_ids", "event_times_months", "attn_mask", "lengths", "event_values", "event_value_masks", "gender_idx", "ethnicity_idx", "townsend_q", "continuous_static")
    inputs = {key:batch[key].to(DEVICE) for key in keys}
    model.eval()
    out = model(inputs)
    mass = out["p_condition"] + out["p_death_k"] + out["p_neither"]
    torch.testing.assert_close(mass, torch.ones_like(mass), rtol=1e-5, atol=1e-6)
    previous = torch.zeros_like(mass[:, :, 0])
    for horizon in cfg.EVAL_HORIZONS_MONTHS:
        stop = (cfg.ENTRY_DELAY_MONTHS + int(horizon)) // haz.BIN_WIDTH_MONTHS - 1
        window = conditional_window_cif_from_probs(out["p_condition"], out["p_death_k"], out["p_neither"],
                                                   window_start_bin(cfg.ENTRY_DELAY_MONTHS), stop)
        risk = window["window_cif_condition"]
        if not torch.isfinite(risk).all() or (risk + 1e-6 < previous).any() or (risk > 1 + 1e-6).any():
            raise AssertionError("Invalid conditional disease incidence.")
        previous = risk
    print("Forward checks passed on predictor-only batches for probability mass and post-entry risk curves.")

check_landmark_forward()
