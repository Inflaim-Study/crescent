# SurvTRACE-aligned — Feature tokenizer, Transformer encoder, and interval hazard heads
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

N_BINS = haz.N_BINS
EVENT_VOCAB_SIZE, EVENT_PAD_ID, TYPE_VOCAB_SIZE = EV_VOCAB_SIZE_MODEL, EV_PAD_ID, TYPE_VOCAB_SIZE_MODEL
N_GENDER, N_ETHNICITY, N_TOWNSEND = N_GENDER_MODEL, N_ETHNICITY_MODEL, N_TOWNSEND_MODEL
TRAIN_CONDITION_INITIAL_LOG_RATE = initializers["condition_log_rate"]
TRAIN_DEATH_INITIAL_LOG_RATE = initializers["death_log_rate"]
# =============================================================================
# SurvTRACE-style feature tokenizer and Transformer
# =============================================================================

class NumericalFeatureEmbedding(nn.Module):
    """One learnable affine vector embedding per scalar numerical feature."""

    def __init__(self, n_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2 or values.shape[1] != self.weight.shape[0]:
            raise ValueError("Numerical feature matrix has the wrong shape.")
        return values.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class FeatureTransformerBlock(nn.Module):
    """Pre-norm self-attention/FFN block used by the adapted SurvTRACE encoder."""

    def __init__(self):
        super().__init__()
        d = CFG.d_model
        self.attention_norm = nn.LayerNorm(d)
        self.attention = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=CFG.n_heads,
            dropout=CFG.dropout,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(CFG.dropout)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, CFG.ff_dim),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(CFG.ff_dim, d),
        )
        self.ffn_dropout = nn.Dropout(CFG.dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        tokens = tokens + self.attention_dropout(attended)
        tokens = tokens + self.ffn_dropout(self.ffn(self.ffn_norm(tokens)))
        if key_padding_mask is not None:
            tokens = tokens.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return tokens


class SurvTRACEFeatureEncoder(nn.Module):
    """Feature-token encoder retaining raw and contextualized feature shortcuts."""

    CONDITION_TYPE = 0
    GENDER_TYPE = 1
    ETHNICITY_TYPE = 2
    TOWNSEND_TYPE = 3
    CONTINUOUS_TYPE = 4
    HISTORY_TYPE = 5
    AGE_TYPE = 6
    TEMPORAL_EVENT_TYPE = 7
    N_FEATURE_TYPES = 8

    def __init__(self):
        super().__init__()
        d = CFG.d_model
        self.fixed_tokens = (
            K + 3 + N_CONTINUOUS_STATIC + N_HISTORY_GLOBAL_FEATURES + 1
        )
        self.total_tokens = self.fixed_tokens + int(CFG.max_event_feature_tokens)
        if self.total_tokens > CFG.max_feature_tokens:
            raise ValueError(
                f"SurvTRACE feature count {self.total_tokens} exceeds "
                f"EHR_SURVTRACE_MAX_FEATURE_TOKENS={CFG.max_feature_tokens}."
            )

        self.condition_id = nn.Embedding(K, d)
        self.condition_state = nn.Embedding(2, d)
        self.condition_history = nn.Linear(N_HISTORY_CONDITION_CHANNELS, d)
        self.gender = nn.Embedding(N_GENDER, d)
        self.ethnicity = nn.Embedding(N_ETHNICITY, d)
        self.townsend = nn.Embedding(N_TOWNSEND, d)
        self.continuous = NumericalFeatureEmbedding(N_CONTINUOUS_STATIC, d)
        self.global_history = NumericalFeatureEmbedding(
            N_HISTORY_GLOBAL_FEATURES, d
        )
        self.index_age = NumericalFeatureEmbedding(1, d)
        self.temporal_event_id = nn.Embedding(
            EVENT_VOCAB_SIZE, d, padding_idx=EVENT_PAD_ID
        )
        self.temporal_event_type = nn.Embedding(
            TYPE_VOCAB_SIZE, d, padding_idx=TYPE_PAD_ID
        )
        self.temporal_event_features = nn.Linear(
            N_EVENT_FEATURE_CHANNELS, d
        )
        self.feature_type = nn.Embedding(self.N_FEATURE_TYPES, d)
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(CFG.dropout)
        self.blocks = nn.ModuleList(
            [FeatureTransformerBlock() for _ in range(CFG.n_layers)]
        )
        self.output_norm = nn.LayerNorm(d)
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.condition_fusion = nn.Sequential(
            nn.LayerNorm(2 * d),
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.register_buffer(
            "continuous_mean",
            torch.as_tensor(CONTINUOUS_STATIC_MEAN, dtype=torch.float32),
        )
        self.register_buffer(
            "continuous_std",
            torch.as_tensor(CONTINUOUS_STATIC_STD, dtype=torch.float32),
        )
        self.register_buffer(
            "condition_ids", torch.arange(K, dtype=torch.long), persistent=False
        )

    def _add_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_tensor = torch.full(
            tokens.shape[:2], int(type_id), dtype=torch.long, device=tokens.device
        )
        return tokens + self.feature_type(type_tensor)

    def tokenize(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch["gender_idx"].shape[0]
        condition_state = (batch["condition_summary"][..., 0] > 0.5).long()
        condition = (
            self.condition_id(self.condition_ids).unsqueeze(0).expand(
                batch_size, -1, -1
            )
            + self.condition_state(condition_state)
            + self.condition_history(batch["condition_summary"].float())
        )
        condition = self._add_type(condition, self.CONDITION_TYPE)

        gender = self._add_type(
            self.gender(batch["gender_idx"]).unsqueeze(1), self.GENDER_TYPE
        )
        ethnicity = self._add_type(
            self.ethnicity(batch["ethnicity_idx"]).unsqueeze(1),
            self.ETHNICITY_TYPE,
        )
        townsend = self._add_type(
            self.townsend(batch["townsend_q"]).unsqueeze(1), self.TOWNSEND_TYPE
        )

        continuous_values = (
            batch["continuous_static"].float() - self.continuous_mean
        ) / self.continuous_std
        continuous = self._add_type(
            self.continuous(continuous_values), self.CONTINUOUS_TYPE
        )
        history = self._add_type(
            self.global_history(batch["global_summary"].float()),
            self.HISTORY_TYPE,
        )
        age = (
            batch["index_age_months"].float() - float(INDEX_AGE_MEAN_MONTHS)
        ) / float(INDEX_AGE_STD_MONTHS)
        age = self._add_type(self.index_age(age[:, None]), self.AGE_TYPE)

        temporal_event = (
            self.temporal_event_id(batch["event_feature_ids"].long())
            + self.temporal_event_type(
                batch["event_feature_type_ids"].long()
            )
            + self.temporal_event_features(
                batch["event_feature_values"].float()
            )
        )
        temporal_event = self._add_type(
            temporal_event, self.TEMPORAL_EVENT_TYPE
        )

        tokens = torch.cat(
            [
                condition,
                gender,
                ethnicity,
                townsend,
                continuous,
                history,
                age,
                temporal_event,
            ],
            dim=1,
        )
        fixed_padding = torch.zeros(
            (batch_size, self.fixed_tokens),
            dtype=torch.bool,
            device=tokens.device,
        )
        padding_mask = torch.cat(
            [fixed_padding, ~batch["event_feature_mask"].bool()], dim=1
        )
        if tokens.shape[1] != padding_mask.shape[1]:
            raise AssertionError("SurvTRACE token and padding axes differ.")
        if tokens.shape[1] > self.total_tokens:
            raise AssertionError("SurvTRACE tokenizer exceeded the feature cap.")
        tokens = self.input_dropout(self.input_norm(tokens))
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return tokens, padding_mask

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw, padding_mask = self.tokenize(batch)
        encoded = raw
        for block in self.blocks:
            encoded = block(encoded, key_padding_mask=padding_mask)
        encoded = self.output_norm(encoded)
        encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        valid = (~padding_mask).to(encoded.dtype).unsqueeze(-1)
        denominator = valid.sum(dim=1).clamp_min(1.0)
        encoded_mean = (encoded * valid).sum(dim=1) / denominator
        raw_mean = (raw * valid).sum(dim=1) / denominator
        shared = self.shared_projection(
            torch.cat([encoded_mean, raw_mean], dim=-1)
        )
        condition = self.condition_fusion(
            torch.cat(
                [encoded[:, :K], shared[:, None, :].expand(-1, K, -1)], dim=-1
            )
        )
        return condition, shared, encoded


class SurvTRACECanonicalCompetingRiskModel(nn.Module):
    """SurvTRACE backbone with canonical multi-condition/death intensity heads."""

    def __init__(self):
        super().__init__()
        d = CFG.d_model
        self.encoder = SurvTRACEFeatureEncoder()
        self.condition_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(d, N_BINS),
        )
        self.death_head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(d, N_BINS),
        )
        self.condition_log_rate_intercept = nn.Parameter(
            torch.as_tensor(
                TRAIN_CONDITION_INITIAL_LOG_RATE, dtype=torch.float32
            ).clone()
        )
        self.death_log_rate_intercept = nn.Parameter(
            torch.tensor(float(TRAIN_DEATH_INITIAL_LOG_RATE), dtype=torch.float32)
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        condition_representation, shared, encoded = self.encoder(batch)
        condition_residual = self.condition_head(condition_representation)
        death_residual = self.death_head(shared)
        lower = LOG_RATE_MIN
        upper = LOG_RATE_MAX
        condition_log_rate = (
            self.condition_log_rate_intercept[None, :, None]
            + condition_residual
        ).clamp(lower, upper)
        death_log_rate = (
            self.death_log_rate_intercept + death_residual
        ).clamp(lower, upper)
        condition_rate = torch.exp(condition_log_rate)
        death_rate = torch.exp(death_log_rate)
        return competing_risk_outputs(condition_rate, death_rate)


def initialize_from_scratch(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.Embedding):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02)
            if layer.padding_idx is not None:
                with torch.no_grad():
                    layer.weight[layer.padding_idx].zero_()
        elif isinstance(layer, nn.MultiheadAttention):
            if layer.in_proj_weight is not None:
                nn.init.xavier_uniform_(layer.in_proj_weight)
            if layer.in_proj_bias is not None:
                nn.init.zeros_(layer.in_proj_bias)
    if isinstance(module, SurvTRACECanonicalCompetingRiskModel):
        for head in (module.condition_head, module.death_head):
            final = head[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(final.bias)
        with torch.no_grad():
            module.condition_log_rate_intercept.copy_(
                torch.as_tensor(
                    TRAIN_CONDITION_INITIAL_LOG_RATE,
                    dtype=module.condition_log_rate_intercept.dtype,
                )
            )
            module.death_log_rate_intercept.fill_(
                float(TRAIN_DEATH_INITIAL_LOG_RATE)
            )

model = SurvTRACECanonicalCompetingRiskModel().to(DEVICE)
initialize_from_scratch(model)
origin = {"mode":"scratch", "feature_scalers":"training split only", "event_limit":FULL_SEQUENCE_MAX_EVENTS}
model_meta = {"architecture": ARCHITECTURE_SOURCE, "backbone": vars(CFG), "parameters":sum(p.numel() for p in model.parameters()),
              "initialization":origin, "head_bins":haz.N_BINS}
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
    batch = feature_collate(rows, training=False)
    keys = ("condition_summary", "global_summary", "event_feature_ids", "event_feature_type_ids", "event_feature_values", "event_feature_mask", "index_age_months", "gender_idx", "ethnicity_idx", "townsend_q", "continuous_static")
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
