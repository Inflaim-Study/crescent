# Delphi-2M-inspired — Causal Transformer and next-event / waiting-time likelihood
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 16 ---
class DelphiLayerNorm(nn.Module):
    def __init__(self, dimension: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension)) if bias else None

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            value, self.weight.shape, self.weight, self.bias, 1e-5
        )


def delphi_gelu(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * value * (
        1.0
        + torch.tanh(
            math.sqrt(2.0 / math.pi)
            * (value + 0.044715 * torch.pow(value, 3.0))
        )
    )


class DelphiAgeEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        divisor = torch.exp(
            torch.arange(0, CFG.d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / CFG.d_model)
        )
        self.register_buffer("divisor", divisor)
        self.linear = nn.Linear(CFG.d_model, CFG.d_model, bias=False)

    def forward(self, age_days: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=age_days.device.type, enabled=False):
            angle = age_days.float().unsqueeze(-1) / DAYS_PER_YEAR * self.divisor.float()
            encoded = torch.zeros(
                *age_days.shape,
                CFG.d_model,
                device=age_days.device,
                dtype=torch.float32,
            )
            encoded[..., 0::2] = torch.sin(angle)
            encoded[..., 1::2] = torch.cos(angle)
            return F.linear(encoded, self.linear.weight.float())


def build_delphi_attention_mask(
    input_age_days: torch.Tensor,
    attention_mask: torch.Tensor,
    target_age_days: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Official-style causal mask, additionally blocking target-time ties."""
    batch_size, length = input_age_days.shape
    device = input_age_days.device
    causal = torch.tril(
        torch.ones(length, length, device=device, dtype=torch.bool)
    ).view(1, 1, length, length)
    real = attention_mask.bool()
    allowed = (
        causal
        & real[:, None, :, None]
        & real[:, None, None, :]
    )
    if target_age_days is not None and CFG.mask_ties:
        key_age = input_age_days[:, None, None, :]
        target_age = target_age_days[:, None, :, None]
        allowed = allowed & (key_age != target_age)

    # Delphi restores the diagonal when masking leaves a row without a key.
    empty = ~allowed.any(dim=-1, keepdim=True)
    diagonal = torch.eye(
        length, device=device, dtype=torch.bool
    ).view(1, 1, length, length)
    allowed = allowed | (empty & diagonal)
    return allowed


class DelphiCausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_heads = CFG.n_heads
        self.head_dim = CFG.d_model // CFG.n_heads
        self.c_attn = nn.Linear(
            CFG.d_model, 3 * CFG.d_model, bias=CFG.bias
        )
        self.c_proj = nn.Linear(
            CFG.d_model, CFG.d_model, bias=CFG.bias
        )
        self.attention_dropout = nn.Dropout(CFG.dropout)
        self.residual_dropout = nn.Dropout(CFG.dropout)

    def forward(
        self,
        value: torch.Tensor,
        allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, dimension = value.shape
        query, key, content = self.c_attn(value).split(
            CFG.d_model, dim=-1
        )
        query = query.view(
            batch_size, length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        content = content.view(
            batch_size, length, self.n_heads, self.head_dim
        ).transpose(1, 2)
        score = (
            query @ key.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)
        score = score.masked_fill(~allowed, float("-inf"))
        attention = torch.softmax(score.float(), dim=-1).to(value.dtype)
        attention = torch.nan_to_num(attention, nan=0.0)
        attention = self.attention_dropout(attention)
        output = attention @ content
        output = output.transpose(1, 2).contiguous().view(
            batch_size, length, dimension
        )
        return self.residual_dropout(self.c_proj(output)), attention


class DelphiMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(CFG.d_model, CFG.d_ff, bias=CFG.bias)
        self.c_proj = nn.Linear(CFG.d_ff, CFG.d_model, bias=CFG.bias)
        self.dropout = nn.Dropout(CFG.dropout)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(delphi_gelu(self.c_fc(value))))


class DelphiBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln_1 = DelphiLayerNorm(CFG.d_model, CFG.bias)
        self.attention = DelphiCausalSelfAttention()
        self.ln_2 = DelphiLayerNorm(CFG.d_model, CFG.bias)
        self.mlp = DelphiMLP()

    def forward(
        self,
        value: torch.Tensor,
        allowed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        update, attention = self.attention(self.ln_1(value), allowed)
        value = value + update
        value = value + self.mlp(self.ln_2(value))
        return value, attention


class DelphiStaticContext(nn.Module):
    """Birth-anchored demographic context (sex, ethnicity, Townsend; no index-age leak)."""
    def __init__(self):
        super().__init__()
        self.gender = nn.Embedding(N_GENDER, CFG.d_model)
        self.ethnicity = nn.Embedding(N_ETHNICITY, CFG.d_model)
        self.townsend = nn.Embedding(N_TOWNSEND, CFG.d_model)
        self.continuous = nn.Linear(
            N_CONTINUOUS_STATIC, CFG.d_model, bias=CFG.bias
        )
        self.norm = DelphiLayerNorm(CFG.d_model, CFG.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        cont = torch.nan_to_num(
            batch["continuous_static"].float(), nan=0.0, posinf=0.0, neginf=0.0
        ).clone()
        # Column 0 is standardized index_age_years; zero it out at birth (age 0.0)
        # so pre-index steps cannot observe future survival to recruitment age.
        # Exact continuous age is encoded at every token by DelphiAgeEncoding.
        cont[:, 0] = 0.0
        context = (
            self.gender(batch["gender_idx"])
            + self.ethnicity(batch["ethnicity_idx"])
            + self.townsend(batch["townsend_q"])
            + self.continuous(cont)
        )
        return self.norm(context)


class DelphiCanonicalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(
            DELPHI_VOCAB_SIZE,
            CFG.d_model,
        )
        self.event_value_direction = nn.Embedding(
            DELPHI_VOCAB_SIZE,
            CFG.d_model,
        )
        self.age_encoding = DelphiAgeEncoding()
        self.static_context = DelphiStaticContext()
        self.input_norm = (
            DelphiLayerNorm(CFG.d_model, CFG.bias)
            if CFG.use_input_norm
            else nn.Identity()
        )
        self.token_dropout = nn.Dropout(CFG.token_dropout)
        self.dropout = nn.Dropout(CFG.dropout)
        self.blocks = nn.ModuleList(
            [DelphiBlock() for _ in range(CFG.n_layers)]
        )
        self.final_norm = DelphiLayerNorm(CFG.d_model, CFG.bias)
        self.output_head = nn.Linear(
            CFG.d_model, DELPHI_VOCAB_SIZE, bias=False
        )
        # Published Delphi weight tying.
        self.output_head.weight = self.token_embedding.weight
        self.output_bias = (
            nn.Parameter(torch.zeros(DELPHI_VOCAB_SIZE))
            if CFG.use_output_bias
            else None
        )
        self.apply(self._initialize)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    parameter,
                    mean=0.0,
                    std=0.02 / math.sqrt(2 * CFG.n_layers),
                )

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        training_targets: bool = False,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        target_age = (
            batch["target_age_days"] if training_targets else None
        )
        allowed = build_delphi_attention_mask(
            batch["input_age_days"],
            batch["attention_mask"],
            target_age,
        )
        token_ids = batch["token_ids"]
        event_values = torch.nan_to_num(
            batch["event_values"].float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        event_value_masks = torch.nan_to_num(
            batch["event_value_masks"].float(), nan=0.0, posinf=0.0, neginf=0.0
        )
        # log1p keeps heavy-tailed mg/L CRP values commensurate with O(0.02) embeddings.
        scaled_values = torch.log1p(event_values.clamp_min(0.0)) * event_value_masks
        numeric_value = scaled_values.unsqueeze(-1) * self.event_value_direction(token_ids)
        value = self.token_dropout(
            self.token_embedding(token_ids) + numeric_value
        )
        static_locations = (
            token_ids == DELPHI_STATIC_ID
        ).unsqueeze(-1)
        value = (
            value
            + self.age_encoding(batch["input_age_days"]).to(value.dtype)
            + static_locations.to(value.dtype)
            * self.static_context(batch).unsqueeze(1)
        )
        value = self.dropout(self.input_norm(value))
        final_attention = None
        for block in self.blocks:
            value, final_attention = block(value, allowed)
        # Compute final norm and tied output projection in fp32 to prevent bf16
        # 0.0625 logit quantization (~6.5% rate step error) at rare-disease rates.
        with torch.autocast(device_type=value.device.type, enabled=False):
            normed = F.layer_norm(
                value.float(),
                self.final_norm.weight.shape,
                self.final_norm.weight.float(),
                None if self.final_norm.bias is None else self.final_norm.bias.float(),
                1e-5,
            )
            logits = F.linear(
                normed,
                self.output_head.weight.float(),
                None if self.output_bias is None else self.output_bias.float(),
            )
        result = {"logits": logits, "attention_allowed": allowed}
        if return_attention and final_attention is not None:
            result["final_attention"] = final_attention
        return result

    def count_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters()))

# --- Notebook cell 18 ---
def delphi_likelihood_terms(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    exclude_no_event_targets: bool = False,
    include_censor_survival: bool = False,
) -> Dict[str, torch.Tensor]:
    if include_censor_survival:
        raise ValueError(
            "The Delphi AR comparator does not include a separate terminal "
            "administrative-censor survival term."
        )
    with torch.autocast(device_type=outputs["logits"].device.type, enabled=False):
        logits = outputs["logits"].float()
        allowed_output_array = ALLOWED_OUTPUT_MASK.copy()
        if exclude_no_event_targets:
            # Remove NO_EVENT from both the token softmax and total intensity.
            allowed_output_array[DELPHI_NO_EVENT_ID] = False
        allowed_outputs = torch.as_tensor(
            allowed_output_array, device=logits.device, dtype=torch.bool
        )
        position_allowed = allowed_outputs.view(1, 1, -1).expand_as(logits).clone()

        is_post_index = batch["target_age_days"] > (
            batch["index_age_days"].unsqueeze(1) + 1e-3
        )
        # UK Biobank participants survived to index_age by enrolment construction;
        # mask DELPHI_DEATH_ID prior to index_age to avoid immortal-time bias.
        pre_index_no_death = position_allowed.clone()
        pre_index_no_death[..., DELPHI_DEATH_ID] = False
        position_allowed = torch.where(
            is_post_index.unsqueeze(-1),
            position_allowed,
            pre_index_no_death,
        )

        if CFG.post_index_target_output_restriction and "post_index_output_mask" in batch:
            post_allowed = batch["post_index_output_mask"].bool() & allowed_outputs.view(1, -1)
            position_allowed = torch.where(
                is_post_index.unsqueeze(-1),
                post_allowed.unsqueeze(1),
                position_allowed,
            )

        if CFG.dynamic_first_event_risk_sets:
            real_keys = batch["attention_mask"].bool()
            seq_len = logits.shape[1]
            causal_pairs = torch.tril(
                torch.ones(seq_len, seq_len, device=logits.device, dtype=torch.bool)
            ).unsqueeze(0)
            strictly_prior = (
                causal_pairs
                & real_keys.unsqueeze(1)
                & (
                    batch["input_age_days"].unsqueeze(1)
                    < (batch["target_age_days"].unsqueeze(2) - 1e-4)
                )
            )
            token_one_hot = F.one_hot(
                batch["token_ids"].clamp(0, DELPHI_VOCAB_SIZE - 1),
                num_classes=DELPHI_VOCAB_SIZE,
            ).to(dtype=torch.float32)
            occurred_prior = (strictly_prior.to(dtype=torch.float32) @ token_one_hot) > 0.5
            first_occ_mask = torch.as_tensor(
                FIRST_OCCURRENCE_OUTPUT_MASK, device=logits.device, dtype=torch.bool
            ).view(1, 1, -1)
            position_allowed = position_allowed & ~(occurred_prior & first_occ_mask)

        has_any_allowed = position_allowed.any(dim=-1, keepdim=True)
        position_allowed = torch.where(
            has_any_allowed, position_allowed, allowed_outputs.view(1, 1, -1)
        )

        masked_logits = logits.masked_fill(~position_allowed, float("-inf"))
        targets = batch["target_ids"]
        safe_targets = targets.clamp(0, DELPHI_VOCAB_SIZE - 1)
        target_allowed = position_allowed.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)

        valid_step = (
            batch["attention_mask"].bool()
            & (targets >= 0)
            & (targets != DELPHI_PAD_ID)
        )
        event_valid = valid_step & target_allowed
        censor_valid = torch.zeros_like(batch["attention_mask"], dtype=torch.bool)
        zero = logits.sum() * 0.0

        if bool(event_valid.any()):
            selected_logits = masked_logits[event_valid]
            selected_targets = targets[event_valid]
            token_numerator = F.cross_entropy(
                selected_logits,
                selected_targets,
                reduction="sum",
            )
        else:
            token_numerator = zero

        log_total_rate = torch.logsumexp(masked_logits, dim=-1)
        # Published Delphi stabilizer.
        stable_log_total_rate = -torch.log(
            torch.exp(-log_total_rate) + CFG.t_min_days
        )
        total_rate_per_day = torch.exp(stable_log_total_rate)

        # 1) Event-valid transitions: use Delphi's effective_age (>= 1.0 day) so
        #    same-date tied events NEVER receive delta_days == 0 (which has no finite
        #    minimum under the t_min stabilizer and drives logits to +inf).
        if CFG.mask_ties:
            allowed_attention = outputs["attention_allowed"].squeeze(1)
            positions = torch.arange(
                allowed_attention.shape[-1],
                device=logits.device,
                dtype=torch.long,
            ).view(1, 1, -1)
            previous = torch.where(
                allowed_attention,
                positions,
                torch.zeros_like(positions),
            ).amax(dim=-1)
            effective_age = torch.gather(batch["input_age_days"], 1, previous)
        else:
            effective_age = batch["input_age_days"]

        delta_days = (batch["target_age_days"] - effective_age).clamp_min(1.0)
        event_time_losses = -(
            stable_log_total_rate
            - total_rate_per_day * (delta_days + CFG.t_min_days)
        )
        occurrence_sum = torch.where(
            event_valid, event_time_losses, torch.zeros_like(event_time_losses)
        ).sum()

        # 2) Non-event transitions (input-only CRP/lifestyle tokens during training;
        #    input-only + excluded NO_EVENT tokens during validation): retain disjoint
        #    chronological survival exposure (+Lambda * step_days) while suppressing
        #    timestamps already covered by a same-date event_valid transition.
        step_days = (batch["target_age_days"] - batch["input_age_days"]).clamp_min(0.0)
        if CFG.mask_ties:
            same_target_time = (
                torch.abs(
                    batch["target_age_days"].unsqueeze(2)
                    - batch["target_age_days"].unsqueeze(1)
                )
                <= 1e-4
            )
            covered_by_event = (
                same_target_time & event_valid.unsqueeze(1)
            ).any(dim=-1)
        else:
            covered_by_event = torch.zeros_like(valid_step)

        surv_only_valid = (
            valid_step
            & (~event_valid)
            & (step_days > 0.0)
            & (~covered_by_event)
        )
        exposure_losses = total_rate_per_day * step_days
        exposure_sum = torch.where(
            surv_only_valid, exposure_losses, torch.zeros_like(exposure_losses)
        ).sum()
        time_numerator = occurrence_sum + exposure_sum
        censor_numerator = zero

        event_count = event_valid.float().sum()
        censor_interval_count = censor_valid.float().sum()
        contribution_count = event_count + censor_interval_count
        numerator = token_numerator + time_numerator + censor_numerator
        return {
            "objective": numerator / contribution_count.clamp_min(1.0),
            "nll_numerator": numerator,
            "token_nll_numerator": token_numerator,
            "time_nll_numerator": time_numerator,
            "censor_nll_numerator": censor_numerator,
            "token_count": event_count,
            "censor_interval_count": censor_interval_count,
            "contribution_count": contribution_count,
            "no_event_target_count": (
                event_valid & (targets == DELPHI_NO_EVENT_ID)
            ).float().sum(),
            "disease_death_target_count": (
                event_valid & (targets != DELPHI_NO_EVENT_ID)
            ).float().sum(),
        }


def delphi_monthly_rates_from_logits(
    logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exact prediction-time exp(logit) rate transform, events per month."""
    logits = logits.float()
    maximum_log = math.log(torch.finfo(torch.float32).max) - 2.0
    condition_index = torch.as_tensor(
        DELPHI_CONDITION_IDS, device=logits.device, dtype=torch.long
    )
    condition_rate = (
        torch.exp(logits[:, condition_index].clamp(max=maximum_log))
        * DAYS_PER_MONTH
    )
    death_rate = (
        torch.exp(logits[:, DELPHI_DEATH_ID].clamp(max=maximum_log))
        * DAYS_PER_MONTH
    )
    return condition_rate, death_rate


def model_architecture_manifest(model: nn.Module) -> Dict[str, Any]:
    return {
        "model_family": "Delphi-2M original-loss canonical-index comparator (fixed v3, tie-safe + EMA)",
        "implementation_version": "A10 original-loss 50k full-exposure first-occurrence baseline",
        "published_reference": "Nature 2025; DOI 10.1038/s41586-025-09529-3",
        "encoder": "GPT-style causal transformer with absolute-age encoding",
        "layers": CFG.n_layers,
        "attention_heads": CFG.n_heads,
        "embedding_dimension": CFG.d_model,
        "context_length": CFG.block_size,
        "embedding_output_weight_tying": (
            model.output_head.weight is model.token_embedding.weight
        ),
        "same_time_masking": CFG.mask_ties,
        "transformer_bias": CFG.bias,
        "canonical_event_vocab_size": EVENT_VOCAB_SIZE,
        "autoregressive_vocab_size": DELPHI_VOCAB_SIZE,
        "static_context_injection": (
            f"adult-origin ({CFG.origin_age_years:.1f}y) DELPHI_STATIC_ID (excluding index_age_years)"
        ),
        "input_norm": bool(CFG.use_input_norm),
        "output_bias": bool(CFG.use_output_bias),
        "fp32_age_and_output_head": True,
        "ema_decay": float(CFG.ema_decay),
        "numeric_event_value_injection": (
            "log1p(prepared canonical value) times mask along a token-specific direction"
        ),
        "clinical_history_max_events": MAIN_CLINICAL_HISTORY_MAX_EVENTS,
        "artificial_landmark_removed": True,
        "terminal_censor_survival": False,
        "terminal_boundary_target": "native NO_EVENT token/time likelihood",
        "exogenous_interval_survival_retained": True,
        "pre_index_death_masked_for_left_truncation": True,
        "post_index_output_restriction": bool(CFG.post_index_target_output_restriction),
        "dynamic_first_event_risk_sets": bool(CFG.dynamic_first_event_risk_sets),
        "joint_likelihood_weight": 1.0,
        "t_min_days": CFG.t_min_days,
        "primary_rate_transform": "exp(logit), events per day",
        "input_contract": "supplied A10 canonical shared samples",
        "time_origin": CANONICAL_INDEX_DESIGN,
        "checkpoint_selection": (
            "minimum original Delphi validation next-event plus waiting-time "
            "loss; four-horizon discrimination is diagnostic only"
        ),
        "test_used_for_selection": False,
        "landmark_inference": "pre-index NO_EVENT probes plus single target-age NO_EVENT query",
        "cif": "analytic piecewise-constant condition-versus-death cumulative incidence",
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameter_count": int(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        ),
    }
