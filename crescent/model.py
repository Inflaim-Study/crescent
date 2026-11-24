"""Stage 3: CRESCENT model architecture.

Extracted from the CRESCENT_Final notebook (cell 10). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 10
# =============================================================================

import os
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

assert "cfg" in globals() and "haz" in globals() and "K" in globals(), "Run setup Cells 1/2 first."
assert "train_loader" in globals() and "train_samples" in globals(), "Run canonical loader Cell 2 and dataloader Cell 4.5 first."
assert "EV_VOCAB_SIZE_MODEL" in globals() and "TYPE_VOCAB_SIZE_MODEL" in globals(), "Run canonical loader Cell 2 first."
assert "EV_PAD_ID" in globals() and "TYPE_PAD_ID" in globals(), "Run canonical loader Cell 2 first."
assert "N_GENDER_MODEL" in globals() and "N_ETHNICITY_MODEL" in globals() and "N_TOWNSEND_MODEL" in globals(), "Run canonical loader Cell 2 first."
assert "N_CONTINUOUS_STATIC" in globals(), "Run canonical loader Cell 2 first."
assert "RUN_DIR" in globals(), "Run Cell 1 first."

RUN_DIR = Path(RUN_DIR)
RUN_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VARIANT = str(cfg.MODEL_VARIANT)
ARCHITECTURE_SOURCE = "CRESCENT_one_month_cap_6month_delayed_entry_nll_shared_state_crp_pwl"
DEVICE = torch.device(getattr(cfg, "DEVICE", "cuda:0"))
if str(DEVICE) != "cuda:0" or torch.cuda.current_device() != 0:
    raise RuntimeError(
        f"Model cell requires cuda:0, resolved DEVICE={DEVICE}, "
        f"current_cuda_device={torch.cuda.current_device()}."
    )

D_MODEL = int(getattr(cfg, "D_MODEL", 128))
D_FF = int(getattr(cfg, "D_FF", 512))
N_HEADS = int(getattr(cfg, "N_HEADS", 16))
N_LAYERS = int(cfg.N_LAYERS)

D_STATIC = int(getattr(cfg, "D_STATIC", 32))
DROPOUT_ENCODER = float(getattr(cfg, "DROPOUT_ENCODER", 0.10))
DROPOUT_HAZARD = float(getattr(cfg, "DROPOUT_HAZARD", 0.10))
DROPOUT_DEATH = float(getattr(cfg, "DROPOUT_DEATH", 0.10))
HAZARD_HEAD_HIDDEN_DIM = int(getattr(cfg, "HAZARD_HEAD_HIDDEN_DIM", 128))
HAZARD_HEAD_LAYERS = int(getattr(cfg, "HAZARD_HEAD_LAYERS", 2))
USE_SHARED_STATE_HAZARD_BRANCH = bool(
    getattr(cfg, "USE_SHARED_STATE_HAZARD_BRANCH", True)
)
SHARED_STATE_HEAD_HIDDEN_DIM = int(
    getattr(cfg, "SHARED_STATE_HEAD_HIDDEN_DIM", HAZARD_HEAD_HIDDEN_DIM)
)
SHARED_STATE_HEAD_LAYERS = int(
    getattr(cfg, "SHARED_STATE_HEAD_LAYERS", HAZARD_HEAD_LAYERS)
)
SHARED_STATE_HEAD_ZERO_INIT = bool(
    getattr(cfg, "SHARED_STATE_HEAD_ZERO_INIT", True)
)
DEATH_HEAD_HIDDEN_DIM = int(getattr(cfg, "DEATH_HEAD_HIDDEN_DIM", 128))
DEATH_HEAD_LAYERS = int(getattr(cfg, "DEATH_HEAD_LAYERS", 2))
RESIDUAL_INIT_STD = float(getattr(cfg, "RESIDUAL_INIT_STD", 1e-3))

FINAL_USE_CONDITION_INTERACTION = True
FINAL_CONDITION_INTERACTION_LAYERS = 1
FINAL_CONDITION_INTERACTION_HEADS = int(N_HEADS)
FINAL_USE_RELATIVE_TIME_ATTENTION_BIAS = False
FINAL_USE_READER_TEMPORAL_BIAS = bool(getattr(cfg, "USE_READER_TEMPORAL_BIAS", True))
STATIC_CONT_INPUT_LAYERNORM = bool(getattr(cfg, "STATIC_CONT_INPUT_LAYERNORM", False))
USE_ATTAINED_AGE_TERM = bool(getattr(cfg, "USE_ATTAINED_AGE_TERM", True))
FINAL_READER_TEMPORAL_BIAS_N_BASIS = 16
FINAL_READER_TEMPORAL_BIAS_MAX_ABS = 6.0
FINAL_READER_TEMPORAL_BIAS_MODE = str(
    getattr(cfg, "READER_TEMPORAL_BIAS_MODE", "interaction_mlp")
).strip().lower()
FINAL_READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER = int(
    getattr(cfg, "READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER", 4)
)
FINAL_READER_TEMPORAL_INTERACTION_DROPOUT = float(
    getattr(cfg, "READER_TEMPORAL_INTERACTION_DROPOUT", 0.10)
)
FINAL_USE_DEATH_SEQUENCE_ATTN_POOL = False

N_TIME_BINS = int(haz.N_BINS)
VOCAB_SIZE = int(EV_VOCAB_SIZE_MODEL)
TYPE_VOCAB_SIZE = int(TYPE_VOCAB_SIZE_MODEL)
N_GENDER = int(N_GENDER_MODEL)
N_ETHNICITY = int(N_ETHNICITY_MODEL)
N_TOWNSEND = int(N_TOWNSEND_MODEL)
N_CONT_STATIC = int(N_CONTINUOUS_STATIC)
PROB_EPS = float(getattr(cfg, "PROB_EPS", 1e-7))
RATE_EPS = float(getattr(cfg, "RATE_EPS", 1e-8))
LOG_RATE_MIN = float(getattr(cfg, "LOG_RATE_MIN", -18.0))
LOG_RATE_MAX = float(getattr(cfg, "LOG_RATE_MAX", 2.0))


if D_MODEL % N_HEADS != 0:
    raise ValueError("D_MODEL must be divisible by N_HEADS.")
if not USE_SHARED_STATE_HAZARD_BRANCH:
    raise ValueError(
        "This notebook is the shared-state architecture variant and requires "
        "USE_SHARED_STATE_HAZARD_BRANCH=True."
    )
if not SHARED_STATE_HEAD_ZERO_INIT:
    raise ValueError(
        "The locked shared-state experiment requires an exactly zero-initialised "
        "shared-state output projection."
    )

def make_mlp(
    d_in,
    d_hidden,
    d_out,
    n_layers=2,
    dropout=0.10,
    final_init_std=None,
):
    layers = []

    d_prev = int(d_in)

    for _ in range(max(1, int(n_layers) - 1)):
        layers.extend(
            [
                nn.LayerNorm(d_prev),
                nn.Linear(d_prev, int(d_hidden)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
            ]
        )
        d_prev = int(d_hidden)

    layers.extend(
        [
            nn.LayerNorm(d_prev),
            nn.Linear(d_prev, int(d_out)),
        ]
    )

    net = nn.Sequential(*layers)

    if final_init_std is not None:
        final_linear = None

        for m in reversed(net):
            if isinstance(m, nn.Linear):
                final_linear = m
                break

        if final_linear is not None:
            nn.init.normal_(final_linear.weight, mean=0.0, std=float(final_init_std))
            nn.init.zeros_(final_linear.bias)

    return net


class ContinuousTimeEncoding(nn.Module):
    """
    Sinusoidal absolute-age encoding from event_times_months.

    Input:
      event_times_months: (B, L)

    Output:
      time_emb: (B, L, D)
    """

    def __init__(self, d_model, base=10000.0):
        super().__init__()

        self.d_model = int(d_model)
        self.base = float(base)

        if self.d_model % 2 != 0:
            raise ValueError("D_MODEL must be even for sinusoidal time encoding.")

    def forward(self, event_times_months):
        device = event_times_months.device
        dtype = event_times_months.dtype

        t_years = event_times_months.float() / 12.0

        half = self.d_model // 2

        div = torch.exp(
            torch.arange(0, half, device=device, dtype=torch.float32)
            * (-math.log(self.base) / max(half - 1, 1))
        )

        angles = t_years.unsqueeze(-1) * div.view(1, 1, -1)

        emb = torch.cat(
            [
                torch.sin(angles),
                torch.cos(angles),
            ],
            dim=-1,
        )

        return emb.to(dtype=dtype)


class RelTimeBias(nn.Module):
    """
    Learned relative-time attention bias for the event transformer.

    Input:
      event_times_months: (B, L)

    Output:
      additive attention mask in PyTorch MultiheadAttention format: (B * H, L, L)

    The final linear layer is zero-initialised so the initial bias is exactly
    zero and the encoder starts equivalent to the previous implementation.
    """

    def __init__(self, n_heads, n_basis=16):
        super().__init__()

        self.n_heads = int(n_heads)

        self.mlp = nn.Sequential(
            nn.Linear(1, int(n_basis)),
            nn.GELU(),
            nn.Linear(int(n_basis), self.n_heads),
        )

        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, event_times_months):
        t = torch.nan_to_num(
            event_times_months.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        dt = torch.abs(t[:, :, None] - t[:, None, :])
        feat = torch.log1p(dt).unsqueeze(-1)

        bias = self.mlp(feat).permute(0, 3, 1, 2).contiguous()
        B, H, L, _ = bias.shape

        return bias.view(B * H, L, L)


class ReaderTemporalBias(nn.Module):
    """
    Target-aware temporal kernel for the condition-specific history reader.

    It produces an additive cross-attention mask for:
      condition queries: (B, K, D)
      event keys/values: (B, L, D)

    Output shape follows PyTorch MultiheadAttention convention:
      (B * H, K, L)

    The final projection is zero-initialised, so the initial model is exactly
    equivalent to the existing condition reader.
    """

    def __init__(
        self,
        n_conditions,
        vocab_size,
        type_vocab_size,
        n_heads,
        n_basis=16,
        max_abs_bias=6.0,
        mode="interaction_mlp",
        interaction_hidden_multiplier=4,
        interaction_dropout=0.10,
    ):
        super().__init__()

        self.K = int(n_conditions)
        self.V = int(vocab_size)
        self.T = int(type_vocab_size)
        self.H = int(n_heads)
        self.R = int(n_basis)
        self.max_abs_bias = float(max_abs_bias)
        self.mode = str(mode).strip().lower()
        if self.mode not in {"additive", "interaction_mlp"}:
            raise ValueError(
                "ReaderTemporalBias mode must be 'additive' or 'interaction_mlp'."
            )

        self.cond_emb = nn.Embedding(self.K, self.R)
        self.event_emb = nn.Embedding(self.V, self.R)
        self.type_emb = nn.Embedding(self.T, self.R)

        self.dt_mlp = nn.Sequential(
            nn.Linear(1, self.R),
            nn.GELU(),
            nn.Linear(self.R, self.R),
        )

        if self.mode == "additive":
            self.out = nn.Linear(4 * self.R, self.H)
            final_projection = self.out
        else:
            hidden = int(interaction_hidden_multiplier) * self.R
            self.out = nn.Sequential(
                nn.LayerNorm(6 * self.R),
                nn.Linear(6 * self.R, hidden),
                nn.GELU(),
                nn.Dropout(float(interaction_dropout)),
                nn.Linear(hidden, self.H),
            )
            final_projection = self.out[-1]

        # Preserve the original zero-bias starting point for a controlled
        # comparison; the preceding interaction layer still receives gradients
        # after the final projection moves away from zero.
        nn.init.zeros_(final_projection.weight)
        nn.init.zeros_(final_projection.bias)

    def forward(
        self,
        cond_ids,
        event_ids,
        event_type_ids,
        event_times_months,
        index_age_months,
    ):
        B, L = event_times_months.shape
        K = int(cond_ids.numel())

        event_ids = event_ids.long().clamp(0, self.V - 1)
        event_type_ids = event_type_ids.long().clamp(0, self.T - 1)

        t = torch.nan_to_num(
            event_times_months.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        index_t = torch.nan_to_num(
            index_age_months.float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).view(B, 1)

        # Look-back time from prediction/index age to each historical event.
        # Future/malformed values are clipped to zero rather than producing
        # negative recency features.
        dt = torch.clamp(index_t - t, min=0.0)
        dt_feat = torch.log1p(dt / 12.0).unsqueeze(-1)

        c = self.cond_emb(cond_ids.long().clamp(0, self.K - 1))      # (K, R)
        e = self.event_emb(event_ids)                                # (B, L, R)
        ty = self.type_emb(event_type_ids)                           # (B, L, R)
        d = self.dt_mlp(dt_feat)                                     # (B, L, R)

        c = c.view(1, K, 1, self.R).expand(B, K, L, self.R)
        e = e.unsqueeze(1).expand(B, K, L, self.R)
        ty = ty.unsqueeze(1).expand(B, K, L, self.R)
        d = d.unsqueeze(1).expand(B, K, L, self.R)

        if self.mode == "additive":
            x = torch.cat([c, e, ty, d], dim=-1)                     # (B, K, L, 4R)
        else:
            event_context = e + ty
            x = torch.cat(
                [
                    c,
                    event_context,
                    d,
                    c * event_context,  # disease x event/type
                    c * d,              # disease x recency
                    event_context * d,  # event/type x recency
                ],
                dim=-1,
            )                                                        # (B, K, L, 6R)
        bias = self.out(x)                                           # (B, K, L, H)

        if self.max_abs_bias > 0.0:
            bias = self.max_abs_bias * torch.tanh(bias / self.max_abs_bias)

        bias = bias.permute(0, 3, 1, 2).contiguous()                 # (B, H, K, L)
        return bias.view(B * self.H, K, L)


class StaticCovariateEncoder(nn.Module):
    """
    Encodes categorical static covariates plus continuous static covariates.

    Output:
      static_vec: (B, D_MODEL)
    """

    def __init__(
        self,
        n_gender,
        n_ethnicity,
        n_townsend,
        n_continuous_static,
        d_model,
        d_static=32,
        dropout=0.10,
    ):
        super().__init__()

        self.n_continuous_static = int(n_continuous_static)
        self.d_static = int(d_static)
        self.d_model = int(d_model)

        self.gender_emb = nn.Embedding(int(n_gender), self.d_static)
        self.ethnicity_emb = nn.Embedding(int(n_ethnicity), self.d_static)
        self.townsend_emb = nn.Embedding(int(n_townsend), self.d_static)

        if self.n_continuous_static > 0:
            # REVISION: LayerNorm over [age_z, townsend_z, townsend_observed] normalises across three
            # different covariates, so (0,0,1), (-1,-1,1) and (-3,-3,1) all map to the same vector and
            # only one degree of freedom survives. The inputs are already standardised, so the norm is
            # dropped (set CRESCENT_STATIC_CONT_LAYERNORM=1 to restore the original behaviour).
            _cont_layers = []
            if STATIC_CONT_INPUT_LAYERNORM:
                _cont_layers.append(nn.LayerNorm(self.n_continuous_static))
            _cont_layers.extend([
                nn.Linear(self.n_continuous_static, self.d_static),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.d_static, self.d_static),
            ])
            self.cont_net = nn.Sequential(*_cont_layers)
        else:
            self.cont_net = None

        d_in = 4 * self.d_static

        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.d_model, self.d_model),
            nn.LayerNorm(self.d_model),
        )

    def forward(
        self,
        gender_idx,
        ethnicity_idx,
        townsend_q,
        continuous_static=None,
    ):
        B = gender_idx.shape[0]
        device = gender_idx.device

        g = self.gender_emb(gender_idx.long())
        e = self.ethnicity_emb(ethnicity_idx.long())
        t = self.townsend_emb(townsend_q.long())

        if self.n_continuous_static > 0:
            if continuous_static is None:
                continuous_static = torch.zeros(
                    B,
                    self.n_continuous_static,
                    device=device,
                    dtype=torch.float32,
                )

            continuous_static = torch.nan_to_num(
                continuous_static.float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            c = self.cont_net(continuous_static)
        else:
            c = torch.zeros(
                B,
                self.d_static,
                device=device,
                dtype=g.dtype,
            )

        return self.net(
            torch.cat(
                [
                    g,
                    e,
                    t,
                    c,
                ],
                dim=-1,
            )
        )


CRP_KNOT_QUANTILES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def fit_crp_knots(train_dataset):
    """Fit distinct knots to observed CRP values in the selected training histories."""
    observed_values = []
    for row_index, (_, row) in enumerate(train_dataset.samples.iterrows()):
        history = _extract_genuine_prelandmark_history(
            row, train_dataset.split_name, row_index
        )
        sequence = _select_recent_history_with_crp(
            history, train_dataset.full_sequence_max_events
        )
        crp_mask = _crp_event_mask(sequence["event_ids"], sequence["event_type_ids"])
        values = sequence["event_values"]
        observed = crp_mask & (sequence["event_value_masks"] >= 0.5) & np.isfinite(values)
        observed_values.extend(values[observed])

    if not observed_values:
        return np.empty(0, dtype=np.float32)
    return np.unique(np.quantile(observed_values, CRP_KNOT_QUANTILES).astype(np.float32))


class EventSequenceEncoder(nn.Module):
    """
    Event-stream transformer encoder.

    Embedding:
      event token embedding
      event type embedding
      numeric event-value direction embedding
      continuous absolute-age time encoding
      static patient context
    """

    def __init__(
        self,
        vocab_size,
        type_vocab_size,
        d_model,
        n_heads,
        d_ff,
        n_layers,
        dropout,
        pad_id,
        type_pad_id,
        crp_knots,
        use_rel_time_bias=True,
        rel_time_n_basis=16,
    ):
        super().__init__()

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.pad_id = int(pad_id)
        self.type_pad_id = int(type_pad_id)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )

        self.event_emb = nn.Embedding(int(vocab_size), self.d_model, padding_idx=self.pad_id)
        self.type_emb = nn.Embedding(int(type_vocab_size), self.d_model, padding_idx=self.type_pad_id)

        # Event-specific direction for numeric values.
        # For non-valued events, event_value_mask=0 zeros this contribution.
        self.value_direction_emb = nn.Embedding(int(vocab_size), self.d_model, padding_idx=self.pad_id)
        # Zero knot vectors reproduce the linear value encoding at initialisation.
        self.register_buffer("crp_knots", torch.as_tensor(crp_knots, dtype=torch.float32))
        self.register_buffer(
            "crp_token_ids", torch.as_tensor(CRP_TOKEN_IDS, dtype=torch.long), persistent=False
        )
        self.crp_knot_vectors = nn.Parameter(torch.zeros(self.crp_knots.numel(), self.d_model))

        self.time_enc = ContinuousTimeEncoding(self.d_model)

        if bool(use_rel_time_bias):
            self.rel_time_bias = RelTimeBias(
                n_heads=self.n_heads,
                n_basis=int(rel_time_n_basis),
            )
        else:
            self.rel_time_bias = None

        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(float(dropout))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer,
            num_layers=int(n_layers),
        )

        self.out_norm = nn.LayerNorm(self.d_model)

    def crp_value_residual(self, event_ids, event_type_ids, event_values, event_value_masks):
        if self.crp_knots.numel() < 2:
            # There is no observed range to interpolate on.
            return event_values.new_zeros(())

        is_crp = (event_type_ids == CRP_TYPE_ID) | (
            event_ids.unsqueeze(-1) == self.crp_token_ids
        ).any(dim=-1)
        observed = is_crp & (event_value_masks >= 0.5) & torch.isfinite(event_values)
        values = torch.where(observed, event_values, self.crp_knots[0])
        # Use the boundary residual for values outside the training range.
        values = values.clamp(min=self.crp_knots[0], max=self.crp_knots[-1])
        upper = torch.searchsorted(self.crp_knots, values.contiguous(), right=True)
        upper = upper.clamp(1, self.crp_knots.numel() - 1)
        lower = upper - 1
        left = self.crp_knots[lower]
        right = self.crp_knots[upper]
        fraction = (values - left) / (right - left)
        residual = torch.lerp(
            self.crp_knot_vectors[lower], self.crp_knot_vectors[upper], fraction.unsqueeze(-1)
        )
        return torch.where(observed.unsqueeze(-1), residual, 0.0)

    def forward(
        self,
        event_ids,
        event_type_ids,
        event_times_months,
        event_values,
        event_value_masks,
        attn_mask,
        static_vec,
        lengths,
    ):
        B, L = event_ids.shape

        tok = self.event_emb(event_ids.long())
        typ = self.type_emb(event_type_ids.long())

        time = self.time_enc(event_times_months.float())

        val_dir = self.value_direction_emb(event_ids.long())
        val = (
            torch.nan_to_num(event_values.float(), nan=0.0, posinf=0.0, neginf=0.0)
            * torch.nan_to_num(event_value_masks.float(), nan=0.0, posinf=0.0, neginf=0.0)
        ).unsqueeze(-1) * val_dir
        val = val + self.crp_value_residual(
            event_ids, event_type_ids, event_values.float(), event_value_masks
        )

        static = static_vec.unsqueeze(1).expand(B, L, self.d_model)

        x = tok + typ + time + val + static
        x = self.input_dropout(self.input_norm(x))

        # Build ONE additive attention mask that contains both:
        #   1) learned relative-time bias, and
        #   2) key-padding bias.
        #
        # We deliberately do not pass `src_key_padding_mask` separately. Some
        # PyTorch builds warn or behave inconsistently when a 3D float attn mask
        # is combined with a separate key-padding mask. Folding the padding into
        # the same float mask removes the warning and avoids all-masked attention
        # rows producing NaNs.
        pad_bool = attn_mask < 0.5

        # Defensive guard: if a row is entirely padded, MultiheadAttention would
        # otherwise see all keys masked and can return NaNs. Keep position 0
        # unmasked for that row; downstream cls_idx is also clamped to 0.
        all_pad = pad_bool.all(dim=1)
        if bool(all_pad.any()):
            pad_bool = pad_bool.clone()
            pad_bool[all_pad, 0] = False

        pad_bias = torch.zeros(
            B,
            L,
            device=event_ids.device,
            dtype=x.dtype,
        )
        # Use a large finite negative value instead of -inf. This preserves the
        # masking effect while avoiding NaNs in edge cases on some kernels.
        pad_bias = pad_bias.masked_fill(pad_bool, -1.0e4)
        pad_bias = (
            pad_bias[:, None, None, :]
            .expand(B, self.n_heads, L, L)
            .reshape(B * self.n_heads, L, L)
        )

        if self.rel_time_bias is not None:
            rel_bias = self.rel_time_bias(event_times_months).to(dtype=x.dtype)
        else:
            rel_bias = torch.zeros_like(pad_bias)

        encoder_mask = rel_bias + pad_bias
        encoder_mask = torch.nan_to_num(
            encoder_mask,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )

        z = self.encoder(
            x,
            mask=encoder_mask,
            src_key_padding_mask=None,
        )

        z = torch.nan_to_num(
            z,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )
        z = self.out_norm(z)
        z = torch.nan_to_num(
            z,
            nan=0.0,
            posinf=1.0e4,
            neginf=-1.0e4,
        )

        batch_idx = torch.arange(B, device=event_ids.device)
        cls_idx = (lengths.long() - 1).clamp(min=0, max=L - 1)
        cls_vec = z[batch_idx, cls_idx]

        return z, cls_vec


class ConditionHistoryReader(nn.Module):
    """
    Condition-specific history reader.

    Reader query:
      condition embedding + static shift + last-event/CLS shift

    Optional reader-temporal kernel:
      additive cross-attention bias from target condition, source event token/type,
      and log1p(index_age - event_age).
    """

    def __init__(
        self,
        n_conditions,
        d_model,
        n_heads=8,
        dropout=0.10,
        vocab_size=None,
        type_vocab_size=None,
        use_reader_temporal_bias=False,
        reader_temporal_n_basis=16,
        reader_temporal_max_abs_bias=6.0,
        reader_temporal_mode="interaction_mlp",
        reader_temporal_interaction_hidden_multiplier=4,
        reader_temporal_interaction_dropout=0.10,
        use_condition_identity_path=False,
    ):
        super().__init__()

        self.K = int(n_conditions)
        self.D = int(d_model)
        self.n_heads = int(n_heads)
        self.use_reader_temporal_bias = bool(use_reader_temporal_bias)
        self.use_condition_identity_path = bool(use_condition_identity_path)

        self.cond_emb = nn.Embedding(self.K, self.D)

        self.static_shift = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D),
        )

        self.last_shift = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.Linear(self.D, self.D),
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=self.D,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        if self.use_reader_temporal_bias:
            if vocab_size is None or type_vocab_size is None:
                raise ValueError(
                    "vocab_size and type_vocab_size are required when "
                    "use_reader_temporal_bias=True."
                )

            self.reader_temporal_bias = ReaderTemporalBias(
                n_conditions=self.K,
                vocab_size=int(vocab_size),
                type_vocab_size=int(type_vocab_size),
                n_heads=self.n_heads,
                n_basis=int(reader_temporal_n_basis),
                max_abs_bias=float(reader_temporal_max_abs_bias),
                mode=str(reader_temporal_mode),
                interaction_hidden_multiplier=int(
                    reader_temporal_interaction_hidden_multiplier
                ),
                interaction_dropout=float(reader_temporal_interaction_dropout),
            )
        else:
            self.reader_temporal_bias = None

        self.out = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Dropout(float(dropout)),
            nn.Linear(self.D, self.D),
            nn.GELU(),
            nn.LayerNorm(self.D),
        )
        self.query_residual_norm = nn.LayerNorm(self.D)

    def forward(
        self,
        seq_z,
        attn_mask,
        static_vec,
        cls_vec,
        event_ids=None,
        event_type_ids=None,
        event_times_months=None,
        index_age_months=None,
    ):
        B, L = seq_z.shape[:2]
        device = seq_z.device

        cond_ids = torch.arange(self.K, device=device, dtype=torch.long)
        q = self.cond_emb(cond_ids).unsqueeze(0).expand(B, self.K, self.D)

        q = q + self.static_shift(static_vec).unsqueeze(1)
        q = q + self.last_shift(cls_vec).unsqueeze(1)

        key_padding_mask = attn_mask < 0.5

        # Safety guard: MultiheadAttention returns NaNs if every key is masked
        # for a row. This should not happen if the CLS token is present, but the
        # guard makes the cell robust to malformed/empty streams.
        all_masked = key_padding_mask.all(dim=1)
        if bool(all_masked.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False

        reader_bias = None
        key_padding_for_attn = key_padding_mask

        if self.reader_temporal_bias is not None:
            if event_ids is None or event_type_ids is None or event_times_months is None:
                raise ValueError(
                    "event_ids, event_type_ids, and event_times_months are required "
                    "when reader_temporal_bias is enabled."
                )

            if index_age_months is None:
                # Fallback: the collate function appends CLS at the index age;
                # recover that time from the final valid sequence position.
                idx = torch.clamp(attn_mask.long().sum(dim=1) - 1, min=0)
                index_age_months = event_times_months.gather(1, idx.view(-1, 1)).squeeze(1)

            reader_bias = self.reader_temporal_bias(
                cond_ids=cond_ids,
                event_ids=event_ids,
                event_type_ids=event_type_ids,
                event_times_months=event_times_months,
                index_age_months=index_age_months,
            ).to(dtype=seq_z.dtype)

            # Combine padding into the additive mask to avoid mixing bool and
            # float mask types inside MultiheadAttention.
            pad = key_padding_mask.view(B, 1, 1, L).expand(B, self.n_heads, self.K, L)
            pad = pad.reshape(B * self.n_heads, self.K, L)
            reader_bias = reader_bias.masked_fill(pad, -1.0e4)
            key_padding_for_attn = None

        z, _ = self.attn(
            q,
            seq_z,
            seq_z,
            attn_mask=reader_bias,
            key_padding_mask=key_padding_for_attn,
            need_weights=False,
        )

        z = torch.nan_to_num(z, nan=0.0, posinf=1.0e4, neginf=-1.0e4)
        if self.use_condition_identity_path:
            z = self.query_residual_norm(q + z)
        return self.out(z)


class ConditionTimeHead(nn.Module):
    """
    Factorised condition-specific time head.

    Input:
      features: (B, K, D)

    Output:
      logits: (B, K, T)
    """

    def __init__(
        self,
        d_in,
        n_time_bins,
        hidden_dim=128,
        n_layers=2,
        dropout=0.10,
        final_init_std=1e-3,
    ):
        super().__init__()

        self.net = make_mlp(
            d_in=int(d_in),
            d_hidden=int(hidden_dim),
            d_out=int(n_time_bins),
            n_layers=int(n_layers),
            dropout=float(dropout),
            final_init_std=final_init_std,
        )

    def forward(self, x):
        return self.net(x)


class DeathTimeHead(nn.Module):
    """
    Shared death head.

    Input:
      patient_vec: (B, D_in)

    Output:
      death_logits: (B, T)
    """

    def __init__(
        self,
        d_in,
        n_time_bins,
        hidden_dim=128,
        n_layers=2,
        dropout=0.10,
        final_init_std=1e-3,
    ):
        super().__init__()

        self.net = make_mlp(
            d_in=int(d_in),
            d_hidden=int(hidden_dim),
            d_out=int(n_time_bins),
            n_layers=int(n_layers),
            dropout=float(dropout),
            final_init_std=final_init_std,
        )

    def forward(self, patient_vec):
        return self.net(patient_vec)


class AttnPool(nn.Module):
    """
    Learned single-query attention pooling.

    Sequence use:
      H: (B, L, D), key_padding_mask: (B, L)

    Condition-axis use:
      H: (B, K, D), key_padding_mask: (B, K)
    """

    def __init__(self, d_model, n_heads=4, dropout=0.10):
        super().__init__()

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )

        self.q = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, H, key_padding_mask=None):
        if key_padding_mask is not None:
            # MultiheadAttention cannot handle rows where every key is masked.
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False

        q = self.q.expand(H.shape[0], -1, -1)
        z, _ = self.attn(
            q,
            H,
            H,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.norm(z.squeeze(1))


class ConditionInteractionBlock(nn.Module):
    """
    Self-attention across the condition axis.

    Input/output:
      z: (B, K, D)

    The residual gates are initialised at zero, so the block starts as an exact
    no-op and learns condition-condition interactions only if useful.
    """

    def __init__(self, d_model, n_heads, dropout=0.10):
        super().__init__()

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}."
            )

        self.norm1 = nn.LayerNorm(self.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.n_heads,
            dropout=float(dropout),
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(self.d_model)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, 2 * self.d_model),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.d_model, self.d_model),
        )

        self.g_attn = nn.Parameter(torch.tensor(0.0))
        self.g_ff = nn.Parameter(torch.tensor(0.0))

    def forward(self, z):
        h = self.norm1(z)
        a, _ = self.attn(h, h, h, need_weights=False)
        z = z + self.g_attn * a
        z = z + self.g_ff * self.ff(self.norm2(z))
        return z



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


class A10CanonicalExactRateModel(nn.Module):
    """
    A10 reader-temporal-kernel representation with a direct shared patient-state
    disease log-rate on the locked exact-time task.

    The original condition-specific pathway is preserved. A zero-initialised
    time-specific term derived from `cls_vec` is added with equal weight to every
    condition's log-rate before the rate clamp and competing-risk calculations.
    """
    def __init__(self, n_conditions, n_time_bins, initializers, crp_knots):
        super().__init__()
        self.K = int(n_conditions)
        self.T = int(n_time_bins)
        self.D = int(D_MODEL)
        self.use_condition_identity_path = bool(cfg.USE_CONDITION_IDENTITY_PATH)

        self.static_encoder = StaticCovariateEncoder(
            n_gender=N_GENDER,
            n_ethnicity=N_ETHNICITY,
            n_townsend=N_TOWNSEND,
            n_continuous_static=N_CONT_STATIC,
            d_model=self.D,
            d_static=D_STATIC,
            dropout=DROPOUT_ENCODER,
        )
        self.sequence_encoder = EventSequenceEncoder(
            vocab_size=VOCAB_SIZE,
            type_vocab_size=TYPE_VOCAB_SIZE,
            d_model=self.D,
            n_heads=N_HEADS,
            d_ff=D_FF,
            n_layers=N_LAYERS,
            dropout=DROPOUT_ENCODER,
            pad_id=int(EV_PAD_ID),
            type_pad_id=int(TYPE_PAD_ID),
            crp_knots=crp_knots,
            use_rel_time_bias=FINAL_USE_RELATIVE_TIME_ATTENTION_BIAS,
            rel_time_n_basis=16,
        )
        self.history_reader = ConditionHistoryReader(
            n_conditions=self.K,
            d_model=self.D,
            n_heads=N_HEADS,
            dropout=DROPOUT_HAZARD,
            vocab_size=VOCAB_SIZE,
            type_vocab_size=TYPE_VOCAB_SIZE,
            use_reader_temporal_bias=FINAL_USE_READER_TEMPORAL_BIAS,
            reader_temporal_n_basis=FINAL_READER_TEMPORAL_BIAS_N_BASIS,
            reader_temporal_max_abs_bias=FINAL_READER_TEMPORAL_BIAS_MAX_ABS,
            reader_temporal_mode=FINAL_READER_TEMPORAL_BIAS_MODE,
            reader_temporal_interaction_hidden_multiplier=(
                FINAL_READER_TEMPORAL_INTERACTION_HIDDEN_MULTIPLIER
            ),
            reader_temporal_interaction_dropout=(
                FINAL_READER_TEMPORAL_INTERACTION_DROPOUT
            ),
            use_condition_identity_path=self.use_condition_identity_path,
        )

        self.baseline_feature = nn.Sequential(
            nn.LayerNorm(2 * self.D),
            nn.Linear(2 * self.D, self.D),
            nn.GELU(),
            nn.Dropout(DROPOUT_HAZARD),
            nn.Linear(self.D, self.D),
            nn.LayerNorm(self.D),
        )
        self.history_feature = nn.Sequential(
            nn.LayerNorm(3 * self.D),
            nn.Linear(3 * self.D, self.D),
            nn.GELU(),
            nn.Dropout(DROPOUT_HAZARD),
            nn.Linear(self.D, self.D),
            nn.LayerNorm(self.D),
        )
        self.condition_interaction = nn.Sequential(
            ConditionInteractionBlock(
                d_model=self.D,
                n_heads=FINAL_CONDITION_INTERACTION_HEADS,
                dropout=DROPOUT_HAZARD,
            )
        )
        self.baseline_head = ConditionTimeHead(
            d_in=self.D, n_time_bins=self.T, hidden_dim=HAZARD_HEAD_HIDDEN_DIM,
            n_layers=HAZARD_HEAD_LAYERS, dropout=DROPOUT_HAZARD,
            final_init_std=RESIDUAL_INIT_STD,
        )
        self.history_head = ConditionTimeHead(
            d_in=self.D, n_time_bins=self.T, hidden_dim=HAZARD_HEAD_HIDDEN_DIM,
            n_layers=HAZARD_HEAD_LAYERS, dropout=DROPOUT_HAZARD,
            final_init_std=RESIDUAL_INIT_STD,
        )
        self.death_feature = nn.Sequential(
            nn.LayerNorm(3 * self.D),
            nn.Linear(3 * self.D, self.D),
            nn.GELU(),
            nn.Dropout(DROPOUT_DEATH),
            nn.Linear(self.D, self.D),
            nn.LayerNorm(self.D),
        )
        self.death_head = DeathTimeHead(
            d_in=self.D, n_time_bins=self.T, hidden_dim=DEATH_HEAD_HIDDEN_DIM,
            n_layers=DEATH_HEAD_LAYERS, dropout=DROPOUT_DEATH,
            final_init_std=RESIDUAL_INIT_STD,
        )

        # Instantiate this branch after every pre-existing stochastic module so
        # their parameter initialisation remains unchanged relative to the
        # baseline architecture. Dropout is deliberately zero here, preventing
        # the new branch from perturbing the RNG stream used by existing dropout
        # modules on subsequent batches. The final projection is exactly zero,
        # making the initial disease outputs identical to this architecture
        # with its shared-state branch disabled.
        self.shared_state_head = ConditionTimeHead(
            d_in=self.D,
            n_time_bins=self.T,
            hidden_dim=SHARED_STATE_HEAD_HIDDEN_DIM,
            n_layers=SHARED_STATE_HEAD_LAYERS,
            dropout=0.0,
            final_init_std=0.0,
        )
        # Zero initialization keeps the baseline decoder identical in the
        # control run while allowing condition-specific static effects to learn.
        self.condition_baseline_embedding = nn.Parameter(
            torch.zeros(self.K, self.D)
        )

        self.condition_log_rate_intercept = nn.Parameter(
            torch.as_tensor(initializers["condition_log_rate"], dtype=torch.float32)
        )
        self.death_log_rate_intercept = nn.Parameter(
            torch.as_tensor(float(initializers["death_log_rate"]), dtype=torch.float32)
        )

        # REVISION: attained-age term. The time heads emit T independent bins, so the model would have to
        # re-learn the rise of risk with age separately in every bin. This term lets the log-rate depend on
        # attained age (index age + bin centre) through a small shared basis, with a condition- and
        # sex-specific loading. The loading is zero-initialised, so the model starts exactly where it
        # would without the term (the baseline-equivalence assertion below still holds).
        self.use_attained_age_term = bool(USE_ATTAINED_AGE_TERM)
        if self.use_attained_age_term:
            _centres = torch.as_tensor(np.asarray(haz.BIN_CENTERS), dtype=torch.float32) / 12.0
            if int(_centres.numel()) != self.T:
                raise ValueError(
                    f"haz.BIN_CENTERS has {int(_centres.numel())} entries but the model has T={self.T} bins."
                )
            self.register_buffer("bin_center_years", _centres, persistent=True)
            self.attained_age_basis = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 4))
            self.attained_age_loading = nn.Parameter(torch.zeros(N_GENDER, self.K, 4))

    def forward(self, batch):
        static_vec = self.static_encoder(
            batch["gender_idx"], batch["ethnicity_idx"], batch["townsend_q"],
            batch.get("continuous_static", None),
        )
        seq_z, cls_vec = self.sequence_encoder(
            event_ids=batch["event_ids"],
            event_type_ids=batch["event_type_ids"],
            event_times_months=batch["event_times_months"],
            event_values=batch["event_values"],
            event_value_masks=batch["event_value_masks"],
            attn_mask=batch["attn_mask"],
            static_vec=static_vec,
            lengths=batch["lengths"],
        )
        hist_cond = self.history_reader(
            seq_z=seq_z,
            attn_mask=batch["attn_mask"],
            static_vec=static_vec,
            cls_vec=cls_vec,
            event_ids=batch["event_ids"],
            event_type_ids=batch["event_type_ids"],
            event_times_months=batch["event_times_months"],
            index_age_months=batch.get("index_age_months", None),
        )

        B = static_vec.shape[0]
        static_cond = static_vec.unsqueeze(1).expand(B, self.K, self.D)
        zero_state_cond = torch.zeros_like(static_cond)
        baseline_condition_state = (
            self.condition_baseline_embedding.unsqueeze(0).expand(B, -1, -1)
            if self.use_condition_identity_path
            else zero_state_cond
        )
        baseline_features = self.baseline_feature(
            torch.cat([baseline_condition_state, static_cond], dim=-1)
        )
        history_features = self.history_feature(
            torch.cat([hist_cond, zero_state_cond, static_cond], dim=-1)
        )
        history_features_interacted = self.condition_interaction(history_features)

        baseline_residual = self.baseline_head(baseline_features)
        history_residual = self.history_head(history_features_interacted)
        condition_log_rate_without_shared = (
            self.condition_log_rate_intercept.view(1, self.K, 1)
            + baseline_residual
            + history_residual
        )

        zero_state_pool = torch.zeros_like(static_vec)
        death_features = self.death_feature(
            torch.cat([cls_vec, static_vec, zero_state_pool], dim=-1)
        )
        death_residual = self.death_head(death_features)
        death_log_rate_unclamped = self.death_log_rate_intercept + death_residual
        death_log_rate = death_log_rate_unclamped.clamp(LOG_RATE_MIN, LOG_RATE_MAX)
        death_rate = torch.exp(death_log_rate).clamp_min(RATE_EPS)

        # The shared patient-state signal varies by participant and time bin.
        # Every condition receives that signal with the same weight.
        shared_state_residual = self.shared_state_head(cls_vec)
        shared_state_condition_residual = shared_state_residual.unsqueeze(1).expand(B, self.K, self.T)
        if self.use_attained_age_term:
            attained_age = (
                batch["index_age_years"].float().view(B, 1)
                + self.bin_center_years.view(1, self.T)
            )                                                            # (B, T) years
            phi = self.attained_age_basis(((attained_age - 60.0) / 10.0).unsqueeze(-1))   # (B, T, 4)
            loading = self.attained_age_loading[batch["gender_idx"].long()]             # (B, K, 4)
            attained_age_residual = torch.einsum("bkr,btr->bkt", loading, phi)          # (B, K, T)
        else:
            attained_age_residual = torch.zeros_like(shared_state_residual).unsqueeze(1)
        condition_log_rate_unclamped = (
            condition_log_rate_without_shared
            + shared_state_condition_residual
            + attained_age_residual
        )
        condition_log_rate = condition_log_rate_unclamped.clamp(
            LOG_RATE_MIN, LOG_RATE_MAX
        )
        condition_rate = torch.exp(condition_log_rate).clamp_min(RATE_EPS)

        bin_width = float(haz.BIN_WIDTH_MONTHS)
        p_cond_given_alive = -torch.expm1(-condition_rate * bin_width)
        p_death = -torch.expm1(-death_rate * bin_width)
        death_rate_k = death_rate.unsqueeze(1).expand(-1, self.K, -1)
        total_rate = condition_rate + death_rate_k
        p_neither = torch.exp(-total_rate * bin_width)
        pair_failure = -torch.expm1(-total_rate * bin_width)
        p_condition = (condition_rate / total_rate.clamp_min(RATE_EPS)) * pair_failure
        p_death_k = (death_rate_k / total_rate.clamp_min(RATE_EPS)) * pair_failure
        p_sum = p_condition + p_death_k + p_neither

        return {
            "condition_logits": condition_log_rate,
            "death_logits": death_log_rate,
            "condition_rate": condition_rate,
            "death_rate": death_rate,
            "p_condition": p_condition,
            "p_death_k": p_death_k,
            "p_neither": p_neither,
            "p_death": p_death,
            "p_cond_given_alive": p_cond_given_alive,
            "p_sum": p_sum,
            "baseline_logits": baseline_residual,
            "history_logits": history_residual,
            "history_residual_logits": history_residual,
            "shared_state_logits": shared_state_residual,
            "attained_age_residual": attained_age_residual,
            "seq_z": seq_z,
            "cls_vec": cls_vec,
            "global_vec": cls_vec,
            "static_vec": static_vec,
            "hist_cond": hist_cond,
            "history_features": history_features,
            "history_features_interacted": history_features_interacted,
            "condition_log_rate_without_shared": condition_log_rate_without_shared,
            "condition_log_rate_unclamped": condition_log_rate_unclamped,
            "death_log_rate_unclamped": death_log_rate_unclamped,
            "condition_log_rate_saturation_fraction": condition_log_rate_unclamped.ne(condition_log_rate).float().mean(),
            "death_log_rate_saturation_fraction": death_log_rate_unclamped.ne(death_log_rate).float().mean(),
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


crp_knots = fit_crp_knots(train_ds)
initializers = estimate_train_global_log_rate_initializers(train_samples, int(K))
model = A10CanonicalExactRateModel(
    n_conditions=int(K),
    n_time_bins=int(N_TIME_BINS),
    initializers=initializers,
    crp_knots=crp_knots,
).to(DEVICE)

if len(model.sequence_encoder.encoder.layers) != int(cfg.N_LAYERS):
    raise AssertionError("Active encoder depth does not match cfg.N_LAYERS.")
if (
    model.history_reader.reader_temporal_bias is not None
    and model.history_reader.reader_temporal_bias.mode
    != FINAL_READER_TEMPORAL_BIAS_MODE
):
    raise AssertionError("Reader temporal-bias mode was not propagated correctly.")


with torch.no_grad():
    b = next(iter(train_loader))
    b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
    out = model(b)
    assert out["condition_rate"].shape == (b["event_ids"].shape[0], int(K), int(N_TIME_BINS))
    assert out["death_rate"].shape == (b["event_ids"].shape[0], int(N_TIME_BINS))
    assert out["shared_state_logits"].shape == (
        b["event_ids"].shape[0], int(N_TIME_BINS)
    )
    assert float((out["p_sum"] - 1.0).abs().max().item()) < 1e-5
    assert torch.isfinite(out["condition_rate"]).all()
    assert torch.isfinite(out["death_rate"]).all()
    shared_state_initial_max_abs = float(
        out["shared_state_logits"].abs().max().item()
    )
    initial_equivalence_max_abs = float(
        (
            out["condition_log_rate_unclamped"]
            - out["condition_log_rate_without_shared"]
        ).abs().max().item()
    )
    if shared_state_initial_max_abs != 0.0:
        raise AssertionError(
            "Shared-state head is not exactly zero at initialization: "
            f"max_abs={shared_state_initial_max_abs}."
        )
    if initial_equivalence_max_abs != 0.0:
        raise AssertionError(
            "Shared-state variant does not reproduce the baseline disease "
            "log-rates at initialization: "
            f"max_abs={initial_equivalence_max_abs}."
        )

model_meta = {
    "MODEL_VARIANT": MODEL_VARIANT,
    "primary_brier_estimator": "ipcw_horvitz_thompson_eligible_denominator",
    "architecture_source": ARCHITECTURE_SOURCE,
    "task_adapter": "canonical samples + exact post-entry piecewise-exponential rate likelihood",
    "delayed_entry_contract": delayed_entry_contract(),
    "dimensions": {
        "K": int(K), "N_TIME_BINS": int(N_TIME_BINS), "D_MODEL": D_MODEL,
        "D_FF": D_FF, "N_HEADS": N_HEADS, "N_LAYERS": N_LAYERS,
        "D_STATIC": D_STATIC, "N_CONT_STATIC": N_CONT_STATIC,
    },
    "encoder_configuration": {
        "actual_encoder_layers": len(model.sequence_encoder.encoder.layers),
        "configured_encoder_layers": int(cfg.N_LAYERS),
    },
    "architecture_extensions": {
        "condition_identity_path": bool(model.use_condition_identity_path),
        "condition_reader_query_residual": bool(model.use_condition_identity_path),
        "condition_aware_baseline": bool(model.use_condition_identity_path),
        "condition_interaction": True,
        "reader_temporal_bias": bool(FINAL_USE_READER_TEMPORAL_BIAS),
        "static_continuous_input_layernorm": bool(STATIC_CONT_INPUT_LAYERNORM),
        "attained_age_term": bool(USE_ATTAINED_AGE_TERM),
        "n_heads": int(N_HEADS),
        "reader_temporal_bias_mode": FINAL_READER_TEMPORAL_BIAS_MODE,
        "reader_temporal_interaction_features": (
            "condition,event_type_context,recency,condition_x_event,"
            "condition_x_recency,event_x_recency"
        ),
        "encoder_relative_time_attention_bias": False,
        "death_sequence_attention_pool": False,
        "shared_state_hazard_branch": True,
        "shared_state_source": "cls_vec",
        "shared_state_output_shape": "participant_by_time_bin_shared_across_conditions",
        "shared_state_condition_loading": "equal_weight_for_all_conditions",
        "crp_value_embedding": "linear_value_direction_plus_piecewise_linear_residual",
        "crp_knot_quantiles": list(CRP_KNOT_QUANTILES),
        "crp_knots_training_values": crp_knots.tolist(),
        "shared_state_dropout": 0.0,
        "shared_state_final_projection_zero_initialised": True,
        "shared_state_initial_max_abs": shared_state_initial_max_abs,
        "baseline_equivalence_initial_max_abs": initial_equivalence_max_abs,
        "condition_residual_centering": False,
        "state_branch": "direct cls_vec-to-time shared disease log-rate branch",
    },
    "train_post_entry_mle_initialization": {
        "condition_events": initializers["condition_events"].tolist(),
        "condition_exposure_months": initializers["condition_exposure_months"].tolist(),
        "death_events": float(initializers["death_events"]),
        "death_exposure_months": float(initializers["death_exposure_months"]),
    },
    "parameters": {
        "n_total": int(sum(p.numel() for p in model.parameters())),
        "n_trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    },
}
with open(RUN_DIR / "cell5_model_metadata_death_adjusted.json", "w") as f:
    json.dump(model_meta, f, indent=2, default=str)

print("A10 architecture instantiated on locked task:")
print(json.dumps(model_meta["dimensions"], indent=2))
print("Total parameters:", f"{model_meta['parameters']['n_total']:,}")
print(
    "Shared-state zero-initialisation / baseline-equivalence max abs:",
    shared_state_initial_max_abs,
    initial_equivalence_max_abs,
)
print("✅ Cell 5 complete — shared-state CRESCENT adapted to the locked exact-time task.")
