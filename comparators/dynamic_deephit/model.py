# Dynamic-DeepHit-inspired — GRU and attention architecture, empirical rate priors, and post-entry likelihood
# Stage file executed in a shared namespace via run_comparator.py (not an importable module).


# --- Notebook cell 7 ---
from types import SimpleNamespace
from typing import Dict, Tuple

CFG = SimpleNamespace(d_static=64,d_model=256,hidden_dim=512,gru_layers=1,dropout=.10,
                      condition_chunk_size=128,min_rate_per_month=1e-8,max_rate_per_month=1.)
N_GENDER, N_ETHNICITY, N_TOWNSEND = [int(st[k]) for k in ("N_GENDER_MODEL","N_ETHNICITY_MODEL","N_TOWNSEND_MODEL")]
N_CONTINUOUS_STATIC = n_cont
EVENT_VOCAB_SIZE, TYPE_VOCAB_SIZE = int(ev["EV_VOCAB_SIZE_MODEL"]), int(ty["TYPE_VOCAB_SIZE_MODEL"])
EVENT_PAD_ID, TYPE_PAD_ID = PAD, TPAD
N_BINS, MAX_HORIZON_MONTHS = BINS, WIDTH * BINS

def empirical_prior(active, status, stop):
    """Train-only per-bin log rates, shrunk to each cause's overall event rate."""
    starts = np.arange(BINS) * WIDTH
    k = active.shape[1]
    exposure = np.zeros((k, BINS), dtype=np.float64)
    events = np.zeros_like(exposure)
    for first in range(0, len(active), 2048):
        a = active[first:first+2048]
        t = stop[first:first+2048]
        s = status[first:first+2048]
        for j, start in enumerate(starts):
            left = max(start, ENTRY)
            duration = np.clip(t - left, 0, max(0, start + WIDTH - left))
            exposure[:,j] += np.where(a, duration, 0.).sum(axis=0)
        i, condition = np.nonzero(a & (s == 1) & (t <= WIDTH * BINS))
        bins = (np.ceil(t[i, condition] / WIDTH).astype(int) - 1).clip(0, BINS-1)
        np.add.at(events, (condition, bins), 1)
    pseudo = 120.
    baseline = (events.sum(axis=1) + .5) / (exposure.sum(axis=1) + pseudo)
    rates = (events + pseudo * baseline[:,None]) / (exposure + pseudo)
    return np.log(rates.clip(1e-8, 1.)).astype("float32")

def train_priors(targets):
    disease = empirical_prior(targets["active"], targets["status"], targets["stop"])
    death = empirical_prior(targets["death_active"][:,None], targets["death_status"][:,None],
                            targets["death_stop"][:,None])[0]
    return disease, death

class ContinuousTimeEncoding(nn.Module):
    def __init__(self, dimension: int, base: float = 10_000.0):
        super().__init__()
        if dimension % 2:
            raise ValueError("Time-encoding dimension must be even.")
        self.dimension = int(dimension)
        self.base = float(base)

    def forward(self, months: torch.Tensor) -> torch.Tensor:
        years = months.float() / 12.0
        half = self.dimension // 2
        divisor = torch.exp(
            torch.arange(half, device=months.device, dtype=torch.float32)
            * (-math.log(self.base) / max(half - 1, 1))
        )
        angle = years.unsqueeze(-1) * divisor
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class StaticEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        ds, d = CFG.d_static, CFG.d_model
        self.gender = nn.Embedding(N_GENDER, ds)
        self.ethnicity = nn.Embedding(N_ETHNICITY, ds)
        self.townsend = nn.Embedding(N_TOWNSEND, ds)
        self.continuous = nn.Sequential(
            nn.Linear(N_CONTINUOUS_STATIC, ds),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(ds, ds),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(4 * ds),
            nn.Linear(4 * ds, d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        pieces = [
            self.gender(batch["cats"][:,0]),
            self.ethnicity(batch["cats"][:,1]),
            self.townsend(batch["cats"][:,2]),
            self.continuous(batch["numeric"].float()),
        ]
        return self.fusion(torch.cat(pieces, dim=-1))


class AdditiveTemporalAttention(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.key = nn.Linear(hidden, hidden)
        self.query = nn.Linear(hidden, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(
        self,
        sequence: torch.Tensor,
        query: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(
            torch.tanh(self.key(sequence) + self.query(query).unsqueeze(1))
        ).squeeze(-1)
        logits = logits.masked_fill(~mask.bool(), -1e4)
        weights = torch.softmax(logits.float(), dim=-1).to(sequence.dtype)
        context = torch.einsum("bl,bld->bd", weights, sequence)
        return context, weights


class EventSequenceEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        d, hidden = CFG.d_model, CFG.hidden_dim
        self.event = nn.Embedding(EVENT_VOCAB_SIZE, d, padding_idx=EVENT_PAD_ID)
        self.event_type = nn.Embedding(TYPE_VOCAB_SIZE, d, padding_idx=TYPE_PAD_ID)
        self.value_direction = nn.Embedding(EVENT_VOCAB_SIZE, d, padding_idx=EVENT_PAD_ID)
        self.value_state = nn.Embedding(2, d)
        self.time = ContinuousTimeEncoding(d)
        self.input_norm = nn.LayerNorm(d)
        self.input_dropout = nn.Dropout(CFG.dropout)
        self.gru = nn.GRU(
            input_size=d,
            hidden_size=hidden,
            num_layers=CFG.gru_layers,
            batch_first=True,
            dropout=CFG.dropout if CFG.gru_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.attention = AdditiveTemporalAttention(hidden)
        self.sequence_projection = nn.Linear(hidden, d)
        self.summary_projection = nn.Sequential(
            nn.LayerNorm(2 * hidden),
            nn.Linear(2 * hidden, d),
            nn.GELU(),
        )
        self.output_norm = nn.LayerNorm(d)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        static: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed = (batch["mask"] > 0.5).long()
        value_component = (
            batch["value"].float().unsqueeze(-1)
            * observed.float().unsqueeze(-1)
            * self.value_direction(batch["ids"])
        )
        inputs = (
            self.event(batch["ids"])
            + self.event_type(batch["types"])
            + self.time(batch["time"])
            + self.value_state(observed)
            + value_component
            + static.unsqueeze(1)
        )
        inputs = self.input_dropout(self.input_norm(inputs))
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs,
            batch["lengths"].detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_hidden, _ = self.gru(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(
            packed_hidden,
            batch_first=True,
            total_length=inputs.shape[1],
        )
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        last = hidden[rows, batch["lengths"] - 1]
        context, weights = self.attention(hidden, last, batch["attention_mask"])
        sequence = self.output_norm(self.sequence_projection(hidden))
        summary = self.output_norm(
            self.summary_projection(torch.cat([context, last], dim=-1))
        )
        return sequence, summary, weights


class ConditionTimeReader(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG.d_model
        self.condition_embedding = nn.Parameter(torch.empty(K, d))
        nn.init.normal_(self.condition_embedding, std=0.02)
        self.patient_film = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.Tanh())
        self.query = nn.Linear(d, d, bias=False)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.recency = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.fusion = nn.Sequential(
            nn.LayerNorm(4 * d),
            nn.Linear(4 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(2 * d, d),
            nn.GELU(),
        )
        self.residual = nn.Linear(d, N_BINS)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        sequence: torch.Tensor,
        summary: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        d = CFG.d_model
        scale, shift = self.patient_film(torch.cat([summary, static], dim=-1)).chunk(2, dim=-1)
        base = self.condition_embedding.unsqueeze(0)
        queries = base * (1.0 + 0.25 * scale.unsqueeze(1)) + 0.25 * shift.unsqueeze(1)

        recency = (
            batch["index_age_months"][:, None] - batch["time"]
        ).clamp_min(0.0)
        is_cls = torch.zeros_like(recency)
        is_cls.scatter_(1, (batch["lengths"] - 1).view(-1, 1), 1.0)
        recency_features = torch.stack([
            torch.log1p(recency) / math.log1p(max(MAX_HORIZON_MONTHS * 10.0, 1.0)),
            torch.exp(-recency / 24.0),
            is_cls,
        ], dim=-1)
        keys = self.key(sequence + self.recency(recency_features))
        values = self.value(sequence)

        chunks = []
        for start in range(0, K, CFG.condition_chunk_size):
            stop = min(K, start + CFG.condition_chunk_size)
            q = self.query(queries[:, start:stop])
            scores = torch.einsum("bkd,bld->bkl", q, keys) / math.sqrt(d)
            scores = scores.masked_fill(~batch["attention_mask"][:, None, :], -1e4)
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            pooled = torch.einsum("bkl,bld->bkd", weights, values)
            width = stop - start
            fused = torch.cat([
                pooled,
                queries[:, start:stop],
                summary[:, None, :].expand(-1, width, -1),
                static[:, None, :].expand(-1, width, -1),
            ], dim=-1)
            chunks.append(self.residual(self.fusion(fused)))
        return torch.cat(chunks, dim=1)


class SharedDeathHead(nn.Module):
    def __init__(self):
        super().__init__()
        d = CFG.d_model
        self.query = nn.Linear(2 * d, d)
        self.key = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * d),
            nn.Linear(3 * d, 2 * d),
            nn.GELU(),
            nn.Dropout(CFG.dropout),
            nn.Linear(2 * d, d),
            nn.GELU(),
        )
        self.residual = nn.Linear(d, N_BINS)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        sequence: torch.Tensor,
        summary: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        query = self.query(torch.cat([summary, static], dim=-1))
        scores = torch.einsum("bd,bld->bl", query, self.key(sequence)) / math.sqrt(CFG.d_model)
        scores = scores.masked_fill(~batch["attention_mask"], -1e4)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        pooled = torch.einsum("bl,bld->bd", weights, self.value(sequence))
        return self.residual(self.fusion(torch.cat([pooled, summary, static], dim=-1)))


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
    if isinstance(module, DDHCanonicalCompetingRiskModel):
        nn.init.zeros_(module.condition_reader.residual.weight)
        nn.init.zeros_(module.condition_reader.residual.bias)
        nn.init.zeros_(module.death_head.residual.weight)
        nn.init.zeros_(module.death_head.residual.bias)

class DDHCanonicalCompetingRiskModel(nn.Module):
    def __init__(self, condition_prior, death_prior):
        super().__init__()
        self.static = StaticEncoder()
        self.sequence = EventSequenceEncoder()
        self.condition_reader = ConditionTimeReader()
        self.death_head = SharedDeathHead()
        self.register_buffer("condition_log_prior", torch.from_numpy(condition_prior.copy()))
        self.register_buffer("death_log_prior", torch.from_numpy(death_prior.copy()))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        static = self.static(batch)
        sequence, summary, temporal_attention = self.sequence(batch, static)
        condition_residual = self.condition_reader(batch, sequence, summary, static)
        death_residual = self.death_head(batch, sequence, summary, static)
        low, high = math.log(CFG.min_rate_per_month), math.log(CFG.max_rate_per_month)
        condition_log_rate = (
            self.condition_log_prior.unsqueeze(0) + condition_residual
        ).clamp(low, high)
        death_log_rate = (
            self.death_log_prior.unsqueeze(0) + death_residual
        ).clamp(low, high)
        return condition_log_rate, death_log_rate, temporal_attention

def nll_sums(log_rates,active,status,stop):
    stop=stop.to(log_rates.dtype)
    starts=torch.arange(BINS,device=stop.device,dtype=stop.dtype)*WIDTH
    exposure=((stop.unsqueeze(-1)-starts).clamp(0,WIDTH) - (ENTRY-starts).clamp(0,WIDTH)).clamp_min(0.)
    event=active&(status==1)&(stop<=WIDTH*BINS+ENTRY_EPS)
    event_bin=(torch.ceil(stop/WIDTH).long()-1).clamp(0,BINS-1)
    log_at_event=log_rates.gather(-1,event_bin.unsqueeze(-1)).squeeze(-1)
    loss=(log_rates.exp()*exposure).sum(-1)-event*log_at_event
    return loss.masked_fill(~active,0.).sum(),active.sum()

def components(log_rates,y):
    a,b=log_rates[:2]
    dn,nd=nll_sums(a,y["active"],y["status"],y["stop"])
    mn,nm=nll_sums(b,y["death_active"],y["death_status"],y["death_stop"])
    return dn,nd,mn,nm

def risk_curves(log_rates,horizons):
    a,b=(r.exp()[...,ENTRY//WIDTH:] for r in log_rates[:2])
    total=a+b.unsqueeze(1)
    failure=-torch.expm1(-WIDTH*total)
    cumulative=(WIDTH*total).cumsum(-1)
    survival_before=torch.cat([torch.ones_like(total[...,:1]),torch.exp(-cumulative[...,:-1])],-1)
    idx=[h//WIDTH-1 for h in horizons]
    adj=(survival_before*failure*a/total).cumsum(-1)[...,idx]
    return adj,(-torch.expm1(-(WIDTH*a).cumsum(-1)))[...,idx],(-torch.expm1(-(WIDTH*b).cumsum(-1)))[...,idx]

# Exact boundary and bin semantics, checked before training on real data.
def check_loss():
    logs=torch.full((3,1,BINS),math.log(.01),dtype=torch.float64,requires_grad=True)
    a=torch.tensor([[False],[True],[True]])
    s=torch.tensor([[1],[1],[2]])
    u=torch.tensor([[6.],[12.],[9.]])
    d,n=nll_sums(logs,a,s,u)
    assert n.item()==2 and math.isclose(d.item(),.09-math.log(.01),rel_tol=1e-10)
    d.backward()
    assert (logs.grad[...,0]==0).all() and (logs.grad[0]==0).all()
    v=risk_curves((logs.detach(),logs[:,0].detach()),(12,24))[0]
    np.testing.assert_allclose(v[0,0],.5*(1-np.exp(-.02*np.array([12,24]))),rtol=1e-7)
check_loss()
model=DDHCanonicalCompetingRiskModel(*train_priors(train.targets)).to(DEVICE)
initialize_from_scratch(model)
# Exercise the real input, GRU, readers, loss and gradients before starting the long run.
model.train()
sample_x, sample_y = collate(list(range(min(4,len(train)))),train)
sample_output = model(device_batch(sample_x))
assert sample_output[0].shape == (len(sample_x["lengths"]),K,BINS)
assert sample_output[1].shape == (len(sample_x["lengths"]),BINS)
torch.testing.assert_close(sample_output[2].sum(-1),torch.ones_like(sample_output[2][:,0]))
sn, sd, dn, dd = components(sample_output,device_batch(sample_y))
sample_loss = sn / sd.clamp_min(1) + .1 * dn / dd.clamp_min(1)
if not torch.isfinite(sample_loss):
    raise FloatingPointError("Non-finite preflight loss")
sample_loss.backward()
nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
model.zero_grad(set_to_none=True)
del sample_x,sample_y,sample_output,sample_loss,sn,sd,dn,dd
log(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

model_meta = {"model_variant":contract["model_variant"], "architecture":type(model).__name__,
              "n_parameters":sum(p.numel() for p in model.parameters()), "training_settings":contract["config"],
              "hazard_binning":contract["hazard_binning"], "delayed_entry_contract":entry_contract}
(RUN_DIR/"cell5_model_metadata_death_adjusted.json").write_text(json.dumps(model_meta,indent=2))
