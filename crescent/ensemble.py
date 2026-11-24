"""Stage 5 (optional): seed ensemble; does nothing unless CRESCENT_ENSEMBLE_RUN_DIRS is set.

Extracted from the CRESCENT_Final notebook (cell 15). The code is unchanged apart from
machine-specific default paths. Stages share one namespace, as the notebook cells did, so run
them through `run_crescent.py` rather than importing this file on its own.
"""

# =============================================================================
# Notebook cell 15
# =============================================================================

import os, json
import torch
import torch.nn as nn
from pathlib import Path

ENSEMBLE_RUN_DIRS = [p.strip() for p in os.getenv("CRESCENT_ENSEMBLE_RUN_DIRS", "").split(",") if p.strip()]


class RateAveragingEnsemble(nn.Module):
    """Average condition and death hazards over members, then recompute the competing-risk probabilities."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        self.use_condition_identity_path = bool(members[0].use_condition_identity_path)

    def load_state_dict(self, state_dict, strict=True):
        # The locked evaluation re-loads this run's primary checkpoint; it belongs to member 0 only.
        return self.members[0].load_state_dict(state_dict, strict=strict)

    def forward(self, batch):
        outs = [m(batch) for m in self.members]
        condition_rate = torch.stack([o["condition_rate"] for o in outs]).mean(dim=0)
        death_rate = torch.stack([o["death_rate"] for o in outs]).mean(dim=0)
        width = float(haz.BIN_WIDTH_MONTHS)
        K_ = condition_rate.shape[1]
        p_cond_given_alive = -torch.expm1(-condition_rate * width)
        p_death = -torch.expm1(-death_rate * width)
        death_rate_k = death_rate.unsqueeze(1).expand(-1, K_, -1)
        total_rate = condition_rate + death_rate_k
        p_neither = torch.exp(-total_rate * width)
        pair_failure = -torch.expm1(-total_rate * width)
        p_condition = (condition_rate / total_rate.clamp_min(RATE_EPS)) * pair_failure
        p_death_k = (death_rate_k / total_rate.clamp_min(RATE_EPS)) * pair_failure
        return {
            "condition_rate": condition_rate, "death_rate": death_rate,
            "condition_logits": torch.log(condition_rate.clamp_min(RATE_EPS)),
            "death_logits": torch.log(death_rate.clamp_min(RATE_EPS)),
            "p_condition": p_condition, "p_death_k": p_death_k, "p_neither": p_neither,
            "p_death": p_death, "p_cond_given_alive": p_cond_given_alive,
            "p_sum": p_condition + p_death_k + p_neither,
        }


def _load_ensemble_member(run_dir):
    path = Path(run_dir).expanduser() / "best_model_by_val_loss.pt"
    if not path.exists():
        raise FileNotFoundError(f"Ensemble member checkpoint not found: {path}")
    ck = torch.load(path, map_location=DEVICE)
    prepared_hashes = {k: v for k, v in ck.get("model_input_hashes", {}).items() if k in INPUT_HASHES}
    if prepared_hashes != {k: INPUT_HASHES[k] for k in prepared_hashes} or not prepared_hashes:
        raise ValueError(f"{path}: prepared-data hashes differ from this run.")
    if int(ck.get("K", -1)) != int(K) or [str(x) for x in ck.get("condition_names", [])] != [str(x) for x in condition_names]:
        raise ValueError(f"{path}: condition list differs from this run.")
    if ck.get("hazard_binning") != {"BIN_WIDTH_MONTHS": int(haz.BIN_WIDTH_MONTHS),
                                    "MAX_HORIZON_MONTHS": int(haz.MAX_HORIZON_MONTHS), "N_BINS": int(haz.N_BINS)}:
        raise ValueError(f"{path}: hazard binning differs from this run.")
    if str(ck.get("model_variant", "")) != str(cfg.MODEL_VARIANT) or ck.get("architecture_source") != ARCHITECTURE_SOURCE:
        raise ValueError(f"{path}: model variant / architecture source differs from this run.")
    if ck.get("delayed_entry_contract") != delayed_entry_contract():
        raise ValueError(f"{path}: delayed-entry contract differs from this run.")
    member = A10CanonicalExactRateModel(
        n_conditions=int(K), n_time_bins=int(N_TIME_BINS),
        initializers=initializers, crp_knots=crp_knots,
    ).to(DEVICE)
    member.load_state_dict(ck["model_state_dict"], strict=True)   # strict: architecture switches must match
    member.eval()
    for p in member.parameters():
        p.requires_grad_(False)
    return member, path, ck


if ENSEMBLE_RUN_DIRS:
    if isinstance(model, RateAveragingEnsemble):
        raise RuntimeError("`model` is already an ensemble; restart from the training cell.")
    _loaded = [_load_ensemble_member(d) for d in ENSEMBLE_RUN_DIRS]
    _own = model.eval()
    for p in _own.parameters():
        p.requires_grad_(False)
    model = RateAveragingEnsemble([_own] + [m for m, _, _ in _loaded]).to(DEVICE).eval()
    _manifest = {
        "n_members": len(model.members),
        "members": [{"run_dir": str(RUN_DIR), "checkpoint": str(best_loss_path), "best_val_loss": float(best_val_loss),
                     "best_iteration": int(best_loss_iteration)}]
                   + [{"run_dir": str(p.parent), "checkpoint": str(p),
                       "best_val_loss": ck.get("extra", {}).get("row", {}).get("val_loss"),
                       "best_iteration": ck.get("iteration"), "sha256": sha256_file(p)} for _, p, ck in _loaded],
        "combination": "mean of condition and death hazards, probabilities recomputed",
    }
    with open(RUN_DIR / "ensemble_manifest.json", "w") as _h:
        json.dump(_manifest, _h, indent=2, default=str)
    print(f"Ensemble of {len(model.members)} models active for sections 7 onwards (manifest: ensemble_manifest.json).")
else:
    print("No ensemble requested (CRESCENT_ENSEMBLE_RUN_DIRS unset); the single restored checkpoint is evaluated.")
