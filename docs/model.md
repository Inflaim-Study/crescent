# CRESCENT model details

Summary of manuscript §2.3-2.5. Code: `crescent/` (stages 1-5) and `evaluation/` (stages 6-8).

## Prediction task
- **Landmark:** the first UK Biobank recruitment assessment. Predictors are fixed at this point.
- **Delayed entry:** for each condition, a participant enters the risk set at **month 6** if alive, under
  observation beyond month 6 and with no record of that condition on or before month 6.
- **Competing event:** death.
- **Horizons:** 12, 24, 60 and 120 months after entry (recruitment months 18, 30, 66 and 126).
- **Targets:** 58 of the 60 InflAIM conditions (those with at least 100 training cases).

## Inputs
| Type | Inputs |
|---|---|
| Static | Age, recorded sex, ethnicity, Townsend deprivation (score, quintile, missing flag) |
| Timed events (up to 25) | Adult diagnoses, BMI category, smoking status, alcohol-intake frequency, CRP value |

For longer histories the encoder keeps the most recent records and retains a CRP record if it would otherwise
be dropped.

## Architecture
1. **Static encoder:** learned embeddings for categorical inputs and an MLP for continuous inputs, giving `s_i`.
2. **Event encoder:** each event is the sum of label and type embeddings, a sinusoidal age encoding, a linear
   value term, a CRP piecewise-linear residual (zero-initialised vectors at training CRP quantiles
   0, 10, 25, 50, 75, 90, 100%) and `s_i`. A 3-layer pre-norm Transformer (width 128, **16 heads**, feed-forward
   512, GELU, dropout 0.20) encodes the sequence; the classification token gives `g_i`.
3. **Condition-specific reader:** for each condition, a query built from a condition embedding, `s_i` and `g_i`
   attends to the encoded events with an additive temporal bias that depends on the condition, the event, its
   type and the time since it was recorded.
4. **Condition interaction:** a 16-head attention block with zero-initialised gates shares information across
   conditions.
5. **Hazards:** each condition's log-hazard in six-month bin *u* is the sum of five terms:

   $$\eta^{C}_{iku} = b^{C}_{k} + r^{stat}_{iku} + r^{hist}_{iku} + r^{shared}_{iu} + a_{iku}$$

   (intercept, static baseline, history, shared patient state with unit weight, attained age). Death has its own
   pathway, $\eta^{D}_{iu} = b_{D} + r^{D}_{iu}$. Rates are $\lambda = \max\{10^{-8}, \exp(\mathrm{clip}(\eta,-18,2))\}$ per month.

## From hazards to risk
With $\Delta = 6$ months and $\Lambda = \lambda^{C} + \lambda^{D}$:

$$p^{C}_{iku} = \frac{\lambda^{C}_{iku}}{\Lambda_{iku}}\left(1-e^{-\Lambda_{iku}\Delta}\right), \qquad S_{iku} = \prod_{m\le u} e^{-\Lambda_{ikm}\Delta}$$

$$F^{C}_{ik}(u) = \sum_{m=0}^{u} S_{ik,m-1}\,p^{C}_{ikm}, \qquad R^{C}_{ik}(h \mid \text{entry}) = \frac{F^{C}_{ik}(h/6) - F^{C}_{ik}(0)}{S_{ik}(0)}$$

## Training
| Setting | Value |
|---|---|
| Objective | Delayed-entry piecewise-exponential NLL: $L = L_C + 0.10\,L_D$ (no auxiliary losses) |
| Optimiser | AdamW, batch 256, learning rate 1.2e-4, weight decay 0.05 (not on 0-/1-D parameters) |
| Regularisation | Dropout 0.20, 10% event masking, gradient clipping at 1.0 |
| Schedule | 25,000 updates, 1,000 warm-up, cosine decay to 5% |
| Checkpoint | Validation weighted NLL every 1,000 updates on EMA weights (decay 0.999) |
| Published fit | Seed 42, 1,194,593 parameters, selected at update 15,000 (validation NLL 0.153281) |

## Running
```bash
export CRESCENT_PREPARED_DATA_DIR=/path/to/splits_229950_one_month_cap
export EHR_CRESCENT_FINAL_OUTPUT=runs/CRESCENT_Final     # must be a new, empty folder
python run_crescent.py              # all stages
python run_crescent.py --list       # show stages
```
A single CUDA GPU visible as `cuda:0` is required.

## Settings (environment variables)
Defaults reproduce the published model.

| Variable | Default | Meaning |
|---|---|---|
| `CRESCENT_PREPARED_DATA_DIR` | `splits_229950_one_month_cap` | Folder with the prepared splits |
| `EHR_CRESCENT_FINAL_OUTPUT` | `runs/CRESCENT_Final` | Output folder (must be new) |
| `CRESCENT_SEED` | `42` | Random seed |
| `CRESCENT_N_HEADS` | `16` | Attention heads |
| `CRESCENT_BATCH_SIZE` | `256` | Batch size |
| `CRESCENT_LR` | `1.2e-4` | Learning rate |
| `CRESCENT_WEIGHT_DECAY` | `5e-2` | Weight decay |
| `CRESCENT_WD_EXEMPT_1D` | `1` | No weight decay on 0-/1-D parameters |
| `CRESCENT_DROPOUT` | `0.20` | Dropout |
| `CRESCENT_EVENT_MASK_PROB` | `0.10` | Event masking during training |
| `CRESCENT_EMA_DECAY` | `0.999` | EMA decay (0 turns EMA off) |
| `CRESCENT_ATTAINED_AGE_TERM` | `1` | Attained-age term (0 = ablation A1) |
| `CRESCENT_READER_TEMPORAL_BIAS` | `1` | Reader temporal-bias network |
| `A10_USE_CONDITION_IDENTITY_PATH` | `1` | Condition identity path |
| `CRESCENT_STATIC_CONT_LAYERNORM` | `0` | LayerNorm over continuous static inputs |
| `CRESCENT_TIME_SMOOTHNESS_WEIGHT` | `0.0` | Optional smoothness penalty (off) |
| `CRESCENT_MIN_TRAIN_PARTICIPANTS_PER_CONDITION` | `100` | Minimum training cases per target |
| `CRESCENT_CONDITION_LIST` | (empty) | Reuse an earlier run's condition list |
| `CRESCENT_ENSEMBLE_RUN_DIRS` | (empty) | Optional seed ensemble |
| `A10_CANONICAL_MAX_ITERATIONS` | `25000` | Training updates |
| `A10_CANONICAL_WARMUP_ITERATIONS` | `1000` | Warm-up updates |
| `A10_CANONICAL_VALIDATION_INTERVAL` | `1000` | Validation interval |

Names beginning `A10_` are internal names kept from development.

## Main outputs (in the run folder)
| File | Contents |
|---|---|
| `best_model_by_val_loss.pt` | Selected checkpoint (do not publish without UK Biobank approval) |
| `training_history.csv` | Validation history |
| `test_summary_publication_with_ci.csv` | Headline test metrics with 95% CIs |
| `test_per_condition_metrics_with_ci.csv` | Per-condition metrics |
| `test_integrated_metrics_with_ci.csv` | Integrated Brier score, NBLL and calibration |
| `figure2_revised_inputs_2yr/` | Aggregate inputs for manuscript Fig. 3 |

Files containing participant-level rows (`*.pkl`, `*.npz`) must stay in approved storage.
