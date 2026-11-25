# CRESCENT model details

This page summarises the model as described in the manuscript (§2.3–2.5) and links it to the code:
`crescent/` (stages 1–5) and `evaluation/` (stages 6–8). The symbols follow the manuscript.

## Prediction task

| Item | Definition |
|---|---|
| Prediction time | The first UK Biobank recruitment assessment. Predictors are fixed at this point. |
| Lead period | The first six months after recruitment. Risks start at month 6, to reduce the influence of disease already present but not yet recorded. |
| Who is at risk | For each condition separately: participants alive, still under observation and with no record of that condition by month 6. A previous diagnosis stays in the history; the participant is simply not at risk for that condition. |
| Outcome | The first recorded diagnosis of the condition in primary care, before death. |
| Competing event | Death, from linked death registrations. |
| Time grid | 22 six-month intervals from recruitment (0–132 months). Interval 0 is the lead period. |
| Horizons | 12, 24, 60 and 120 months after entry (months 18, 30, 66 and 126 after recruitment). |
| End of follow-up | Primary-care data cut-off: 31 May 2016 (England), 31 March 2017 (Scotland), 31 August 2017 (Wales). |
| Conditions | 58 of the 60 prespecified InflAIM conditions: those with at least 100 cases in the training partition. |

## Inputs

| Type | Inputs |
|---|---|
| Fixed characteristics | Age, recorded sex, ethnicity and Townsend deprivation: quintile (as a category) and standardised score (missing values set to the training median, with a missing-value flag) |
| Dated history (up to 25 records) | Adult diagnoses, BMI category, smoking status, alcohol-intake frequency and CRP value, each with the age at which it was recorded |

If a participant has more than 25 earlier records (0.22% of participants), the most recent are kept, and a CRP
record is retained if it would otherwise be dropped. Death and two cancer categories end a participant's diagnosis
sequence: no later condition record is used. All scaling and category codes are learned from the training
partition and applied unchanged to validation and test.

## How the model works

### 1. Encoding the participant

The fixed characteristics are combined into one vector, $`s_i`$. Each dated record $`j`$ is turned into a vector
that adds what the record is, its type, the age when it was recorded, its value (for CRP) and the participant's
fixed characteristics:

```math
h^{0}_{ij} = e_{v_{ij}} + e_{r_{ij}} + \varphi(a_{ij}) + m_{ij}\,x_{ij}\,d_{v_{ij}} + c_{ij} + s_i
```

| Symbol | Meaning |
|---|---|
| $`e_{v_{ij}}`$, $`e_{r_{ij}}`$ | Learned vectors for the record's label (e.g. a diagnosis) and its type |
| $`\varphi(a_{ij})`$ | Encoding of the age at which the record was made |
| $`m_{ij}\,x_{ij}\,d_{v_{ij}}`$ | The CRP value scaled by a learned vector; omitted when CRP is not measured ($`m_{ij} = 0`$) |
| $`c_{ij}`$ | An extra CRP term that lets the effect of CRP bend across its range (set by training-data quantiles; starts at zero) |

A 3-layer Transformer (width 128, 16 attention heads, feed-forward width 512, dropout 0.20) then lets every record
take account of the others. A summary token placed after the history gives one vector for the whole history, $`g_i`$.

### 2. Reading the history separately for each condition

For each condition $`k`$, the model forms a question ("query") from a learned condition vector, the fixed
characteristics and the history summary:

```math
q_{ik} = e^{C}_{k} + f_s(s_i) + f_g(g_i)
```

The query attends to the encoded records. A learned adjustment changes how much weight each record gets,
depending on the condition, the record, its type and how long ago it was recorded:

```math
b_{ikhj} = \psi_h\!\left(e^{C}_{k},\; e(v_{ij}),\; e(r_{ij}),\; \log\!\left[1 + \frac{\max(t_{i0} - a_{ij},\,0)}{12}\right]\right)
```

Here $`t_{i0} - a_{ij}`$ is the time in months between the record and the prediction time, and $`h`$ is the attention
head. So the same past diagnosis can matter more for one condition than another, and more if it is recent.

### 3. Sharing information across conditions

The 58 condition-specific summaries then pass through an attention block that lets them inform one another. Its
gates start at zero, so this sharing is learned only if it helps.

### 4. Condition and death rates

For participant $`i`$, condition $`k`$ and six-month interval $`u`$, the log of the monthly rate is a sum of five terms:

```math
\eta^{C}_{iku} = b^{C}_{k} + r^{\mathrm{stat}}_{iku} + r^{\mathrm{hist}}_{iku} + r^{\mathrm{shared}}_{iu} + a_{iku}
```

| Term | Meaning |
|---|---|
| $`b^{C}_{k}`$ | Condition intercept, started at the training rate of the condition |
| $`r^{\mathrm{stat}}_{iku}`$ | Effect of the fixed characteristics |
| $`r^{\mathrm{hist}}_{iku}`$ | Effect of the history, from steps 2–3 |
| $`r^{\mathrm{shared}}_{iu}`$ | A participant-level term from the history summary $`g_i`$, added equally to every condition |
| $`a_{iku}`$ | Effect of attained age (age at recruitment plus time in follow-up), separately by condition and sex |

Death has its own rate, based on the fixed characteristics and the history summary:

```math
\eta^{D}_{iu} = b_D + r^{D}_{iu}
```

Both are turned into monthly rates in the same way, with limits that keep the numbers stable:

```math
\lambda = \max\lbrace 10^{-8},\; \exp(\mathrm{clip}(\eta,\,-18,\,2))\rbrace
```

### 5. From rates to risk

In each six-month interval ($`\Delta = 6`$ months), with total rate $`\Lambda_{iku} = \lambda^{C}_{iku} + \lambda^{D}_{iu}`$,
a participant who starts the interval alive and free of condition $`k`$ either is diagnosed, dies, or neither:

```math
p^{C}_{iku} = \frac{\lambda^{C}_{iku}}{\Lambda_{iku}}\left(1 - e^{-\Lambda_{iku}\Delta}\right), \qquad
p^{D}_{iku} = \frac{\lambda^{D}_{iu}}{\Lambda_{iku}}\left(1 - e^{-\Lambda_{iku}\Delta}\right), \qquad
p^{\varnothing}_{iku} = e^{-\Lambda_{iku}\Delta}
```

The chance of still being alive and free of the condition after interval $`u`$, and the cumulative risk of the
condition with death taken into account, are:

```math
S_{ik}(u) = \prod_{m=0}^{u} p^{\varnothing}_{ikm}, \quad S_{ik}(-1) = 1, \qquad
F^{C}_{ik}(u) = \sum_{m=0}^{u} S_{ik}(m-1)\, p^{C}_{ikm}
```

Reported risks start at month 6 and apply to people still alive and free of the condition then. For a horizon of
$`h`$ months after entry:

```math
R^{C}_{ik}(h \mid \text{entry}) = \frac{F^{C}_{ik}(h/6) - F^{C}_{ik}(0)}{S_{ik}(0)}
```

## Training

| Setting | Value |
|---|---|
| What is optimised | The likelihood of the observed diagnoses, deaths and follow-up times from month 6 onwards (piecewise-constant rates, partial last interval included): $`L = L_C + 0.10\,L_D`$, condition part averaged over at-risk participant–condition pairs, death part over participants. No other loss terms. |
| Optimiser | AdamW, batch 256, learning rate 1.2 × 10⁻⁴, weight decay 0.05 (not applied to scalar and vector parameters) |
| Regularisation | Dropout 0.20; 10% of history records hidden at random during training; gradient norm capped at 1.0 |
| Schedule | 25,000 updates: 1,000 warm-up, then cosine decay to 5% of the starting learning rate |
| Model selection | Validation loss every 1,000 updates, computed on a running average of the weights (decay 0.999); minimum improvement 10⁻⁴ |
| Published fit | Seed 42; 1,194,593 parameters; selected at update 15,000 (validation loss 0.153281) |
| Software | Python 3.11.4, Polars 1.37.1 (data preparation), PyTorch 2.6.0, CUDA 12.4; seeds fixed and deterministic GPU settings where supported |

## Running

```bash
export CRESCENT_PREPARED_DATA_DIR=/path/to/splits_229950_one_month_cap
export EHR_CRESCENT_FINAL_OUTPUT=runs/CRESCENT_Final     # must be a new, empty folder
python run_crescent.py              # all stages
python run_crescent.py --list       # show stages
```

A single CUDA GPU visible as `cuda:0` is required.

## Settings (environment variables)

The defaults reproduce the published model.

| Variable | Default | Meaning |
|---|---|---|
| `CRESCENT_PREPARED_DATA_DIR` | `splits_229950_one_month_cap` | Folder with the prepared data partitions |
| `EHR_CRESCENT_FINAL_OUTPUT` | `runs/CRESCENT_Final` | Output folder (must be new) |
| `CRESCENT_SEED` | `42` | Random seed |
| `CRESCENT_N_HEADS` | `16` | Attention heads |
| `CRESCENT_BATCH_SIZE` | `256` | Batch size |
| `CRESCENT_LR` | `1.2e-4` | Learning rate |
| `CRESCENT_WEIGHT_DECAY` | `5e-2` | Weight decay |
| `CRESCENT_WD_EXEMPT_1D` | `1` | No weight decay on scalar and vector parameters |
| `CRESCENT_DROPOUT` | `0.20` | Dropout |
| `CRESCENT_EVENT_MASK_PROB` | `0.10` | Share of history records hidden during training |
| `CRESCENT_EMA_DECAY` | `0.999` | Running-average weights used for selection (0 turns this off) |
| `CRESCENT_ATTAINED_AGE_TERM` | `1` | Attained-age term (0 = ablation A1) |
| `CRESCENT_READER_TEMPORAL_BIAS` | `1` | Time-dependent weighting of history records |
| `A10_USE_CONDITION_IDENTITY_PATH` | `1` | Condition identity pathway |
| `CRESCENT_STATIC_CONT_LAYERNORM` | `0` | Normalisation of continuous fixed inputs |
| `CRESCENT_TIME_SMOOTHNESS_WEIGHT` | `0.0` | Optional smoothness penalty (off) |
| `CRESCENT_MIN_TRAIN_PARTICIPANTS_PER_CONDITION` | `100` | Minimum training cases for a condition to be modelled |
| `CRESCENT_CONDITION_LIST` | (empty) | Reuse an earlier run's condition list |
| `CRESCENT_ENSEMBLE_RUN_DIRS` | (empty) | Optional average over several seeds |
| `A10_CANONICAL_MAX_ITERATIONS` | `25000` | Training updates |
| `A10_CANONICAL_WARMUP_ITERATIONS` | `1000` | Warm-up updates |
| `A10_CANONICAL_VALIDATION_INTERVAL` | `1000` | Updates between validation checks |

Names beginning `A10_` are internal names kept from development.

## Main outputs (in the run folder)

| File | Contents |
|---|---|
| `best_model_by_val_loss.pt` | Selected model weights (not to be shared without UK Biobank approval) |
| `training_history.csv` | Validation history |
| `test_summary_publication_with_ci.csv` | Headline test results with 95% confidence intervals |
| `test_per_condition_metrics_with_ci.csv` | Results for each condition |
| `test_integrated_metrics_with_ci.csv` | Integrated Brier score, log-likelihood and calibration |
| `figure2_revised_inputs_2yr/` | Summary tables for manuscript Fig. 3 |

Files with one row per participant (`*.pkl`, `*.npz`) must stay in approved storage.
