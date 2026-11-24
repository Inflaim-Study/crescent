# Comparator models

Summaries of manuscript §2.6 and Supplementary Methods S1.

## Penalised Cox (`comparators/cox/`)
Cause-specific ridge-penalised Cox models on full-history summaries and static predictors; ridge penalty chosen by validation Harrell concordance; training Breslow hazards give cumulative incidence with competing death (§2.6.1).

## XGBoost (`comparators/xgboost/`)
Histogram boosted trees with the Cox partial-likelihood objective, using the same features and risk sets as Cox; training Breslow hazards (§2.6.2).

## SurvTRACE-aligned (`comparators/survtrace/`)
Feature Transformer over static feature tokens and condition-history summaries; condition-specific piecewise-constant intensities and a shared death intensity (§2.6.4).

## Dynamic-DeepHit-inspired (`comparators/dynamic_deephit/`)
Unidirectional GRU with temporal attention over the event sequence; condition-specific history attention and one shared death head (§2.6.3).

## MOTOR-FT (`comparators/motor_ft/`)
MOTOR architecture and time-to-event objective, pretrained locally on the study data, then fine-tuned to the fixed-index task (§2.6.6).

## Delphi-2M-inspired (`comparators/delphi_2m/`)
Autoregressive causal Transformer predicting the next event and its timing; risks estimated from simulated trajectories eligible at month 6 (§2.6.5).

## Frozen Gemma 4 E4B (`comparators/gemma_e4b/`)
Cached final-layer states of a frozen Gemma 4 E4B encoder over structured text histories, projected 2,560 to 128, with a trainable survival decoder (§2.6.7).

## CRESCENT-PT (`comparators/crescent_pt/`)
CRESCENT pretrained on time to the first subsequent diagnosis or death at training-partition cutpoints, then transferred to the fixed-recruitment task (§2.6.8).

## Interpretation
These are local implementations adapted to a common cohort, eligibility rules and evaluation. Results for MOTOR-FT and the Delphi-2M-inspired model describe these local versions, not the released original systems.
