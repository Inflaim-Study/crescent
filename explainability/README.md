# Explanations (Figs. 4-5)

All analyses use the final seed-42 checkpoint (update 15,000) and the held-out test participants, with
five-year risk after six-month entry. They produce the aggregate CSVs that `figures/figure4/` and
`figures/figure5/` plot.

| File | Figure | Analysis |
|---|---|---|
| `figure4_shapley.py` | Fig. 4A-C | Exact Shapley values for 7 input groups (128 coalitions) |
| `figure4_lifestyle.py` | Fig. 4D | Lifestyle category substitution risk ratios; prediction-error analysis |
| `figure5_population.py` | Fig. 5a-d | Diagnosis removal, diagnosis-pair risk ratios, age ALE, look-back windows |
| `figure5_cases.py` | Fig. 5e-f | Two case studies, 8-group Shapley (256 coalitions) |

Participant-level outputs (individual contributions, case timelines) stay in approved storage.
