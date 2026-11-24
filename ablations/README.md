# Architecture ablations (Table 2)

Seven configurations, each trained with seeds 42, 101 and 202 (21 fits), using the same data, objective and
checkpoint rule as CRESCENT (manuscript §2.9).

| ID | Change | How it is switched |
|---|---|---|
| M1 | CRESCENT_Final (reference) | defaults |
| V1 | Linear-only CRP embedding (piecewise-linear residual fixed at zero) | TODO |
| A1 | No explicit attained-age term | `CRESCENT_ATTAINED_AGE_TERM=0` (check) |
| P1 | No event history or shared state in the disease rates | TODO |
| P2 | P1, and no event history in the mortality pathway | TODO |
| R1 | Condition-independent history reader | TODO (possibly `A10_USE_CONDITION_IDENTITY_PATH=0`) |
| T1xR3 | No historical-event age encoding and no reader recency | TODO |

`run_ablations.py` loops over configurations and seeds. Results go to `results/`:
per-seed, per-condition metrics and the three-seed macro summaries for Tables 2a-2c.
