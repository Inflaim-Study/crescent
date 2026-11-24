# Figures

| Folder | Manuscript figure | Contents |
|---|---|---|
| `figure1/` | Fig. 1 overview | Overview graphic (PNG for the README, PDF and SVG masters) |
| `figure2/` | Fig. 2 benchmark | `plot.py` + `source_data/` |
| `figure3/` | Fig. 3 discrimination and calibration | `plot.py` + `source_data/` |
| `figure4/` | Fig. 4 input contributions | `plot.py` + `source_data/` |
| `figure5/` | Fig. 5 population and individual explanations | `plot.py` + `source_data/` |

Each `plot.py` redraws its figure from the aggregate CSVs in `source_data/`, without UK Biobank access:

```bash
python figures/figure2/plot.py
```

Only aggregate, non-identifying tables go in `source_data/`. Participant-level inputs (for example the
Fig. 5e-f case timelines) stay in UK Biobank-approved storage.
