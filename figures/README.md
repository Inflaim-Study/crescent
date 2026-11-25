# Figures

| Folder | Manuscript figure | Contents |
|---|---|---|
| `figure1/` | Fig. 1 overview | Overview graphic: animated SVG for the README; static PNG, PDF and SVG for the paper |
| `figure2/` | Fig. 2 benchmark against eight comparators | `plot.py`, `collect_source_data.py`, `source_data/` |
| `figure3/` | Fig. 3 discrimination and calibration | `plot.py`, `build_source_data.py`, `config.py`, `source_data/` |
| `figure4/` | Fig. 4 input contributions | `plot.py`, `collect_source_data.py`, `source_data/` |
| `figure5/` | Fig. 5 population and individual explanations | `plot.py`, `collect_source_data.py`, `source_data/` |

## Redrawing a figure

Each `plot.py` redraws its figure from the aggregate tables in `source_data/`, without UK Biobank
access, and writes the files to `output/` (not tracked by git):

```bash
pip install -r requirements.txt
python figures/figure2/plot.py
python figures/figure3/plot.py
python figures/figure4/plot.py
python figures/figure5/plot.py
```

The scripts are the publication notebooks split into files, with the cells unchanged and their
numbers kept; only input and output folders were replaced (and the few lines marked
"repository"). The notebooks numbered the figures differently: their "Figure 1" is manuscript
Fig. 2 and their "Figure 2" is manuscript Fig. 3, so some internal names and output file names
still say `FIG1`/`FIG2`.

## Where the inputs come from

Model outputs live in `runs/` at the repository root (git-ignored), one folder per model:

```
runs/
├── CRESCENT_Final/            run_crescent.py output (EHR_CRESCENT_FINAL_OUTPUT), including stage 8 exports
├── cox/  xgboost/  survtrace/  dynamic_deephit/  motor_ft/
├── delphi_2m/  gemma_e4b/  crescent_pt/      comparator outputs (folder names follow comparators/)
├── explainability/figure4/    export of the Fig. 4 data generator (explainability/)
├── explainability/figure5/    export of the Fig. 5 data generator (explainability/)
└── metadata/con_abb.json      condition abbreviations
```

Set `CRESCENT_RUNS_DIR` to use a `runs/` folder kept elsewhere. Where UK Biobank data are held,
fill `source_data/` with:

| Figure | Command | Reads |
|---|---|---|
| Fig. 2 | `python figures/figure2/collect_source_data.py` | Result tables of all nine models |
| Fig. 3 | `python figures/figure3/build_source_data.py` | CRESCENT test predictions and stage 8 calibration exports; Delphi-2M metrics |
| Fig. 4 | `python figures/figure4/collect_source_data.py` | Fig. 4 export |
| Fig. 5 | `python figures/figure5/collect_source_data.py` | Fig. 5 export |

## What is published

Only aggregate, non-identifying tables go in `source_data/`. Every script stops if a table has an
`eid` column, removes machine paths from manifests and lists counts between 1 and 9 for review.
Not published:

- prediction arrays, cohort files and checkpoints;
- the threshold column of the Fig. 3 ROC table (one predicted risk per curve step);
- the Fig. 4 sampled participant–condition contributions (`fig4D_beeswarm_points.csv`), unless
  UK Biobank agrees (`--include-sampled-points`); `plot.py` needs this file for the
  individual-contribution panel;
- the Fig. 5e–f patient tables, unless `--include-synthetic-patients` is given and every profile
  is marked synthetic; without them `plot.py` draws panels a–d and the extended-data figures.
