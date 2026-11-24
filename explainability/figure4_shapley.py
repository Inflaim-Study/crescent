"""Exact group Shapley values.

Manuscript: §2.8.1, Fig. 4A-C

What it does:
    - Groups: diagnosis history, landmark age, BMI, smoking, alcohol, CRP, Townsend; sex and ethnicity held as observed
    - References: delete diagnoses; age 58.44 years with event ages shifted; normal BMI; never smoking; alcohol 1-2 times/week; CRP 1.37 mg/L; Townsend z = -0.284
    - Enumerate all 128 coalitions; check contributions sum to observed minus all-reference risk
    - Summaries over 58 conditions; up to 200 sampled pairs per condition for panel B; 200 bootstrap resamples

Inputs:
    - Final checkpoint
    - Test participants

Outputs:
    - figures/figure4/source_data/*.csv
"""

# TODO: add code
