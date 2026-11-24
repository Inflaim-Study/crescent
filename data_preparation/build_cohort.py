"""Step 2: build histories and outcomes.

Manuscript: §2.3

What it does:
    - Map Read v2 / SNOMED CT codes to the 60 InflAIM conditions (codelists v2.0)
    - Keep the first diagnosis per condition from age 18
    - Treat death and the two malignant-neoplasm categories as absorbing events
    - Add timed BMI, smoking, alcohol and CRP records; static age, sex, ethnicity, Townsend

Inputs:
    - Step 1 extracts
    - InflAIM codelists

Outputs:
    - One row per participant with event arrays
"""

# TODO: add code
