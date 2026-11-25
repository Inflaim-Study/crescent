"""Fig. 3 settings shared by `build_source_data.py` and `plot.py`.

Notebook cell 1 of `Figure2_CRESCENT_publication` without its machine-specific paths
(the notebook calls this "Figure 2"; it is Fig. 3 in the manuscript). Names that start
`FIG2_` were kept for fidelity.
"""

# =============================================================================
# Notebook cell 1 (settings only; paths are set in each script)
# =============================================================================

FIG2_TARGET_MODEL = "CRESCENT"
FIG2_COMPARATOR_MODEL = "Delphi-2M"

# Clinical chapter assignments supplied in Con_chapters__V3.csv.
# The source has 60 target conditions plus Death; Death has its own
# mortality head and is excluded from the disease colour key.
FIG2_CHAPTER_CROSSWALK = (
    ('Hypertensive diseases', 'Chapter IX', 'Circulatory diseases'),
    ('Dermatitis and eczema', 'Chapter XII', 'Skin diseases'),
    ('Rhinitis and Sinusitis', 'Chapter X', 'Respiratory diseases'),
    ('Ischaemic heart diseases', 'Chapter IX', 'Circulatory diseases'),
    ('Osteoarthritis', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Diseases of oesophagus, stomach and duodenum', 'Chapter XI', 'Digestive diseases'),
    ('Chronic lower respiratory diseases', 'Chapter X', 'Respiratory diseases'),
    ('Neurotic, stress-related and somatoform disorders', 'Chapter V', 'Mental disorders'),
    ('Malignant neoplasms', 'Chapter II', 'Neoplasms'),
    ('Spinal degenerative disease', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Conductive and sensorineural hearing loss', 'Chapter VIII', 'Ear diseases'),
    ('Diseases of veins, lymphatic vessels and lymph nodes, not elsewhere classified', 'Chapter IX', 'Circulatory diseases'),
    ('Diabetes', 'Chapter IV', 'Metabolic diseases'),
    ('Mood disorders', 'Chapter V', 'Mental disorders'),
    ('Migraine', 'Chapter VI', 'Nervous system diseases'),
    ('Irritable bowel syndrome', 'Chapter XI', 'Digestive diseases'),
    ('Disorders of thryoid gland', 'Chapter IV', 'Metabolic diseases'),
    ('Inflammatory polyarthropathies', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Diverticular disease of the intestine', 'Chapter XI', 'Digestive diseases'),
    ('Chronic kidney disease', 'Chapter XIV', 'Genitourinary diseases'),
    ('Inflammatory diseases of the female pelvic organs', 'Chapter XIV', 'Genitourinary diseases'),
    ('Iron deficiency anaemia', 'Chapter III', 'Blood and immune disorders'),
    ('Disorders of bone density and structure', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Other forms of heart disease', 'Chapter IX', 'Circulatory diseases'),
    ('Cataract', 'Chapter VII', 'Eye diseases'),
    ('Psoriasis', 'Chapter XII', 'Skin diseases'),
    ('Disorders of the prostate', 'Chapter XIV', 'Genitourinary diseases'),
    ('Polyneuropathies and other disorders of the peripheral nervous system', 'Chapter VI', 'Nervous system diseases'),
    ('Chronic viral hepatitis', 'Chapter I', 'Infectious diseases'),
    ('Cerebrovascular disease', 'Chapter IX', 'Circulatory diseases'),
    ('Disorders of gallbladder, biliary tract and pancreas', 'Chapter XI', 'Digestive diseases'),
    ('Glaucoma', 'Chapter VII', 'Eye diseases'),
    ('Diseases of arteries, arterioles and capillaries', 'Chapter IX', 'Circulatory diseases'),
    ('Ankylosing spondylitis', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Sleep disorders', 'Chapter VI', 'Nervous system diseases'),
    ('Systemic connective tissue disorders', 'Chapter XIII', 'Musculoskeletal diseases'),
    ('Noninfective enteritis and colitis', 'Chapter XI', 'Digestive diseases'),
    ('Epilepsy', 'Chapter VI', 'Nervous system diseases'),
    ('Malignant neoplasms, stated or presumed to be primary, of lymphoid, haematopoietic and related tissue', 'Chapter II', 'Neoplasms'),
    ('Diseases of the liver', 'Chapter XI', 'Digestive diseases'),
    ('Drug or alcohol misuse', 'Chapter V', 'Mental disorders'),
    ('In situ neoplasms', 'Chapter II', 'Neoplasms'),
    ('Other reintal disorders (Macular degeneration)', 'Chapter VII', 'Eye diseases'),
    ('Tuberculosis', 'Chapter I', 'Infectious diseases'),
    ('Retinal detachments and breaks', 'Chapter VII', 'Eye diseases'),
    ('Visual disturbances and blindness', 'Chapter VII', 'Eye diseases'),
    ('Dementia', 'Chapter V', 'Mental disorders'),
    ('Endometriosis', 'Chapter XIV', 'Genitourinary diseases'),
    ('Multiple sclerosis (and optic neuritis)', 'Chapter VI', 'Nervous system diseases'),
    ('Parkinson disease', 'Chapter VI', 'Nervous system diseases'),
    ('Meniere disease', 'Chapter VIII', 'Ear diseases'),
    ('Schizophrenia', 'Chapter V', 'Mental disorders'),
    ('Other interstitial pulmonary diseases', 'Chapter X', 'Respiratory diseases'),
    ('Eating disorders', 'Chapter V', 'Mental disorders'),
    ('Hyperparathyroidism', 'Chapter IV', 'Metabolic diseases'),
    ('HIV disease', 'Chapter I', 'Infectious diseases'),
    ('Other spirochaetal infections', 'Chapter I', 'Infectious diseases'),
    ("Addison's disease", 'Chapter IV', 'Metabolic diseases'),
    ('Hypoparathyroidism', 'Chapter IV', 'Metabolic diseases'),
    ('Chronic rheumatic heart diseases', 'Chapter IX', 'Circulatory diseases'),
)
FIG2_CHAPTER_CROSSWALK_PROVENANCE = "Con_chapters__V3.csv (embedded; Death excluded)"

# Confirm this encoding against the source-data preprocessing. The notebook
# fails if either binary sex stratum is absent, but it cannot detect a swap.
GENDER_INDEX_TO_SEX = {
    0: "Female",
    1: "Male",
    2: "Unknown",
}

# Fixed ICD-chapter colours used in both the exported key and every plot.
# Chapters I to XIV were checked for separation under normal vision and
# simulated red-green colour blindness, and for at least 3:1 contrast on white.
# Hex codes must stay UPPER CASE (the plotting cell compares them with the
# cached colour key). After editing, re-run the input-build cell.
ICD_CHAPTER_COLORS = {
    1: "#7D9A5F",   # Infectious diseases         sage green
    2: "#5B2671",   # Neoplasms                   deep purple
    3: "#902047",   # Blood and immune disorders  wine
    4: "#6B3E03",   # Metabolic diseases          brown
    5: "#0F449A",   # Mental disorders            navy
    6: "#9F51A4",   # Nervous system diseases     orchid
    7: "#005656",   # Eye diseases                dark teal
    8: "#6E6975",   # Ear diseases                slate grey
    9: "#C43F3E",   # Circulatory diseases        red
    10: "#1AA0B9",  # Respiratory diseases        cyan
    11: "#D0750A",  # Digestive diseases          orange
    12: "#DB6EA5",  # Skin diseases               pink
    13: "#9088F0",  # Musculoskeletal diseases    periwinkle
    14: "#27762F",  # Genitourinary diseases      green

    # Chapters XV to XXII do not occur in this dataset. The keys are kept
    # (the plotting cell requires all 22) with one neutral placeholder; the
    # plotting cell stops if a plotted condition ever maps to one of them.
    15: "#B0B0B0",
    16: "#B0B0B0",
    17: "#B0B0B0",
    18: "#B0B0B0",
    19: "#B0B0B0",
    20: "#B0B0B0",
    21: "#B0B0B0",
    22: "#B0B0B0",
}
FIG2_PRIMARY_HORIZON_MONTHS = 24
FIG2_HORIZONS_MONTHS = [12, 24, 60, 120]
FIG2_AGE_BIN_WIDTH_YEARS = 5
FIG2_MIN_AGE_BIN_ELIGIBLE = 10
FIG2_SELECTED_CONDITIONS = (
    "Cerebrovascular disease",
    "Chronic kidney disease",
    "Chronic lower respiratory diseases",
    "Dementia",
    "Dermatitis and eczema",
    "Diabetes",
    "Hypertensive diseases",
    "Ischaemic heart diseases",
    "Inflammatory polyarthropathies",
    "Mood disorders",
)
FIG2_N_SELECTED_ENDPOINTS = len(FIG2_SELECTED_CONDITIONS)
PRIMARY_RISK_TYPE = "death_adjusted"
