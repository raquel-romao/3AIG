# Complaint2Concern: Chief Complaint as a Second Clinical Lens in Emergency Triage

A cross-validated, safety-focused analysis of structured intake, TF-IDF, and frozen MiniLM complaint embeddings for triage acuity prediction.

## Clinical Problem Statement

Emergency triage is a high-stakes workflow that assigns urgency under uncertainty. In this setting, errors are not equally harmful: undertriage, meaning assigning lower urgency than clinically needed, can delay care for patients at higher risk of deterioration.

Most triage workflows combine structured intake variables, such as vitals, demographics, and history, with free-text chief complaint. In practice, chief complaint text is clinically rich but difficult to standardize, and its contribution is often under-modeled in structured prediction workflows.

This project targets a specific workflow gap: whether chief complaint text carries acuity-relevant signal beyond structured intake fields, and whether disagreement between text-based and structured predictions can identify cases that should be reviewed more carefully by clinicians.

The intended beneficiaries are triage nurses, ED clinicians, and patients with high-acuity presentations where delayed recognition has the greatest safety consequence. This is a decision-support prototype, not an autonomous triage replacement.

## Approach and Methodology

### Data and Target

I used the Kaggle Triagegeist competition files and merged records by `patient_id` from:

- `train.csv`
- `test.csv`
- `chief_complaints.csv`, using `chief_complaint_raw`
- `patient_history.csv`

The target variable is `triage_acuity`, where Level 1 is most urgent and Level 5 is least urgent.

### Leakage Guardrails

Only triage-time information was allowed in model features. Post-triage variables, including disposition, ED length of stay, downstream tests, and other variables only known after triage, were excluded from model inputs to reduce leakage risk.

Because the text-only models achieved extremely high internal cross-validation performance, I treated lexical shortcut and near-duplicate wording risk as an explicit limitation rather than assuming that internal CV performance directly implies real-world generalization.

### Preprocessing

- Structured numeric features: median imputation and scaling
- Structured categorical features: most-frequent imputation and one-hot encoding
- Text: lowercase normalization, punctuation and whitespace cleanup, and abbreviation normalization
- Missing complaint text: empty-string fallback

### Models Compared

- Structured-only CatBoost
- Text-only TF-IDF plus Logistic Regression
- Structured plus TF-IDF plus Logistic Regression
- Text-only MiniLM embeddings plus LightGBM
- Structured plus MiniLM embeddings, using early fusion with CatBoost

Fusion was implemented by concatenating structured feature vectors and frozen text embeddings before fitting the downstream tabular model.

### Validation and Metrics

I used 5-fold cross-validation with out-of-fold predictions for robust model comparison.

Primary metrics were:

- Accuracy
- Macro-F1
- Off-by-1 accuracy
- Undertriage rate, defined as `prediction > true` because Level 1 is most urgent
- Severe undertriage, defined as `prediction - true >= 2`
- Recall for high acuity, defined as acuity levels 1-2

### Safety and Interpretability Analyses

The safety layer focused on a text-vs-structure concern gap:

`P_text(acuity 1-2) - P_structured(acuity 1-2)`

Cases were flagged for review when model confidence was low and/or the chief complaint model suggested substantially higher acuity than the structured-only model.

I also evaluated robustness to simple text perturbations, including typo-style perturbations and shortened chief complaints.

Interpretability analyses included global feature importance across folds, class-specific TF-IDF terms, modality contribution for mixed models, and embedding semantic anchors that link high-importance embedding dimensions to representative complaint terms and examples.

## Results and Findings

### Core Model Performance

In this run, text-driven models performed best under out-of-fold evaluation:

| Model | Accuracy | Macro-F1 | Undertriage |
|---|---:|---:|---:|
| Text-only TF-IDF | ~0.9994 | ~0.9980 | ~0.000638 |
| Text-only MiniLM embeddings | ~0.9993 | ~0.9977 | ~0.000650 |
| Structured plus TF-IDF | strong | ~0.9947 | low |
| Structured plus MiniLM fusion | strong | ~0.9921 | low |
| Structured-only | ~0.8530 | ~0.8686 | ~0.0798 |

### Clinical Interpretation

In this dataset, chief complaint text appears to contain strong acuity signal and materially improves safety-oriented metrics relative to structured-only modeling. In particular, undertriage was substantially lower in text-dominant models.

Fusion remained strong but did not outperform text-only models in this run. This suggests that the chief complaint field captured much of the most discriminative signal under the current synthetic data conditions.

### Safety-Layer Findings

Using concern-gap and confidence thresholds:

- Review flag rate: ~0.1912
- High-acuity share among flagged cases: ~0.5779

This supports a practical decision-support use case: flagging a smaller subset of cases enriched for potentially urgent presentations for enhanced human review.

### Robustness

- Typo perturbation stability: ~0.9640
- Shortened complaint stability: ~0.9365

These checks suggest that the text models were reasonably stable to simple complaint perturbations, while still requiring stronger robustness testing before any real-world use.

### Subgroup Behavior

Subgroup snapshots by age, sex, and site showed broadly consistent performance with modest variation. These results support preliminary reliability confidence, but they are not sufficient for deployment-level fairness claims.

### Interpretability Findings

Cross-fold importance identified stable structured and text drivers. TF-IDF class-term analysis provided direct lexical interpretation of class-specific complaint signals. Modality-contribution analysis quantified structured-versus-text influence in mixed models. Embedding semantic anchors improved readability of dense embedding features by linking important dimensions to representative complaint terms and examples.

## Code and Artifact Links

Public Kaggle notebook: TODO: add Kaggle notebook URL
GitHub repository: TODO: add GitHub repository URL
Competition page: https://www.kaggle.com/competitions/triagegeist

Executable notebook:

- [`notebooks/submission_notebook_kaggle.ipynb`](../notebooks/submission_notebook_kaggle.ipynb)

Final prediction artifact:

- [`submission.csv`](../submission.csv)

Core implementation files:

- Data loading and merge: [`src/data_io.py`](../src/data_io.py)
- Text preprocessing: [`src/preprocess.py`](../src/preprocess.py)
- TF-IDF features: [`src/features_text.py`](../src/features_text.py)
- MiniLM embeddings: [`src/embeddings.py`](../src/embeddings.py)
- Structured features: [`src/features_structured.py`](../src/features_structured.py)
- Model utilities: [`src/utils.py`](../src/utils.py)
- Inference/export: [`src/inference.py`](../src/inference.py)
- Main configs: [`configs/minilm_catboost.yaml`](../configs/minilm_catboost.yaml), [`configs/base.yaml`](../configs/base.yaml)

The notebook is the authoritative executable artifact. It reproduces the cross-validated analysis and exports a Kaggle-format prediction file from a clean run when the competition data and project source files are available.

## Limitations and Future Work

The dataset is synthetic, so real ED documentation and workflow variability may be broader than represented here. Strong internal CV performance does not establish external validity across hospitals, regions, time periods, or documentation styles.

Very high text-only performance may partly reflect recurring complaint wording patterns, lexical shortcuts, or near-duplicate synthetic phrasing. This is why group-site validation, temporal validation, and stronger negative controls should be prioritized before interpreting the model as clinically generalizable.

The review-flag thresholds are heuristic and would require calibration under real workflow constraints to balance sensitivity with alert burden. Fairness analysis is also preliminary: subgroup snapshots are useful but not enough for deployment-level equity claims.

Future work priorities include:

- Group-site and temporal validation as primary generalization checks
- Expanded shortcut-risk controls and stronger negative controls
- Calibration optimization and threshold selection under workflow constraints
- Clinician-reviewed mismatch case analysis
- Prospective simulation of triage-assist impact on safety and throughput

## Reproducibility Notes

- Data: Kaggle Triagegeist files, including `train.csv`, `test.csv`, `chief_complaints.csv`, and `patient_history.csv`
- Notebook: `notebooks/submission_notebook_kaggle.ipynb`
- Submission artifact: `submission.csv`
- Environment: Python 3.11 with `pandas`, `numpy`, `scikit-learn`, `scipy`, `catboost`, `lightgbm`, `sentence-transformers`, and `pyyaml`
- Random seed: 42
- Validation: 5-fold cross-validation with OOF aggregation
- Hardware: CPU-compatible; embedding stages are slower but feasible on a standard laptop
