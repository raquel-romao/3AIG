# Complaint2Concern - writeup draft

## Clinical motivation
- Why triage acuity prediction matters in emergency care.
- Why undertriage can be safety-critical.

## Project hypothesis
Chief complaint text contains concern signals that can complement structured intake variables.

## Data and leakage policy
- Use triage-time variables only.
- Exclude post-triage outcomes and downstream care artifacts.

## Modeling
- Structured baseline.
- TF-IDF baseline.
- Frozen embedding fusion model.

## Evaluation
- Accuracy, macro-F1, confusion matrix, off-by-1.
- TODO: add undertriage and severe undertriage metrics.

## Analysis extensions
- TODO: complaint clustering.
- TODO: text-vs-structure disagreement analysis.
- TODO: review-flag logic for high-risk disagreement cases.

## Limitations
- Synthetic/competition setting vs real-world deployment.
- External validity and fairness checks.
