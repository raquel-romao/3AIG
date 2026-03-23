# Kaggle notebook outline

## 1) Clinical framing
- Triage acuity prediction as a triage-time decision support task.
- Safety emphasis: undertriage awareness.

## 2) Data loading and leakage guardrails
- Load train/test from `/kaggle/input/...`.
- Exclude post-triage variables.

## 3) Baselines
- Structured-only model.
- TF-IDF text baseline.

## 4) Main model
- Frozen MiniLM complaint embeddings.
- Structured + embedding fusion with CatBoost/LightGBM.

## 5) Evaluation
- Accuracy, macro-F1, confusion matrix, off-by-1.
- TODO: undertriage metrics.

## 6) Analysis
- TODO: complaint clustering from embedding space.
- TODO: text-vs-structure disagreement analysis.
- TODO: review flag logic.

## 7) Inference and submission
- Load best artifact.
- Export `submission.csv`.
