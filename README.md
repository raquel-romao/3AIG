# Triagegeist - Complaint2Concern

Lightweight, clinically grounded starter project for the Kaggle **Triagegeist** emergency-triage challenge.

Core idea: treat chief complaint text as a second clinical signal and compare it against structured triage inputs to detect potential under-concern from structured features alone.

## Project goals
- Predict triage acuity using triage-time data only.
- Build reproducible local + Kaggle-compatible pipelines.
- Compare structured-only, text-only, and fused models.
- Keep training feasible on a normal laptop.

## Repository layout
```text
.
├── configs/
├── data/
├── kaggle/
├── notebooks/
├── outputs/
├── reports/
├── scripts/
└── src/
```

## Quick start
1. Create environment:
   - Conda: `conda env create -f environment.yml && conda activate triagegeist`
   - Pip: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Put Kaggle CSV files in `data/raw/` for local runs.
3. Run baseline: `python -m src.train_baseline --config configs/base.yaml`
4. Run fusion: `python -m src.train_fusion --config configs/minilm_catboost.yaml`
5. Export test predictions: `python scripts/export_submission.py --config configs/minilm_catboost.yaml`

## Leakage guardrail
Only use variables available at triage time. Do **not** include post-triage outcomes (e.g., length of stay, disposition, later tests).

## Notes
- Keep notebooks thin; business logic belongs in `src/`.
- Final Kaggle notebook should call `src` functions and run end-to-end.


