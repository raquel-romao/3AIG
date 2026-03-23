#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Training structured baseline..."
python -m src.train_baseline --config configs/base.yaml

echo "[2/3] Training fusion model (MiniLM + CatBoost)..."
python -m src.train_fusion --config configs/minilm_catboost.yaml

echo "[3/3] Exporting fusion submission..."
python scripts/export_submission.py --config configs/minilm_catboost.yaml

echo "Done. Check outputs/ for artifacts and submission CSVs."
