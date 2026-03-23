"""CLI wrapper to export Kaggle submission from saved artifacts."""

from __future__ import annotations

import argparse

from src.config import load_config
from src.inference import export_submission_from_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kaggle submission CSV")
    parser.add_argument("--config", type=str, default="configs/minilm_catboost.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = export_submission_from_artifact(
        cfg=cfg,
        artifact_path=cfg["inference"]["artifact_path"],
        output_path=cfg["inference"]["submission_path"],
    )
    print(f"Submission saved to: {output_path}")


if __name__ == "__main__":
    main()
