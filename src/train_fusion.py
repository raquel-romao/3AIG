"""Train fusion model with structured features + frozen text embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config import load_config
from src.data_io import load_train_test
from src.embeddings import encode_complaints, maybe_cache_embeddings
from src.evaluate import evaluate_multiclass
from src.features_structured import fit_transform_structured, infer_structured_columns
from src.utils import ensure_dir, make_classifier, set_seed


def train_fusion(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Train a fused model using structured preprocessing + complaint embeddings."""
    set_seed(cfg["train"].get("seed", 42))
    train_df, _ = load_train_test(cfg)

    target_col = cfg["data"]["target_col"]
    text_col = cfg["data"]["text_col"]
    id_col = cfg["data"]["id_col"]

    y_raw = train_df[target_col]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))

    tr_df, va_df, y_tr, y_va = train_test_split(
        train_df,
        y,
        test_size=cfg["train"].get("valid_size", 0.2),
        random_state=cfg["train"].get("seed", 42),
        stratify=y,
    )

    struct_cols = infer_structured_columns(
        tr_df,
        target_col=target_col,
        text_col=text_col,
        id_col=id_col,
        explicit_drop_cols=cfg["data"].get("leakage_columns", []),
    )
    struct_preprocessor, x_tr_struct, x_va_struct = fit_transform_structured(
        tr_df[struct_cols], va_df[struct_cols]
    )

    emb_cfg = cfg["embeddings"]
    model_name = emb_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = emb_cfg.get("batch_size", 64)

    x_tr_emb = encode_complaints(tr_df[text_col], model_name=model_name, batch_size=batch_size)
    x_va_emb = encode_complaints(va_df[text_col], model_name=model_name, batch_size=batch_size)

    x_tr = np.hstack([x_tr_struct, x_tr_emb])
    x_va = np.hstack([x_va_struct, x_va_emb])

    model_cfg = cfg["model"]
    model = make_classifier(
        model_name=model_cfg.get("name", "catboost"),
        params=model_cfg.get("params", {}),
        n_classes=len(label_encoder.classes_),
    )
    model.fit(x_tr, y_tr)

    y_pred = np.asarray(model.predict(x_va)).reshape(-1)
    metrics = evaluate_multiclass(y_true=y_va, y_pred=y_pred)

    # TODO: plug in undertriage metrics by acuity ordering.
    # TODO: add text-vs-structure disagreement analysis here.
    # TODO: add review-flag logic for high-risk disagreement/low confidence cases.

    out_dir = ensure_dir(Path(cfg["train"].get("artifacts_dir", "outputs/artifacts")))
    artifact_path = out_dir / "fusion_artifact.joblib"

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "structured_preprocessor": struct_preprocessor,
            "structured_columns": struct_cols,
            "embedding_model_name": model_name,
            "target_col": target_col,
            "text_col": text_col,
            "id_col": id_col,
            "feature_mode": "structured_plus_embeddings",
        },
        artifact_path,
    )

    if emb_cfg.get("cache_train_embeddings", True):
        maybe_cache_embeddings(out_dir / "train_embeddings.npy", x_tr_emb)
        maybe_cache_embeddings(out_dir / "valid_embeddings.npy", x_va_emb)

    return {
        "artifact_path": str(artifact_path),
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train triage fusion model")
    parser.add_argument("--config", type=str, default="configs/minilm_catboost.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    result = train_fusion(cfg)
    print("Fusion training complete")
    print(f"Artifact: {result['artifact_path']}")
    print(f"Metrics: {result['metrics']}")


if __name__ == "__main__":
    main()
