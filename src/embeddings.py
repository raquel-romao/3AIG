"""Frozen embedding helpers using sentence-transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.preprocess import preprocess_text_series


def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load a sentence-transformer encoder for complaint embeddings."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise ImportError(
            "sentence-transformers is required for embedding features. "
            "Install it with: pip install sentence-transformers"
        ) from exc
    return SentenceTransformer(model_name)


def encode_complaints(
    texts: Iterable[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    backend: str | None = None,
    normalize_embeddings: bool = True,
    **_: object,
) -> np.ndarray:
    """Generate frozen embeddings for complaint text.

    The current implementation uses sentence-transformers.
    The ``backend`` argument is accepted for compatibility with notebook code
    that may pass backend selectors (for example ``backend='sentence_transformers'``).
    """
    embedder = load_embedder(model_name=model_name)
    clean_text = preprocess_text_series(texts)
    embs = embedder.encode(
        clean_text,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=False,
    )
    return embs.astype("float32")


def maybe_cache_embeddings(cache_path: str | Path, embeddings: np.ndarray) -> Path:
    """Save embeddings to disk for reproducibility and speed."""
    out = Path(cache_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, embeddings)
    return out
