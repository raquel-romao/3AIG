"""Text feature builders for chief complaint modeling."""

from __future__ import annotations

from typing import Iterable, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import preprocess_text_series


def build_tfidf_vectorizer(max_features: int = 20000, ngram_range: tuple[int, int] = (1, 2)) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer for complaint text."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        strip_accents="unicode",
    )


def fit_transform_tfidf(
    train_text: Iterable[str],
    valid_text: Iterable[str] | None = None,
    max_features: int = 20000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Tuple[TfidfVectorizer, object, object | None]:
    """Fit TF-IDF on train text and transform train/(optional) valid text."""
    vec = build_tfidf_vectorizer(max_features=max_features, ngram_range=ngram_range)
    clean_train = preprocess_text_series(train_text)
    x_train = vec.fit_transform(clean_train)

    x_valid = None
    if valid_text is not None:
        clean_valid = preprocess_text_series(valid_text)
        x_valid = vec.transform(clean_valid)

    return vec, x_train, x_valid
