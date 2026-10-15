"""Preprocessing routines for structured and text triage inputs."""

from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd


WHITESPACE_RE = re.compile(r"\s+")
ABBREVIATIONS = {
    'uti': 'urinary tract infection', 'urti': 'upper respiratory tract infection',
    'loc': 'loss of consciousness', 'gi': 'gastrointestinal', 'bp': 'blood pressure',
    'dka': 'diabetic ketoacidosis', 'af': 'atrial fibrillation',
    'mva': 'motor vehicle accident', 'sti': 'sexually transmitted infection',
    'nstemi': 'non st elevation myocardial infarction',
    'ibs': 'irritable bowel syndrome', 'stemi': 'st elevation myocardial infarction',
    'gcs': 'glasgow coma scale', 'ecg': 'electrocardiogram',
}
ABBREVIATION_RE = re.compile(r'\b(' + '|'.join(ABBREVIATIONS) + r')\b')
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")


def clean_chief_complaint(text: str) -> str:
    """Normalize chief complaint text for classic NLP baselines.

    This keeps the pipeline simple and laptop-friendly:
    - lowercase
    - remove punctuation
    - collapse whitespace
    """
    if text is None or pd.isna(text):
        return ""
    value = str(text).strip().lower()
    value = ABBREVIATION_RE.sub(lambda match: ABBREVIATIONS[match.group(0)], value)
    value = NON_ALNUM_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def preprocess_text_series(text_series: Iterable[str]) -> List[str]:
    """Apply complaint text cleaning to an iterable of strings."""
    return [clean_chief_complaint(t) for t in text_series]


def drop_leakage_columns(df: pd.DataFrame, leakage_cols: list[str]) -> pd.DataFrame:
    """Drop known leakage columns if present.

    Keep this strict for triage-time framing.
    """
    existing = [c for c in leakage_cols if c in df.columns]
    return df.drop(columns=existing)
