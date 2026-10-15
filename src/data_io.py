"""I/O utilities for loading train/test data in local and Kaggle environments.

Supports two layouts:
1) train/test already contain complaint text column.
2) complaint text lives in a separate table (for example chief_complaints.csv),
   which is merged by patient id at load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from src.config import as_path


def is_kaggle_runtime() -> bool:
    """Return True when running inside a Kaggle notebook/runtime."""
    return Path("/kaggle/input").exists()


def _resolve_base_data_dir(cfg: Dict[str, Any]) -> Path:
    data_cfg = cfg["data"]
    if is_kaggle_runtime():
        return Path(data_cfg.get("kaggle_input_dir", "/kaggle/input/triagegeist"))
    return as_path(data_cfg.get("local_data_dir", "data/raw"))



def _unique_records(df: pd.DataFrame, key: str, table: str) -> pd.DataFrame:
    """Collapse identical records, but reject conflicting values for one identifier."""
    if key not in df or df[key].isna().any():
        raise ValueError(f"{table}: join IDs must be present and non-missing")
    unique = df.drop_duplicates()
    if unique[key].duplicated().any():
        raise ValueError(f"{table}: conflicting duplicate {key} values")
    return unique


def load_train_test(cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test CSV files from local or Kaggle paths.

    Expected config keys:
    - data.train_file
    - data.test_file
    - data.local_data_dir
    - data.kaggle_input_dir
    """
    base_dir = _resolve_base_data_dir(cfg)
    train_path = base_dir / cfg["data"].get("train_file", "train.csv")
    test_path = base_dir / cfg["data"].get("test_file", "test.csv")

    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    id_col = cfg['data']['id_col']
    target_col = cfg['data']['target_col']
    for label, frame in [('train', train_df), ('test', test_df)]:
        if id_col not in frame or frame[id_col].isna().any() or frame[id_col].duplicated().any():
            raise ValueError(f'{label}: IDs must be present, unique and non-missing')
    if target_col not in train_df or train_df[target_col].isna().any():
        raise ValueError('Training labels must be present and non-missing')
    if not set(train_df[target_col].unique()).issubset({1, 2, 3, 4, 5}):
        raise ValueError('Training labels must be acuity levels 1-5')
    if set(train_df[id_col]) & set(test_df[id_col]):
        raise ValueError('Train and test identifiers overlap')
    train_df, test_df = _merge_auxiliary_tables(train_df, test_df, base_dir, cfg)
    return train_df, test_df


def _merge_auxiliary_tables(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_dir: Path,
    cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Merge complaint text and optional patient history when configured."""
    data_cfg = cfg["data"]
    id_col = data_cfg.get("id_col", "patient_id")
    text_col = data_cfg.get("text_col", "chief_complaint_raw")

    complaints_file = data_cfg.get("complaints_file", "chief_complaints.csv")
    complaint_key = data_cfg.get("complaints_join_key", id_col)
    complaint_text_source_col = data_cfg.get("complaints_text_col", text_col)

    needs_text_merge = text_col not in train_df.columns or text_col not in test_df.columns
    complaints_path = base_dir / complaints_file

    if needs_text_merge and not complaints_path.exists():
        raise FileNotFoundError(f"Complaint table required: {complaints_path}")
    if needs_text_merge:
        complaints_df = pd.read_csv(complaints_path)
        complaint_cols = [complaint_key, complaint_text_source_col]
        if not set(complaint_cols).issubset(complaints_df.columns):
            missing = [c for c in complaint_cols if c not in complaints_df.columns]
            raise KeyError(f"Missing complaint columns in {complaints_path}: {missing}")

        merge_df = _unique_records(complaints_df[complaint_cols], complaint_key, "complaints").rename(
            columns={complaint_key: id_col, complaint_text_source_col: text_col}
        )
        if text_col not in train_df:
            train_df = train_df.merge(merge_df, on=id_col, how="left", validate="one_to_one", sort=False)
        if text_col not in test_df:
            test_df = test_df.merge(merge_df, on=id_col, how="left", validate="one_to_one", sort=False)

    include_history = data_cfg.get("include_patient_history", True)
    history_file = data_cfg.get("patient_history_file", "patient_history.csv")
    history_key = data_cfg.get("patient_history_join_key", id_col)
    history_path = base_dir / history_file

    if include_history and not history_path.exists():
        raise FileNotFoundError(f"History table required: {history_path}")
    if include_history:
        history_df = pd.read_csv(history_path)
        if history_key not in history_df.columns:
            raise KeyError(f"Missing history join key '{history_key}' in {history_path}")
        history_df = _unique_records(history_df, history_key, "history").rename(columns={history_key: id_col})
        train_df = train_df.merge(history_df, on=id_col, how="left", validate="one_to_one", sort=False)
        test_df = test_df.merge(history_df, on=id_col, how="left", validate="one_to_one", sort=False)

    for frame in (train_df, test_df):
        frame[text_col] = frame[text_col].fillna("")
    return train_df, test_df


def save_submission(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save a submission DataFrame and return the saved path."""
    out = as_path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
