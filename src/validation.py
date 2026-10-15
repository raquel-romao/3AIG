"""Validation shared by research and submission export."""
import numpy as np
import pandas as pd


def validate_submission(frame, expected_ids, id_col='patient_id', target_col='triage_acuity'):
    """Reject reordered, duplicated, missing or invalid competition predictions."""
    if list(frame.columns) != [id_col, target_col]:
        raise ValueError('Submission must contain exactly ID and target columns')
    actual = frame[id_col].reset_index(drop=True)
    expected = pd.Series(expected_ids).reset_index(drop=True)
    if frame.isna().any().any() or actual.duplicated().any() or not actual.equals(expected):
        raise ValueError('Submission IDs must be unique, complete and in test order')
    if not np.isin(frame[target_col].to_numpy(), [1, 2, 3, 4, 5]).all():
        raise ValueError('Submission labels must be acuity levels 1-5')
