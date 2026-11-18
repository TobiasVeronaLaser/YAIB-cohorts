"""Regression tests for stay-specific outcome windowing."""

import numpy as np
import pandas as pd

from src.cohort_utils import make_outcome_windower


def test_outcome_window_does_not_cross_stays():
    df = pd.DataFrame({
        "stay_id": [1] * 4 + [2] * 4,
        "time": list(range(4)) * 2,
        "label": [np.nan] * 4 + [1, np.nan, np.nan, np.nan],
    })

    result = make_outcome_windower(2)(df)

    assert result["label"].tolist() == [
        0, 0, 0, 0,
        1, 1, 1, 0,
    ]


def test_outcome_window_expands_in_both_directions():
    df = pd.DataFrame({
        "stay_id": [1] * 7,
        "time": list(range(7)),
        "label": [
            np.nan, np.nan, np.nan, 1,
            np.nan, np.nan, np.nan
        ],
    })

    result = make_outcome_windower(2)(df)

    assert result["label"].tolist() == [
        0, 1, 1, 1, 1, 1, 0,
    ]


def test_outcome_window_without_positive_labels():
    df = pd.DataFrame({
        "stay_id": [1, 1, 2, 2],
        "time": [0, 1, 0, 1],
        "label": [np.nan] * 4,
    })

    result = make_outcome_windower(2)(df)

    assert result["label"].tolist() == [0, 0, 0, 0]
