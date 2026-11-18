"""Regression tests for missing required outcome concepts."""

import pandas as pd
import polars as pl
import pytest

from src import weavehr


@pytest.mark.parametrize("outcome", ["aki", "sep3"])
def test_missing_outcome_raises(monkeypatch, outcome):
    monkeypatch.setattr(weavehr, "_files", lambda *args: [])

    with pytest.raises(FileNotFoundError, match="Missing required"):
        weavehr.OUTCOMES[outcome]("hirid", pd.Timedelta(hours=1))


def test_missing_optional_concept_is_empty(monkeypatch):
    monkeypatch.setattr(weavehr, "_files", lambda *args: [])

    result = weavehr._concept("hirid", "creatinine").collect()

    assert result.is_empty()
    assert "numeric_value" in result.columns


def test_required_concept_can_be_empty(monkeypatch, tmp_path):
    path = tmp_path / "aki.parquet"

    pl.DataFrame(
        schema={
            "subject_id": pl.Int64,
            "time": pl.Datetime("us"),
            "numeric_value": pl.Float32,
            "text_value": pl.String,
        }
    ).write_parquet(path)

    monkeypatch.setattr(weavehr, "_files", lambda *args: [path])

    result = weavehr._concept(
        "hirid", "aki", required=True
    ).collect()

    assert result.is_empty()
