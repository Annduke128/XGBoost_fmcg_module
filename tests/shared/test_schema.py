import pandas as pd
import pytest
from ml.shared.schema import validate_columns, REQUIRED_COLS


def test_validate_columns_pass():
    data = {c: ["x"] for c in REQUIRED_COLS}
    df = pd.DataFrame(data)
    result = validate_columns(df)
    assert len(result) == 1


def test_validate_columns_missing():
    df = pd.DataFrame({"week": ["2025-01-06"]})
    with pytest.raises(ValueError, match="Missing columns"):
        validate_columns(df)


def test_validate_columns_empty_df():
    data = {c: [] for c in REQUIRED_COLS}
    df = pd.DataFrame(data)
    result = validate_columns(df)
    assert len(result) == 0
