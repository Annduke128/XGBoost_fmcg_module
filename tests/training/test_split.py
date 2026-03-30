import pandas as pd
from ml.training.data.split import time_split


def test_time_split_order():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=20, freq="W"),
            "units": list(range(20)),
        }
    )
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    assert train["week"].max() < val["week"].min()
    assert val["week"].max() < test["week"].min()


def test_time_split_no_overlap():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=20, freq="W"),
            "units": list(range(20)),
        }
    )
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    all_weeks = pd.concat([train["week"], val["week"], test["week"]])
    assert all_weeks.duplicated().sum() == 0


def test_time_split_covers_all():
    df = pd.DataFrame(
        {
            "week": pd.date_range("2025-01-06", periods=20, freq="W"),
            "units": list(range(20)),
        }
    )
    train, val, test = time_split(df, val_weeks=4, test_weeks=4)
    assert len(train) + len(val) + len(test) == len(df)
