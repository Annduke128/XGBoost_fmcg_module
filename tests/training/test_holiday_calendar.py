"""Tests for holiday_calendar module."""

from __future__ import annotations

import pandas as pd
import pytest
from ml.training.data.holiday_calendar import (
    build_holiday_set,
    get_default_holidays,
)


def test_default_holidays_returns_dates():
    holidays = get_default_holidays(years=[2025])
    assert len(holidays) > 0
    assert all(isinstance(d, pd.Timestamp) for d in holidays)


def test_default_holidays_includes_tet():
    holidays = get_default_holidays(years=[2025])
    tet_date = pd.Timestamp("2025-01-27")
    assert tet_date in holidays.values


def test_default_holidays_includes_fixed():
    holidays = get_default_holidays(years=[2025])
    new_year = pd.Timestamp("2025-01-01")
    assert new_year in holidays.values


def test_build_holiday_set_returns_set():
    result = build_holiday_set(years=[2025])
    assert isinstance(result, set)
    assert len(result) > 0


def test_build_holiday_set_from_csv(tmp_path):
    csv_file = tmp_path / "holidays.csv"
    csv_file.write_text("date,name\n2025-01-01,New Year\n2025-05-01,Labor Day\n")
    result = build_holiday_set(csv_path=str(csv_file))
    assert pd.Timestamp("2025-01-01") in result
    assert pd.Timestamp("2025-05-01") in result
    assert len(result) == 2
