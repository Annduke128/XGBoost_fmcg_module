"""Vietnam public holiday loader.

Loads holidays from CSV or provides built-in defaults.
CSV format: date,name (e.g. 2025-01-01,Tet Duong Lich)
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# Built-in Vietnam public holidays (recurring month-day patterns)
# Users can override by providing a CSV file
_DEFAULT_HOLIDAYS_MD = [
    (1, 1),  # Tet Duong Lich (New Year)
    (4, 30),  # Ngay Giai Phong
    (5, 1),  # Quoc Te Lao Dong
    (9, 2),  # Quoc Khanh
]

# Tet Am Lich dates vary by year — approximate for recent years
_TET_DATES: dict[int, list[str]] = {
    2023: [
        "2023-01-20",
        "2023-01-21",
        "2023-01-22",
        "2023-01-23",
        "2023-01-24",
        "2023-01-25",
        "2023-01-26",
    ],
    2024: [
        "2024-02-08",
        "2024-02-09",
        "2024-02-10",
        "2024-02-11",
        "2024-02-12",
        "2024-02-13",
        "2024-02-14",
    ],
    2025: [
        "2025-01-25",
        "2025-01-26",
        "2025-01-27",
        "2025-01-28",
        "2025-01-29",
        "2025-01-30",
        "2025-01-31",
    ],
    2026: [
        "2026-02-14",
        "2026-02-15",
        "2026-02-16",
        "2026-02-17",
        "2026-02-18",
        "2026-02-19",
        "2026-02-20",
    ],
}


def load_holidays_from_csv(path: str | Path) -> pd.Series:
    """Load holidays from CSV file. Returns Series of dates."""
    df = pd.read_csv(path, parse_dates=["date"])
    return pd.to_datetime(df["date"]).dt.normalize()


def get_default_holidays(years: list[int] | None = None) -> pd.Series:
    """Get built-in Vietnam holidays for given years.

    Returns a Series of datetime dates.
    """
    if years is None:
        years = list(range(2023, 2027))

    dates: list[str] = []

    for year in years:
        # Fixed holidays
        for month, day in _DEFAULT_HOLIDAYS_MD:
            dates.append(f"{year}-{month:02d}-{day:02d}")

        # Tet dates (variable)
        if year in _TET_DATES:
            dates.extend(_TET_DATES[year])

    return pd.to_datetime(pd.Series(dates)).dt.normalize()


def build_holiday_set(
    csv_path: str | Path | None = None,
    years: list[int] | None = None,
) -> set[pd.Timestamp]:
    """Build a set of holiday dates for fast lookup.

    If csv_path is provided, uses that. Otherwise uses built-in defaults.
    """
    if csv_path is not None:
        holidays = load_holidays_from_csv(csv_path)
    else:
        holidays = get_default_holidays(years)

    return set(holidays)
