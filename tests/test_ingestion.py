import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest

from data.ingestion import _assert_data_quality


def make_df(n=300, start="2025-01-01") -> pd.DataFrame:
    idx = pd.bdate_range(start, periods=n)
    close = np.linspace(100, 120, n)
    df = pd.DataFrame({
        "Open": close,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Volume": np.full(n, 1_000_000),
    }, index=idx)
    return df


def test_data_quality_accepts_sufficient_recent_data():
    df = make_df(n=300, start="2025-01-01")
    # Move index to be recent enough regardless of test execution date.
    df.index = pd.bdate_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=350), periods=len(df))
    _assert_data_quality(df, "RELIANCE.NS")


def test_data_quality_rejects_insufficient_rows():
    df = make_df(n=50, start="2025-01-01")
    df.index = pd.bdate_range(pd.Timestamp.now().normalize() - pd.Timedelta(days=100), periods=len(df))
    with pytest.raises(ValueError, match="Insufficient history"):
        _assert_data_quality(df, "RELIANCE.NS")


def test_data_quality_rejects_stale_data():
    df = make_df(n=300, start="2020-01-01")
    with pytest.raises(ValueError, match="Stale data"):
        _assert_data_quality(df, "RELIANCE.NS")
