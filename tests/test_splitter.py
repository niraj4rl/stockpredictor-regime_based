import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from backtest.splitter import WalkForwardSplitter


def make_dummy_df(n=900) -> pd.DataFrame:
    dates = pd.bdate_range("2018-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    df = pd.DataFrame({"Close": close}, index=dates)
    return df


def test_purged_gap_between_train_and_test():
    df = make_dummy_df()
    splitter = WalkForwardSplitter(train_years=2, test_months=3, purge_gap_days=7)

    folds = list(splitter.split(df))
    assert len(folds) > 0

    for _, train_df, test_df, _, _ in folds:
        assert len(train_df) > 0
        assert len(test_df) > 0
        # Ensure there is at least the configured purge gap between windows.
        gap_days = (test_df.index[0] - train_df.index[-1]).days
        assert gap_days >= 7


def test_describe_contains_purge_metadata():
    df = make_dummy_df()
    splitter = WalkForwardSplitter(train_years=2, test_months=3, purge_gap_days=5)
    desc = splitter.describe(df)

    assert "purge_start" in desc.columns
    assert "purge_gap_days" in desc.columns
    assert (desc["purge_gap_days"] == 5).all()
