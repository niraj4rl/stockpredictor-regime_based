import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from features.engineering import build_features, get_feature_cols


def make_dummy_ohlcv(n=400) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.bdate_range("2019-01-01", periods=n)
    close = 100 * np.cumprod(1 + np.random.randn(n) * 0.01)
    df = pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.random.randint(1_000_000, 5_000_000, n),
    }, index=dates)
    df["pct_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def test_no_lookahead():
    df = make_dummy_ohlcv()
    feat_df = build_features(df)
    feat_cols = get_feature_cols(feat_df)
    for col in feat_cols:
        for i in range(1, len(feat_df)):
            val_t = feat_df[col].iloc[i]
            future_close = df["Close"].iloc[i + 1] if i + 1 < len(df) else None
            if future_close is not None and not pd.isna(val_t):
                assert not np.isclose(val_t, future_close), (
                    f"Feature {col} at index {i} looks like future data"
                )


def test_feature_count():
    df = make_dummy_ohlcv()
    feat_df = build_features(df)
    feat_cols = get_feature_cols(feat_df)
    assert len(feat_cols) >= 15, f"Expected ≥15 features, got {len(feat_cols)}"


def test_no_nan_after_build():
    df = make_dummy_ohlcv()
    feat_df = build_features(df)
    feat_cols = get_feature_cols(feat_df)
    nan_counts = feat_df[feat_cols].isna().sum()
    assert nan_counts.sum() == 0, f"NaNs in features after build: {nan_counts[nan_counts>0]}"


def test_target_not_in_features():
    df = make_dummy_ohlcv()
    feat_df = build_features(df)
    feat_cols = get_feature_cols(feat_df)
    assert "target_pct_return" not in feat_cols
    assert "target_price" not in feat_cols
