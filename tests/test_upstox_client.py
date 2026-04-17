import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from data.upstox_client import UpstoxDataClient, TICKER_TO_INSTRUMENT_KEY


def test_known_ticker_resolution():
    """Known tickers should resolve from the built-in mapping without API calls."""
    client = UpstoxDataClient("dummy_token")
    key = client.resolve_instrument_key("RELIANCE.NS")
    assert key == "NSE_EQ|INE002A01018"


def test_all_popular_tickers_mapped():
    """All popular tickers should have entries in the mapping."""
    popular = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "WIPRO.NS", "SBIN.NS", "BAJFINANCE.NS",
        "HINDUNILVR.NS", "ITC.NS", "AXISBANK.NS", "MARUTI.NS",
    ]
    for ticker in popular:
        assert ticker in TICKER_TO_INSTRUMENT_KEY, f"{ticker} missing from mapping"


def test_empty_token_raises():
    """Creating a client with empty token should raise ValueError."""
    with pytest.raises(ValueError, match="access token is required"):
        UpstoxDataClient("")


def test_whitespace_token_raises():
    """Creating a client with whitespace-only token should raise ValueError."""
    with pytest.raises(ValueError, match="access token is required"):
        UpstoxDataClient("   ")


def test_case_insensitive_ticker():
    """Ticker resolution should be case-insensitive."""
    client = UpstoxDataClient("dummy_token")
    key = client.resolve_instrument_key("reliance.ns")
    assert key == "NSE_EQ|INE002A01018"


@patch("data.upstox_client.requests.get")
def test_fetch_historical_candles_format(mock_get):
    """Fetched candles should return a properly formatted DataFrame."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "candles": [
                ["2024-01-03T00:00:00+05:30", 2500.0, 2520.0, 2490.0, 2510.0, 1000000, 0],
                ["2024-01-02T00:00:00+05:30", 2480.0, 2510.0, 2470.0, 2500.0, 900000, 0],
                ["2024-01-01T00:00:00+05:30", 2450.0, 2490.0, 2440.0, 2480.0, 800000, 0],
            ]
        }
    }
    mock_get.return_value = mock_response

    client = UpstoxDataClient("test_token")
    df = client.fetch_historical_candles("NSE_EQ|INE002A01018", "2024-01-01", "2024-01-03")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3
    assert df.index.is_monotonic_increasing


@patch("data.upstox_client.requests.get")
def test_fetch_ohlcv_has_returns(mock_get):
    """fetch_ohlcv should add pct_return and log_return columns."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": {
            "candles": [
                [f"2024-01-{d:02d}T00:00:00+05:30", 100 + d, 105 + d, 95 + d, 100 + d, 1000000, 0]
                for d in range(1, 11)
            ]
        }
    }
    mock_get.return_value = mock_response

    client = UpstoxDataClient("test_token")
    df = client.fetch_ohlcv("RELIANCE.NS", period="1y")

    assert "pct_return" in df.columns
    assert "log_return" in df.columns
    assert "Open" in df.columns
    assert "Close" in df.columns


@patch("data.upstox_client.requests.get")
def test_api_error_raises(mock_get):
    """API returning non-200 should raise ConnectionError."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_get.return_value = mock_response

    client = UpstoxDataClient("bad_token")
    with pytest.raises(ConnectionError, match="401"):
        client.fetch_historical_candles("NSE_EQ|INE002A01018", "2024-01-01", "2024-01-03")
