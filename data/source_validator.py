"""
Validate and cache which data sources reliably provide data for each NSE ticker.
Prevents quote-only fallbacks by pre-screening tickers.
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

import requests

VALIDATOR_CACHE = Path(__file__).parent / "ticker_sources.json"

# Known-good tickers (tested & reliable)
RELIABLE_TICKERS = {
    "RELIANCE.NS": ["yfinance", "nse_quote"],
    "TCS.NS": ["yfinance", "nse_quote"],
    "INFY.NS": ["yfinance", "nse_quote"],
    "HDFCBANK.NS": ["yfinance", "nse_quote"],
    "ICICIBANK.NS": ["yfinance", "nse_quote"],
    "WIPRO.NS": ["yfinance", "nse_quote"],
    "SBIN.NS": ["yfinance", "nse_quote"],
    "BAJFINANCE.NS": ["yfinance", "nse_quote"],
    "MARUTI.NS": ["yfinance", "nse_quote"],
    "ITC.NS": ["yfinance", "nse_quote"],
}


def test_yfinance(ticker: str) -> bool:
    """Test if yfinance can fetch data for ticker."""
    try:
        data = yf.download(ticker, period="1mo", progress=False, threads=False)
        return len(data) > 10
    except Exception:
        return False


def test_nse_quote(ticker: str) -> bool:
    """Test if NSE quote API works for ticker."""
    symbol = ticker.replace(".NS", "").upper()
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        resp = session.get(
            "https://www.nseindia.com/api/quote-equity",
            params={"symbol": symbol},
            headers=headers,
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            info = data.get("priceInfo", {})
            return bool(info.get("close") or info.get("lastPrice"))
    except Exception:
        pass
    return False


def validate_ticker(ticker: str) -> dict:
    """
    Test all data sources for a ticker. Returns dict with results.
    Format: {"ticker": str, "valid": bool, "sources": list, "tested_at": ISO}
    """
    sources = []
    
    if test_yfinance(ticker):
        sources.append("yfinance")
    
    if test_nse_quote(ticker):
        sources.append("nse_quote")
    
    return {
        "ticker": ticker,
        "valid": len(sources) > 0,
        "sources": sources,
        "tested_at": datetime.now().isoformat(),
    }


def load_validator_cache() -> dict:
    """Load cached validation results. Returns dict keyed by ticker."""
    if VALIDATOR_CACHE.exists():
        try:
            with open(VALIDATOR_CACHE) as f:
                data = json.load(f)
                # Only use if < 7 days old
                if data.get("timestamp"):
                    ts = datetime.fromisoformat(data["timestamp"])
                    if datetime.now() - ts < timedelta(days=7):
                        return data.get("results", {})
        except Exception:
            pass
    return {}


def save_validator_cache(results: dict) -> None:
    """Save validation results to cache."""
    cache_data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    try:
        with open(VALIDATOR_CACHE, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"[validator] Failed to save cache: {e}")


def get_valid_tickers(tickers: list) -> dict:
    """
    Filter tickers, returning only those with working data sources.
    Since most NSE-listed stocks work, we trust the NSE list as authoritative.
    Only exclude tickers if we've explicitly tested and confirmed they fail.
    
    Returns: {"valid": [tickers], "invalid": [tickers], "results": {ticker: {valid, sources}}}
    """
    cached = load_validator_cache()
    results = {}
    valid = []
    invalid = []
    
    # Mark all tickers as potentially valid unless we have cached failure
    for ticker in tickers:
        if ticker in cached:
            result = cached[ticker]
            results[ticker] = result
            if result.get("valid"):
                valid.append(ticker)
            else:
                invalid.append(ticker)
        else:
            # Assume valid for NSE tickers unless proven otherwise
            # This avoids blocking the app on first load
            result = {
                "ticker": ticker,
                "valid": True,
                "sources": ["yfinance", "nse_quote"],  # Try both
                "tested_at": None,
                "cached": False,
                "reason": "assumed_valid_nse_ticker",
            }
            results[ticker] = result
            valid.append(ticker)
    
    return {"valid": valid, "invalid": invalid, "results": results}


def validate_and_cache_all(tickers: list) -> None:
    """
    (Expensive operation) Test all tickers and save results.
    Run this as a background job, not during live prediction.
    """
    cached = load_validator_cache()
    results = cached.copy()
    
    for i, ticker in enumerate(tickers):
        if ticker not in results:
            print(f"[validator] Testing {ticker} ({i+1}/{len(tickers)})...")
            results[ticker] = validate_ticker(ticker)
    
    save_validator_cache(results)
    print(f"[validator] Cached validation for {len(results)} tickers")
