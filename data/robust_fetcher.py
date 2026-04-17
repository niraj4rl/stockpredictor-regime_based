"""
Robust multi-source data fetcher for NSE stocks.
Guarantees real data by using intelligent fallback through:
1. Local cache (bootstrap)
2. yfinance with retry & rate-limit handling
3. NSE direct API
4. Yahoo Finance direct chart API
5. Pre-computed bootstrap dataset
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

from utils.config import DATA_DIR, MIN_DATA_ROWS, MAX_STALE_DAYS


class RobustNSEFetcher:
    """
    Multi-source fetcher that guarantees real OHLCV data for NSE stocks.
    Never returns synthetic data; always falls back through real sources.
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds
    
    def __init__(self):
        self.cache_dir = DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def get_ohlcv(
        self,
        ticker: str,
        period: str = "5y",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Get OHLCV data using intelligent fallback through real sources.
        """
        cache_path = self.cache_dir / f"{ticker.replace('.', '_')}.parquet"
        
        # Try cache first (if not forced refresh)
        if cache_path.exists() and not force_refresh:
            try:
                df = pd.read_parquet(cache_path)
                if self._validate_quality(df, ticker):
                    print(f"[robust] Cache hit for {ticker} ({len(df)} rows)")
                    return df
            except Exception as e:
                print(f"[robust] Cache invalid for {ticker}: {e}")
        
        # Try real sources in order
        sources = [
            ("yfinance", self._fetch_yfinance),
            ("yahoo_chart", self._fetch_yahoo_chart),
            ("nse_quote_bootstrap", self._fetch_nse_bootstrap),
        ]
        
        for source_name, source_func in sources:
            try:
                print(f"[robust] Trying {source_name} for {ticker}...")
                df = source_func(ticker, period)
                if self._validate_quality(df, ticker):
                    df.to_parquet(cache_path)
                    print(f"[robust] Successfully fetched {ticker} via {source_name} ({len(df)} rows)")
                    return df
            except Exception as e:
                print(f"[robust] {source_name} failed for {ticker}: {e}")
                time.sleep(self.RETRY_DELAY)
        
        # ALL sources failed
        raise ValueError(
            f"Could not fetch real data for {ticker} from any source. "
            f"Ensure network connectivity and try again."
        )
    
    def _fetch_yfinance(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch from yfinance with retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                df = yf.download(
                    ticker,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
                if len(df) > 0:
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index = pd.to_datetime(df.index)
                    df = df.sort_index()
                    return self._add_returns(df)
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)
                else:
                    raise
        raise ValueError("yfinance retry exhausted")
    
    def _fetch_yahoo_chart(self, ticker: str, period: str) -> pd.DataFrame:
        """Fetch from Yahoo chart API directly."""
        period_map = {"1y": "1y", "2y": "2y", "3y": "3y", "5y": "5y", "10y": "10y"}
        rng = period_map.get(period, "5y")
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        resp = requests.get(
            url,
            params={"range": rng, "interval": "1d"},
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        resp.raise_for_status()
        
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            raise ValueError("No data in Yahoo response")
        
        payload = result[0]
        ts = payload.get("timestamp", [])
        quote = payload.get("indicators", {}).get("quote", [{}])[0]
        
        if not ts or not quote:
            raise ValueError("Empty timestamp or quote in Yahoo response")
        
        df = pd.DataFrame({
            "Open": quote.get("open", []),
            "High": quote.get("high", []),
            "Low": quote.get("low", []),
            "Close": quote.get("close", []),
            "Volume": quote.get("volume", []),
        }, index=pd.to_datetime(ts, unit="s", utc=True).tz_convert(None))
        
        df = df.dropna(subset=["Close"])
        return self._add_returns(df)
    
    def _fetch_nse_bootstrap(self, ticker: str, period: str) -> pd.DataFrame:
        """
        Fetch bootstrap data from NSE via static sources or pre-cached dataset.
        This is a real fallback using NSE-published data.
        """
        symbol = ticker.replace(".NS", "").upper()
        
        # Try NSE historical data via indirect route if available
        bootstrap_file = Path(__file__).parent / "nse_bootstrap" / f"{symbol}.parquet"
        if bootstrap_file.exists():
            df = pd.read_parquet(bootstrap_file)
            return self._add_returns(df)
        
        # If no bootstrap, try NSE quote + synthetic history
        # (Real: use NSE historical archive if available)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        }
        
        try:
            session = requests.Session()
            session.get("https://www.nseindia.com", headers=headers, timeout=8)
            resp = session.get(
                "https://www.nseindia.com/api/quote-equity",
                params={"symbol": symbol},
                headers=headers,
                timeout=8,
            )
            resp.raise_for_status()
            data = resp.json()
            info = data.get("priceInfo", {})
            
            # If we have a current quote, at least return it
            px = float(info.get("close") or info.get("lastPrice") or 0)
            if px > 0:
                # Return 1 row with current price (not ideal, but real)
                df = pd.DataFrame({
                    "Open": [px],
                    "High": [px],
                    "Low": [px],
                    "Close": [px],
                    "Volume": [0],
                }, index=[pd.Timestamp.now()])
                return self._add_returns(df)
        except Exception as e:
            print(f"[robust] NSE bootstrap failed: {e}")
        
        raise ValueError(f"No real data available via NSE bootstrap for {ticker}")
    
    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add percentage and log returns to OHLCV."""
        df = df.copy()
        df["pct_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
        return df
    
    def _validate_quality(self, df: pd.DataFrame, ticker: str) -> bool:
        """Validate data quality."""
        if len(df) < MIN_DATA_ROWS:
            print(f"[robust] Insufficient rows: {len(df)} < {MIN_DATA_ROWS}")
            return False
        
        if df["Close"].isna().all():
            print(f"[robust] All Close prices are NaN")
            return False
        
        last_dt = pd.to_datetime(df.index.max())
        if last_dt.tzinfo:
            last_dt = last_dt.tz_convert("UTC").tz_localize(None)
        now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
        age_days = (now_utc - last_dt).days
        
        if age_days > MAX_STALE_DAYS:
            print(f"[robust] Data is {age_days} days old (max: {MAX_STALE_DAYS})")
            return False
        
        return True


# Global instance
_fetcher = None

def get_robust_fetcher() -> RobustNSEFetcher:
    """Singleton fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = RobustNSEFetcher()
    return _fetcher


def fetch_ohlcv_robust(
    ticker: str,
    period: str = "5y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Convenience function to fetch OHLCV using robust fetcher."""
    fetcher = get_robust_fetcher()
    return fetcher.get_ohlcv(ticker, period, force_refresh)
