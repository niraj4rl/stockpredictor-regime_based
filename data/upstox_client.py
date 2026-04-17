import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


def get_auth_url(api_key: str, redirect_uri: str = "https://127.0.0.1") -> str:
    """Build the Upstox authorization URL the user must visit to login."""
    return (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}"
    )


def exchange_code_for_token(
    api_key: str,
    api_secret: str,
    auth_code: str,
    redirect_uri: str = "https://127.0.0.1",
) -> str:
    """
    Exchange an authorization code for an access token.
    Returns the access_token string.
    """
    resp = requests.post(
        "https://api.upstox.com/v2/login/authorization/token",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": auth_code.strip(),
            "client_id": api_key.strip(),
            "client_secret": api_secret.strip(),
            "redirect_uri": redirect_uri.strip(),
            "grant_type": "authorization_code",
        },
        timeout=15,
    )

    if resp.status_code != 200:
        raise ConnectionError(
            f"Token exchange failed (HTTP {resp.status_code}): {resp.text}"
        )

    data = resp.json()
    token = data.get("access_token", "")
    if not token:
        raise ValueError(f"No access_token in response: {data}")

    return token



# Mapping of popular NSE tickers (yfinance-style) to Upstox instrument keys
# Format: NSE_EQ|<ISIN>
TICKER_TO_INSTRUMENT_KEY = {
    "RELIANCE.NS":   "NSE_EQ|INE002A01018",
    "TCS.NS":        "NSE_EQ|INE467B01029",
    "INFY.NS":       "NSE_EQ|INE009A01021",
    "HDFCBANK.NS":   "NSE_EQ|INE040A01034",
    "ICICIBANK.NS":  "NSE_EQ|INE090A01021",
    "WIPRO.NS":      "NSE_EQ|INE075A01022",
    "SBIN.NS":       "NSE_EQ|INE062A01020",
    "BAJFINANCE.NS": "NSE_EQ|INE296A01024",
    "HINDUNILVR.NS": "NSE_EQ|INE030A01027",
    "ITC.NS":        "NSE_EQ|INE154A01025",
    "AXISBANK.NS":   "NSE_EQ|INE238A01034",
    "MARUTI.NS":     "NSE_EQ|INE585B01010",
    "KOTAKBANK.NS":  "NSE_EQ|INE237A01028",
    "LT.NS":         "NSE_EQ|INE018A01030",
    "BHARTIARTL.NS": "NSE_EQ|INE397D01024",
    "ASIANPAINT.NS": "NSE_EQ|INE021A01026",
    "HCLTECH.NS":    "NSE_EQ|INE860A01027",
    "SUNPHARMA.NS":  "NSE_EQ|INE044A01036",
    "TATAMOTORS.NS": "NSE_EQ|INE155A01022",
    "TATASTEEL.NS":  "NSE_EQ|INE081A01020",
    "ULTRACEMCO.NS": "NSE_EQ|INE481G01011",
    "TECHM.NS":      "NSE_EQ|INE669C01036",
    "POWERGRID.NS":  "NSE_EQ|INE752E01010",
    "NTPC.NS":       "NSE_EQ|INE733E01010",
    "TITAN.NS":      "NSE_EQ|INE280A01028",
    "ADANIENT.NS":   "NSE_EQ|INE423A01024",
    "ADANIPORTS.NS": "NSE_EQ|INE742F01042",
    "ONGC.NS":       "NSE_EQ|INE213A01029",
    "JSWSTEEL.NS":   "NSE_EQ|INE019A01038",
    "DRREDDY.NS":    "NSE_EQ|INE089A01023",
}

BASE_URL = "https://api.upstox.com/v2"


class UpstoxDataClient:
    """Client for fetching stock data from the Upstox API v2."""

    def __init__(self, access_token: str):
        if not access_token or not access_token.strip():
            raise ValueError("Upstox access token is required")
        self.access_token = access_token.strip()
        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}",
        }

    def get_live_price(self, ticker: str) -> float:
        """Fetch the live/last traded price from Upstox market quote API."""
        instrument_key = self.resolve_instrument_key(ticker)
        resp = requests.get(
            f"{BASE_URL}/market-quote/ltp",
            headers=self._headers,
            params={"instrument_key": instrument_key},
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            # Response is keyed by instrument_key
            for key, quote in data.items():
                ltp = quote.get("last_price", 0)
                if ltp and ltp > 0:
                    return float(ltp)
        raise ValueError(f"Could not fetch live price for {ticker}")

    def resolve_instrument_key(self, ticker: str) -> str:
        """
        Convert a yfinance-style NSE ticker (e.g. RELIANCE.NS)
        to an Upstox instrument key (e.g. NSE_EQ|INE002A01018).
        """
        ticker = ticker.upper().strip()

        # Check built-in mapping first
        if ticker in TICKER_TO_INSTRUMENT_KEY:
            return TICKER_TO_INSTRUMENT_KEY[ticker]

        # Fallback: search via Upstox API
        symbol = ticker.replace(".NS", "").replace(".BO", "")
        try:
            resp = requests.get(
                f"{BASE_URL}/search/instruments",
                headers=self._headers,
                params={"query": symbol, "segments": "NSE_EQ"},
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                for item in data:
                    if item.get("trading_symbol", "").upper() == symbol:
                        return item["instrument_key"]
                if data:
                    return data[0]["instrument_key"]
        except Exception as e:
            print(f"[upstox] Instrument search failed for {ticker}: {e}")

        raise ValueError(
            f"Could not resolve Upstox instrument key for ticker: {ticker}. "
            f"Please add it to the TICKER_TO_INSTRUMENT_KEY mapping in data/upstox_client.py"
        )

    def fetch_historical_candles(
        self,
        instrument_key: str,
        from_date: str,
        to_date: str,
        interval: str = "day",
    ) -> pd.DataFrame:
        """
        Fetch historical candle data from Upstox.

        Args:
            instrument_key: e.g. 'NSE_EQ|INE002A01018'
            from_date: 'YYYY-MM-DD'
            to_date: 'YYYY-MM-DD'
            interval: '1minute', '30minute', 'day', 'week', 'month'

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
        """
        url = (
            f"{BASE_URL}/historical-candle/{instrument_key}/{interval}"
            f"/{to_date}/{from_date}"
        )

        resp = requests.get(url, headers=self._headers, timeout=30)

        if resp.status_code != 200:
            raise ConnectionError(
                f"Upstox API returned status {resp.status_code}: {resp.text}"
            )

        data = resp.json().get("data", {}).get("candles", [])
        if not data:
            raise ValueError(f"No candle data returned for {instrument_key}")

        # Upstox candle format: [timestamp, open, high, low, close, volume, oi]
        rows = []
        for candle in data:
            rows.append({
                "Date": pd.to_datetime(candle[0]),
                "Open": float(candle[1]),
                "High": float(candle[2]),
                "Low": float(candle[3]),
                "Close": float(candle[4]),
                "Volume": int(candle[5]),
            })

        df = pd.DataFrame(rows)
        df = df.set_index("Date")
        df = df.sort_index()
        return df

    def fetch_ohlcv(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        High-level method: resolve ticker, calculate date range, fetch candles.
        Returns a DataFrame matching the format expected by Stocker's ingestion pipeline.
        """
        instrument_key = self.resolve_instrument_key(ticker)

        # Calculate date range from period string
        to_date = datetime.now()
        period_map = {
            "1y": timedelta(days=365),
            "2y": timedelta(days=730),
            "3y": timedelta(days=1095),
            "5y": timedelta(days=1825),
            "10y": timedelta(days=3650),
        }
        delta = period_map.get(period, timedelta(days=1825))
        from_date = to_date - delta

        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        print(f"[upstox] Fetching {ticker} ({instrument_key}) from {from_str} to {to_str}...")

        df = self.fetch_historical_candles(instrument_key, from_str, to_str, interval="day")

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        df = df[~df.index.duplicated(keep="last")]

        # Add returns (same as yfinance ingestion)
        df["pct_return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        print(f"[upstox] Loaded {len(df)} rows for {ticker}")
        return df
