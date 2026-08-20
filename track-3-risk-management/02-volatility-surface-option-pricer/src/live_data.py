"""Live spot price and risk-free rate lookups.

This is the actual differentiator versus a purely parametric pricer:
the app auto-populates the underlying price from Twelve Data and the
risk-free rate from FRED instead of asking for both to be typed in by
hand. Both calls are cheap (one request each) and cached briefly by
Streamlit in app.py to stay well inside free-tier rate limits.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

TWELVE_DATA_KEY = os.environ.get("TWELVE_DATA_API_KEY")
FRED_API_KEY = os.environ.get("FRED_API_KEY")

QUOTE_URL = "https://api.twelvedata.com/quote"
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"

# 3-month US Treasury yield, the standard short-term risk-free proxy.
# Swap for an ECB short-term rate series if pricing EUR-denominated options.
FRED_RISK_FREE_SERIES = "DGS3MO"


def fetch_spot(symbol):
    """Latest quoted price for `symbol` via Twelve Data. Free tier is
    end-of-day/latest-available, not sub-second tick data.
    """
    if not TWELVE_DATA_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY not set. Copy .env.example to .env and add a free key from twelvedata.com."
        )
    resp = requests.get(QUOTE_URL, params={"symbol": symbol, "apikey": TWELVE_DATA_KEY}, timeout=15)
    payload = resp.json()
    if payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {payload.get('message')}")

    return {
        "symbol": payload["symbol"],
        "price": float(payload["close"]),
        "previous_close": float(payload["previous_close"]),
        "percent_change": float(payload["percent_change"]),
        "datetime": payload["datetime"],
    }


def fetch_risk_free_rate():
    """Latest 3-month US Treasury yield from FRED, as a decimal (5.23% -> 0.0523).

    Optional: returns None if FRED_API_KEY isn't set, so the app can fall
    back to a manual rate input rather than hard-failing.
    """
    if not FRED_API_KEY:
        return None

    params = {
        "series_id": FRED_RISK_FREE_SERIES,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    resp = requests.get(FRED_URL, params=params, timeout=15)
    payload = resp.json()
    for obs in payload.get("observations", []):
        if obs["value"] != ".":
            return float(obs["value"]) / 100
    return None
