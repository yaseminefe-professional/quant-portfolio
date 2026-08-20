"""Live price data pulled from Twelve Data's free tier."""

import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.twelvedata.com/time_series"
API_KEY = os.environ.get("TWELVE_DATA_API_KEY")


def fetch_prices(symbols, outputsize=500, interval="1day", max_retries=3):
    """Fetch daily close prices for one or more symbols.

    Returns a DataFrame indexed by date, one column per symbol, sorted
    ascending. Twelve Data's free tier returns newest-first and, for a
    multi-symbol request, nests each symbol's series under its own key
    instead of a flat 'values' list, so both shapes are handled here.
    """
    if not API_KEY:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY not set. Copy .env.example to .env and add a free key from twelvedata.com."
        )

    params = {
        "symbol": ",".join(symbols),
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
    }

    payload = None
    for attempt in range(max_retries):
        resp = requests.get(BASE_URL, params=params, timeout=15)
        payload = resp.json()

        is_rate_limited = (
            isinstance(payload, dict)
            and payload.get("status") == "error"
            and "run out of api credits" in payload.get("message", "").lower()
        )
        if is_rate_limited and attempt < max_retries - 1:
            time.sleep(8)
            continue
        break

    if isinstance(payload, dict) and payload.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {payload.get('message')}")

    series = {}
    if len(symbols) == 1:
        series[symbols[0]] = payload["values"]
    else:
        for sym in symbols:
            sym_payload = payload.get(sym, {})
            if sym_payload.get("status") == "error":
                raise RuntimeError(f"Twelve Data error for {sym}: {sym_payload.get('message')}")
            series[sym] = sym_payload["values"]

    frames = []
    for sym, values in series.items():
        df = pd.DataFrame(values)[["datetime", "close"]]
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"close": sym}).set_index("datetime")
        df[sym] = df[sym].astype(float)
        frames.append(df)

    prices = pd.concat(frames, axis=1, join="inner").sort_index()
    return prices


def to_returns(prices):
    """Simple daily percentage returns."""
    return prices.pct_change().dropna()
