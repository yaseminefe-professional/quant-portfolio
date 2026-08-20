# VaR / CVaR Calculator with Backtesting

**Tier:** Foundation (Track 3: Risk Management & Derivatives)

## Problem

Risk desks don't compute Value at Risk one way and trust it. They compute it several ways, because each method makes a different assumption about the return distribution, and they check afterwards whether the number was actually right. This project does the same: three VaR/CVaR methods on the same live portfolio, then a formal test of whether each method's breach rate matches what it predicted.

## Method

1. **Historical VaR/CVaR** — no distributional assumption. Sort realized portfolio returns, take the empirical quantile at the chosen confidence level (95% and 99%). CVaR (expected shortfall) is the mean of the losses beyond that quantile.
2. **Parametric (variance-covariance) VaR/CVaR** — assumes portfolio returns are normally distributed. Estimate the mean and covariance matrix of the constituent assets, derive portfolio variance from position weights, and read the VaR off the normal quantile. Diverges from the historical number when returns are fat-tailed or skewed, which is itself worth reporting.
3. **Monte Carlo VaR/CVaR** — simulate thousands of forward return paths from the estimated covariance matrix, compute simulated portfolio losses, take the empirical quantile of the simulated distribution. More flexible than parametric (can swap in a non-normal simulator later) at the cost of simulation noise.
4. **Kupiec proportion-of-failures backtest** — roll each VaR model forward over the available history, count how many days the realized loss exceeded the predicted VaR, and test via a likelihood-ratio statistic whether that breach rate is statistically consistent with the model's stated confidence level. A model that breaches too often is too aggressive; too rarely, and it's too conservative and wasting capital.

## Data

**Twelve Data** (`https://api.twelvedata.com`), free tier: 800 requests/day, 8 requests/minute. Get a free key at twelvedata.com and put it in `.env` (copy `.env.example`). Daily OHLC bars, `outputsize` controls how much history is pulled per symbol.

**Limitation:** the free tier's daily bars are end-of-day, not tick-level intraday. That's fine for a daily VaR model (which is standard practice) but worth stating explicitly rather than implying this is a real-time tick feed.

## Stack

Python, pandas, numpy, scipy.stats, requests, python-dotenv, matplotlib.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your Twelve Data key
python src/main.py --symbols AAPL MSFT GOOGL --weights 0.4 0.35 0.25
```

## Status

Scaffolded, not yet run against live data (needs a Twelve Data API key).
