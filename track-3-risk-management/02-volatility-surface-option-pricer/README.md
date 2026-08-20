# Options Pricing & Greeks

**Tier:** Applied (Track 3: Risk Management & Derivatives)

## Problem

Pricing an option and knowing how that price moves are two different skills. This app prices European options two ways (closed-form Black-Scholes and Monte Carlo simulation), computes the full first- and second-order Greeks, and visualizes how price and risk sensitivities move across volatility, time, and strike, all against a live underlying price and a live risk-free rate instead of hand-typed assumptions.

## Method

1. **Black-Scholes pricing** — the standard dividend-yield-adjusted closed-form solution for European calls and puts.
2. **Greeks via finite differentiation of the same pricing function** — Delta, Gamma, Vega, Theta, and Rho, plus the six second-order Greeks (Charm, Speed, Color, Zomma, Veta, Volga), are all computed as numerical derivatives of `bs_price`, not as ten separately hand-derived closed-form formulas. That guarantees a Greek can't silently drift from the price it's supposed to describe. Verified against Hull's textbook benchmark (S=K=100, T=1, r=5%, σ=20%: call 10.4506, delta 0.6368, gamma 0.01876, vega 37.52, theta -6.414, rho 53.23, all match to 4 decimal places).
3. **Monte Carlo pricing** — simulates terminal prices under risk-neutral geometric Brownian motion, discounts the average payoff, and reports a 95% confidence interval from the simulation's standard error, so the comparison against Black-Scholes is a statistical statement, not just two numbers.
4. **Sensitivity surfaces** — 3D grids of each Greek against spot price and whichever parameter actually drives it (volatility for Delta/Gamma/Vega, time to expiration for Theta, the risk-free rate for Rho).

## Data

**Twelve Data** for the live underlying spot price (`/quote` endpoint), same free tier as project 1. **FRED** for the live 3-month US Treasury yield as the risk-free rate proxy; optional, the app falls back to a manual rate input if `FRED_API_KEY` isn't set.

**Limitation:** this prices against a live spot but not a live option chain, so there's no implied-volatility back-out or comparison to a real market-quoted option price yet. That's a natural next step (see Future improvements) and is intentionally left for a project that focuses on implied volatility specifically, so this one stays focused on pricing and Greeks.

## Stack

Python, Streamlit, NumPy, pandas, SciPy, Plotly, requests.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add a Twelve Data key; FRED key optional
streamlit run src/app.py
```

## Future improvements

- Pull a live option chain and compare the model price against an actual market quote, not just a live spot.
- Back out implied volatility from that live quote instead of typing volatility in by hand.
- American option support via Least Squares Monte Carlo.

## Status

Scaffolded and verified: Black-Scholes pricing checked against a known textbook benchmark, app starts cleanly. Not yet run against a live Twelve Data key.
