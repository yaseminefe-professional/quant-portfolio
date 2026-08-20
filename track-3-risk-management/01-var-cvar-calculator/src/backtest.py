"""Kupiec proportion-of-failures backtest for a rolling VaR model.

A VaR model isn't validated by computing it once. It's validated by
rolling it forward through history and checking whether the realized
breach rate matches the rate the model itself predicted.
"""

import numpy as np
from scipy.stats import chi2

from var_models import historical_var_cvar, monte_carlo_var_cvar, parametric_var_cvar, portfolio_returns


def historical_var_fn(train_returns, weights, confidence):
    return historical_var_cvar(portfolio_returns(train_returns, weights), confidence)


def parametric_var_fn(train_returns, weights, confidence):
    return parametric_var_cvar(portfolio_returns(train_returns, weights), confidence)


def monte_carlo_var_fn(train_returns, weights, confidence, n_sims=3000):
    return monte_carlo_var_cvar(train_returns, weights, confidence, n_sims=n_sims)


def rolling_var_breaches(returns, weights, var_fn, window=250, confidence=0.95):
    """Roll a VaR model forward: at each day t, estimate VaR from the
    prior `window` days only (no look-ahead), then check whether day t's
    realized loss breached that prediction.

    Returns (breaches, var_series) as numpy arrays, one entry per day
    tested.
    """
    weights = np.asarray(weights)
    port_returns = portfolio_returns(returns, weights)
    n = len(returns)

    breaches, var_series = [], []
    for t in range(window, n):
        train_window = returns.iloc[t - window:t]
        var, _ = var_fn(train_window, weights, confidence)
        realized = port_returns[t]
        breaches.append(1 if -realized > var else 0)
        var_series.append(var)

    return np.array(breaches), np.array(var_series)


def kupiec_test(breaches, confidence=0.95):
    """Likelihood-ratio test of whether the observed breach rate matches
    the model's stated (1 - confidence) exception rate.

    Null hypothesis: the true breach probability equals p = 1 - confidence.
    Rejecting at p_value < 0.05 means the model is miscalibrated, too
    aggressive if it breaches more than expected, too conservative if less.
    """
    n = len(breaches)
    x = int(breaches.sum())
    p = 1 - confidence

    pi_hat = x / n
    ll_null = x * np.log(p) + (n - x) * np.log(1 - p)
    ll_alt = (
        x * np.log(pi_hat) + (n - x) * np.log(1 - pi_hat)
        if 0 < pi_hat < 1
        else 0.0
    )
    lr_stat = -2 * (ll_null - ll_alt)
    p_value = 1 - chi2.cdf(lr_stat, df=1)

    return {
        "n_obs": n,
        "n_breaches": x,
        "breach_rate": x / n,
        "expected_rate": p,
        "lr_stat": lr_stat,
        "p_value": p_value,
        "reject_model": p_value < 0.05,
    }
