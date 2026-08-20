"""Historical, parametric, and Monte Carlo VaR / CVaR on a weighted portfolio.

VaR and CVaR are both reported as positive numbers: the loss magnitude at
the given confidence level, not a signed return. A 95% daily VaR of 2.1%
means "on 95% of days, the portfolio should not lose more than 2.1%."
"""

import numpy as np
from scipy.stats import norm


def portfolio_returns(returns, weights):
    """Weighted daily return series for the portfolio, as a numpy array."""
    weights = np.asarray(weights)
    return returns.values @ weights


def historical_var_cvar(port_returns, confidence=0.95):
    """Empirical quantile VaR, no distributional assumption. CVaR is the
    mean loss among the days that breached VaR.
    """
    alpha = 1 - confidence
    var = -np.percentile(port_returns, alpha * 100)
    tail = port_returns[port_returns <= -var]
    cvar = -tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def parametric_var_cvar(port_returns, confidence=0.95):
    """Variance-covariance VaR: assumes portfolio returns are normal.
    Diverges from the historical estimate when real returns are
    fat-tailed or skewed, which is worth reporting alongside the number.
    """
    mu, sigma = port_returns.mean(), port_returns.std(ddof=1)
    alpha = 1 - confidence
    z = norm.ppf(alpha)
    var = -(mu + z * sigma)
    cvar = -(mu - sigma * norm.pdf(z) / alpha)
    return float(var), float(cvar)


def monte_carlo_var_cvar(returns, weights, confidence=0.95, n_sims=10000, seed=42):
    """Simulate forward portfolio returns from the estimated asset-level
    covariance matrix, then take the empirical quantile of the simulated
    distribution. Uses a multivariate normal here; swapping in a
    Student-t or bootstrapped simulator is the natural next step once
    this baseline is validated.
    """
    rng = np.random.default_rng(seed)
    mu = returns.mean().values
    cov = returns.cov().values
    sims = rng.multivariate_normal(mu, cov, size=n_sims)
    sim_port_returns = sims @ np.asarray(weights)
    return historical_var_cvar(sim_port_returns, confidence)
