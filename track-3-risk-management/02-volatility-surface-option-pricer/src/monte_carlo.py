"""Monte Carlo option pricing under geometric Brownian motion.

Prices are estimated by simulating terminal asset prices under the
risk-neutral measure and discounting the average payoff. `simulate_paths`
generates full intermediate paths for visualization only, it is not used
in the price estimate itself.
"""

import numpy as np


def simulate_terminal_prices(S, T, r, q, sigma, n_sims=10000, seed=None):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sims)
    return S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * z)


def simulate_paths(S, T, r, q, sigma, n_paths=20, n_steps=100, seed=None):
    """Full price paths for the sample-path chart. Shape (n_paths, n_steps + 1)."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(log_increments, axis=1)
    return S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def mc_price(S, K, T, r, q, sigma, option_type="call", n_sims=10000, seed=None):
    """Returns (price, standard_error, terminal_prices)."""
    terminal = simulate_terminal_prices(S, T, r, q, sigma, n_sims, seed)
    if option_type == "call":
        payoff = np.maximum(terminal - K, 0.0)
    else:
        payoff = np.maximum(K - terminal, 0.0)
    discounted = np.exp(-r * T) * payoff
    return discounted.mean(), discounted.std(ddof=1) / np.sqrt(n_sims), terminal
