"""Closed-form Black-Scholes pricing and Greeks via finite differentiation.

Every Greek here, first order and second order, is computed as a
numerical derivative of the same closed-form bs_price function below,
rather than as a separately hand-derived formula for each one. That
guarantees internal consistency (a bug in one Greek can't silently
diverge from the price it's supposed to describe) at a small, deliberate
cost in numerical precision versus a fully analytical second-order Greek.
"""

import numpy as np
from scipy.stats import norm


def bs_price(S, K, T, r, q, sigma, option_type="call"):
    """European option price, dividend-yield adjusted Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def _central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def _second_diff(f, x, h):
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h ** 2)


def _step(x, rel=1e-3, floor=1e-4):
    return max(abs(x) * rel, floor)


# ---- First-order Greeks ----

def delta(S, K, T, r, q, sigma, option_type="call"):
    f = lambda s: bs_price(s, K, T, r, q, sigma, option_type)
    return _central_diff(f, S, _step(S))


def gamma(S, K, T, r, q, sigma, option_type="call"):
    f = lambda s: bs_price(s, K, T, r, q, sigma, option_type)
    return _second_diff(f, S, _step(S))


def vega(S, K, T, r, q, sigma, option_type="call"):
    f = lambda v: bs_price(S, K, T, r, q, v, option_type)
    return _central_diff(f, sigma, _step(sigma))


def theta(S, K, T, r, q, sigma, option_type="call"):
    f = lambda t: bs_price(S, K, t, r, q, sigma, option_type)
    return -_central_diff(f, T, _step(T, floor=1e-5))


def rho(S, K, T, r, q, sigma, option_type="call"):
    f = lambda rate: bs_price(S, K, T, rate, q, sigma, option_type)
    return _central_diff(f, r, _step(r, floor=1e-5))


# ---- Second-order Greeks (derivatives of the first-order Greek functions) ----

def charm(S, K, T, r, q, sigma, option_type="call"):
    f = lambda t: delta(S, K, t, r, q, sigma, option_type)
    return -_central_diff(f, T, _step(T, rel=5e-3, floor=1e-5))


def speed(S, K, T, r, q, sigma, option_type="call"):
    f = lambda s: gamma(s, K, T, r, q, sigma, option_type)
    return _central_diff(f, S, _step(S, rel=5e-3))


def color(S, K, T, r, q, sigma, option_type="call"):
    f = lambda t: gamma(S, K, t, r, q, sigma, option_type)
    return -_central_diff(f, T, _step(T, rel=5e-3, floor=1e-5))


def zomma(S, K, T, r, q, sigma, option_type="call"):
    f = lambda v: gamma(S, K, T, r, q, v, option_type)
    return _central_diff(f, sigma, _step(sigma, rel=5e-3))


def veta(S, K, T, r, q, sigma, option_type="call"):
    f = lambda t: vega(S, K, t, r, q, sigma, option_type)
    return -_central_diff(f, T, _step(T, rel=5e-3, floor=1e-5))


def volga(S, K, T, r, q, sigma, option_type="call"):
    f = lambda v: vega(S, K, T, r, q, v, option_type)
    return _central_diff(f, sigma, _step(sigma, rel=5e-3))


FIRST_ORDER_GREEKS = {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

SECOND_ORDER_GREEKS = {
    "Charm": charm, "Speed": speed, "Color": color,
    "Zomma": zomma, "Veta": veta, "Volga": volga,
}
