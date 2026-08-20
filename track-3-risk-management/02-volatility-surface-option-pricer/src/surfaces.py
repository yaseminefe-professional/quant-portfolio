"""Grid computation for 3D Greek sensitivity surfaces.

Different Greeks are primarily driven by different parameters, so each
surface pairs spot price with the parameter that Greek actually responds
to: Delta, Gamma, and Vega against volatility; Theta against time to
expiration; Rho against the risk-free rate.
"""

import numpy as np

SURFACE_AXES = {
    "Delta": "sigma",
    "Gamma": "sigma",
    "Vega": "sigma",
    "Theta": "T",
    "Rho": "r",
}

AXIS_LABELS = {
    "sigma": "Volatility",
    "T": "Time to expiration (years)",
    "r": "Risk-free rate",
}


def compute_surface(greek_fn, base_params, s_range, x_range, x_param):
    """Evaluate greek_fn over a grid of (spot price, x_param).

    base_params holds the fixed S, K, T, r, q, sigma, option_type;
    s_range varies S and x_range varies whichever parameter x_param
    names. Returns (S_grid, X_grid, Z) ready for a plotly Surface.
    """
    S_grid, X_grid = np.meshgrid(s_range, x_range)
    Z = np.zeros_like(S_grid)
    for i in range(S_grid.shape[0]):
        for j in range(S_grid.shape[1]):
            params = dict(base_params)
            params["S"] = S_grid[i, j]
            params[x_param] = X_grid[i, j]
            Z[i, j] = greek_fn(**params)
    return S_grid, X_grid, Z
