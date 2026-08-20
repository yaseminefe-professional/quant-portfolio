"""Streamlit app: options pricing, Greeks, and sensitivity analysis.

Run with: streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from black_scholes import FIRST_ORDER_GREEKS, SECOND_ORDER_GREEKS, bs_price
from live_data import fetch_risk_free_rate, fetch_spot
from monte_carlo import mc_price, simulate_paths
from surfaces import AXIS_LABELS, SURFACE_AXES, compute_surface

st.set_page_config(page_title="Options Pricing & Greeks", layout="wide")

# ---------------------------------------------------------------- sidebar --

st.sidebar.header("Underlying price")
price_source = st.sidebar.radio("Source", ["Live (Twelve Data)", "Manual"], label_visibility="collapsed")

if price_source == "Live (Twelve Data)":
    symbol = st.sidebar.text_input("Ticker", value="AAPL")
    if st.sidebar.button("Fetch live price"):
        try:
            st.session_state["quote"] = fetch_spot(symbol)
        except Exception as exc:
            st.sidebar.error(str(exc))
    quote = st.session_state.get("quote")
    if quote:
        S = quote["price"]
        st.sidebar.caption(f"{quote['symbol']}: {quote['price']:.2f} ({quote['percent_change']:+.2f}%) as of {quote['datetime']}")
    else:
        S = st.sidebar.number_input("Underlying price (S)", value=100.0, min_value=0.01)
        st.sidebar.caption("Click 'Fetch live price' to pull a real quote.")
else:
    S = st.sidebar.number_input("Underlying price (S)", value=100.0, min_value=0.01)

st.sidebar.header("Option parameters")
option_type = st.sidebar.radio("Type", ["call", "put"], horizontal=True)
K = st.sidebar.number_input("Strike price (K)", value=round(S, 2), min_value=0.01)
days_to_expiry = st.sidebar.number_input("Days to expiration", value=30, min_value=1)
T = days_to_expiry / 365
sigma = st.sidebar.number_input("Volatility (%)", value=20.0, min_value=0.1) / 100
q = st.sidebar.number_input("Dividend yield (%)", value=0.0, min_value=0.0) / 100

st.sidebar.header("Risk-free rate")
rf_source = st.sidebar.radio("Rate source", ["Live (FRED)", "Manual"], label_visibility="collapsed")
if rf_source == "Live (FRED)":
    if st.sidebar.button("Fetch live rate"):
        rate = fetch_risk_free_rate()
        st.session_state["rf_rate"] = rate
    r = st.session_state.get("rf_rate")
    if r is None:
        st.sidebar.caption("Click 'Fetch live rate', or set FRED_API_KEY in .env. Falling back to 4%.")
        r = 0.04
    else:
        st.sidebar.caption(f"3-month US Treasury yield (FRED): {r:.2%}")
else:
    r = st.sidebar.number_input("Risk-free rate (%)", value=4.0, min_value=0.0) / 100

st.sidebar.header("Monte Carlo")
n_sims = st.sidebar.slider("Simulation paths (pricing)", 1000, 50000, 10000, step=1000)
n_display_paths = st.sidebar.slider("Sample paths to display", 1, 50, 20)
if st.sidebar.button("Generate new paths"):
    st.session_state["mc_seed"] = np.random.randint(0, 1_000_000)
seed = st.session_state.get("mc_seed", 42)

params = dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type)

# ------------------------------------------------------------------ header --

st.title("Options pricing & Greeks")
st.caption(
    "Black-Scholes closed form vs. Monte Carlo simulation, first- and second-order Greeks, "
    "and Black-Scholes sensitivity surfaces. Spot price and risk-free rate can be pulled live "
    "instead of typed in."
)

tab_price, tab_sens, tab_mc, tab_greeks, tab_surface = st.tabs(
    ["Pricing", "Sensitivity", "Monte Carlo", "Greeks", "3D surfaces"]
)

# ------------------------------------------------------------------ pricing --

with tab_price:
    bs_c = bs_price(S, K, T, r, q, sigma, "call")
    bs_p = bs_price(S, K, T, r, q, sigma, "put")
    mc_c, mc_c_se, _ = mc_price(S, K, T, r, q, sigma, "call", n_sims=n_sims, seed=seed)
    mc_p, mc_p_se, _ = mc_price(S, K, T, r, q, sigma, "put", n_sims=n_sims, seed=seed)

    col1, col2, col3, col4 = st.columns(4)
    bs_val = bs_c if option_type == "call" else bs_p
    mc_val = mc_c if option_type == "call" else mc_p
    mc_se = mc_c_se if option_type == "call" else mc_p_se
    col1.metric("Black-Scholes price", f"{bs_val:.4f}")
    col2.metric("Monte Carlo price", f"{mc_val:.4f}", f"±{1.96 * mc_se:.4f} (95% CI)")
    diff_pct = (mc_val - bs_val) / bs_val * 100 if bs_val else 0
    col3.metric("Difference", f"{diff_pct:+.3f}%")
    col4.metric("Simulations", f"{n_sims:,}")

    fig = go.Figure()
    fig.add_bar(name="Black-Scholes", x=["Call", "Put"], y=[bs_c, bs_p], marker_color="#1f5e4c")
    fig.add_bar(name="Monte Carlo", x=["Call", "Put"], y=[mc_c, mc_p], marker_color="#a8672a")
    fig.update_layout(barmode="group", title="Black-Scholes vs Monte Carlo price", height=420)
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------- sensitivity --

with tab_sens:
    def sensitivity_line(x_vals, x_param, x_label):
        rows = []
        for x in x_vals:
            p = dict(params)
            p[x_param] = x
            p["option_type"] = "call"
            call_price = bs_price(**p)
            p["option_type"] = "put"
            put_price = bs_price(**p)
            rows.append({x_label: x, "Call": call_price, "Put": put_price})
        df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_scatter(x=df[x_label], y=df["Call"], name="Call", line=dict(color="#1f5e4c"))
        fig.add_scatter(x=df[x_label], y=df["Put"], name="Put", line=dict(color="#a8672a"))
        fig.update_layout(title=f"Price vs {x_label}", xaxis_title=x_label, yaxis_title="Option price", height=360)
        st.plotly_chart(fig, use_container_width=True)

    sensitivity_line(np.linspace(0.05, 1.0, 40), "sigma", "Volatility")
    sensitivity_line(np.linspace(1 / 365, 2.0, 40), "T", "Time to expiration (years)")
    sensitivity_line(np.linspace(S * 0.5, S * 1.5, 40), "K", "Strike price")

# ----------------------------------------------------------------- monte carlo --

with tab_mc:
    paths = simulate_paths(S, T, r, q, sigma, n_paths=n_display_paths, n_steps=100, seed=seed)
    time_axis = np.linspace(0, T * 365, paths.shape[1])

    fig_paths = go.Figure()
    for path in paths:
        fig_paths.add_scatter(x=time_axis, y=path, mode="lines", line=dict(width=1), showlegend=False, opacity=0.6)
    fig_paths.add_hline(y=K, line_dash="dash", line_color="#a8672a", annotation_text="Strike")
    fig_paths.update_layout(title=f"{n_display_paths} sample GBM price paths", xaxis_title="Days", yaxis_title="Price", height=420)
    st.plotly_chart(fig_paths, use_container_width=True)

    _, _, terminal = mc_price(S, K, T, r, q, sigma, option_type, n_sims=n_sims, seed=seed)
    fig_hist = go.Figure()
    fig_hist.add_histogram(x=terminal, nbinsx=60, marker_color="#1f5e4c")
    fig_hist.add_vline(x=K, line_dash="dash", line_color="#a8672a", annotation_text="Strike")
    fig_hist.add_vline(x=terminal.mean(), line_dash="dot", line_color="#4b5750", annotation_text="Mean")
    fig_hist.update_layout(title=f"Distribution of simulated terminal prices ({n_sims:,} paths)", xaxis_title="Terminal price", height=360)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.caption(
        f"Mean {terminal.mean():.2f} · Std dev {terminal.std():.2f} · "
        f"5th pct {np.percentile(terminal, 5):.2f} · 95th pct {np.percentile(terminal, 95):.2f}"
    )

# --------------------------------------------------------------------- greeks --

with tab_greeks:
    first_rows = []
    for name, fn in FIRST_ORDER_GREEKS.items():
        first_rows.append({"Greek": name, "Call": fn(S, K, T, r, q, sigma, "call"), "Put": fn(S, K, T, r, q, sigma, "put")})
    first_df = pd.DataFrame(first_rows).set_index("Greek")

    second_rows = []
    for name, fn in SECOND_ORDER_GREEKS.items():
        second_rows.append({"Greek": name, "Call": fn(S, K, T, r, q, sigma, "call"), "Put": fn(S, K, T, r, q, sigma, "put")})
    second_df = pd.DataFrame(second_rows).set_index("Greek")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("First-order Greeks")
        st.dataframe(first_df.style.format("{:.5f}"), use_container_width=True)
    with col2:
        st.subheader("Second-order Greeks")
        st.dataframe(second_df.style.format("{:.5f}"), use_container_width=True)

    fig_g = go.Figure()
    fig_g.add_bar(name="Call", x=first_df.index, y=first_df["Call"], marker_color="#1f5e4c")
    fig_g.add_bar(name="Put", x=first_df.index, y=first_df["Put"], marker_color="#a8672a")
    fig_g.update_layout(barmode="group", title="First-order Greeks, call vs put", height=380)
    st.plotly_chart(fig_g, use_container_width=True)

# -------------------------------------------------------------------- surfaces --

with tab_surface:
    greek_name = st.selectbox("Greek", list(SURFACE_AXES.keys()))
    x_param = SURFACE_AXES[greek_name]
    x_label = AXIS_LABELS[x_param]
    greek_fn = FIRST_ORDER_GREEKS[greek_name]

    s_range = np.linspace(S * 0.6, S * 1.4, 30)
    ranges = {
        "sigma": np.linspace(0.05, 0.8, 30),
        "T": np.linspace(7 / 365, 2.0, 30),
        "r": np.linspace(0.0, 0.10, 30),
    }
    x_range = ranges[x_param]

    base_params = dict(S=S, K=K, T=T, r=r, q=q, sigma=sigma, option_type=option_type)
    S_grid, X_grid, Z = compute_surface(greek_fn, base_params, s_range, x_range, x_param)

    fig_surf = go.Figure(data=[go.Surface(x=S_grid, y=X_grid, z=Z, colorscale="Viridis")])
    fig_surf.update_layout(
        title=f"{greek_name} surface: spot price vs {x_label}",
        scene=dict(xaxis_title="Spot price", yaxis_title=x_label, zaxis_title=greek_name),
        height=650,
    )
    st.plotly_chart(fig_surf, use_container_width=True)
    st.caption(f"{greek_name} is most sensitive to {x_label.lower()}, so the surface pairs spot price against that parameter rather than a generic grid.")
