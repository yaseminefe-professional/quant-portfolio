"""CLI: pull a live portfolio, report VaR/CVaR three ways, backtest each.

Example:
    python src/main.py --symbols AAPL MSFT GOOGL --weights 0.4 0.35 0.25
"""

import argparse

import numpy as np

from backtest import historical_var_fn, kupiec_test, monte_carlo_var_fn, parametric_var_fn, rolling_var_breaches
from data import fetch_prices, to_returns
from var_models import historical_var_cvar, monte_carlo_var_cvar, parametric_var_cvar, portfolio_returns


def parse_args():
    parser = argparse.ArgumentParser(description="VaR/CVaR calculator with Kupiec backtesting")
    parser.add_argument("--symbols", nargs="+", required=True, help="Ticker symbols, e.g. AAPL MSFT GOOGL")
    parser.add_argument("--weights", nargs="+", type=float, required=True, help="Portfolio weights, must sum to 1.0")
    parser.add_argument("--confidence", nargs="+", type=float, default=[0.95, 0.99])
    parser.add_argument("--outputsize", type=int, default=500, help="Trading days of history to pull")
    parser.add_argument("--backtest-window", type=int, default=250, help="Rolling estimation window for backtesting")
    parser.add_argument("--plot", action="store_true", help="Save a PNG of returns vs VaR breaches")
    return parser.parse_args()


def main():
    args = parse_args()

    if len(args.symbols) != len(args.weights):
        raise SystemExit("Number of --symbols and --weights must match.")
    if not np.isclose(sum(args.weights), 1.0):
        raise SystemExit(f"Weights must sum to 1.0, got {sum(args.weights)}")

    print(f"Fetching {args.outputsize} daily bars for {', '.join(args.symbols)} from Twelve Data...")
    prices = fetch_prices(args.symbols, outputsize=args.outputsize)
    returns = to_returns(prices)
    port_returns = portfolio_returns(returns, args.weights)
    print(f"{len(returns)} trading days pulled, {returns.index.min().date()} to {returns.index.max().date()}\n")

    for confidence in args.confidence:
        print(f"=== {confidence:.0%} confidence ===")
        h_var, h_cvar = historical_var_cvar(port_returns, confidence)
        p_var, p_cvar = parametric_var_cvar(port_returns, confidence)
        m_var, m_cvar = monte_carlo_var_cvar(returns, args.weights, confidence)

        print(f"{'Method':<14}{'VaR':>10}{'CVaR':>10}")
        print(f"{'Historical':<14}{h_var:>10.2%}{h_cvar:>10.2%}")
        print(f"{'Parametric':<14}{p_var:>10.2%}{p_cvar:>10.2%}")
        print(f"{'Monte Carlo':<14}{m_var:>10.2%}{m_cvar:>10.2%}")

        min_history = args.backtest_window + 30
        if len(returns) > min_history:
            print("\nBacktest (Kupiec proportion-of-failures test):")
            methods = [
                ("Historical", historical_var_fn),
                ("Parametric", parametric_var_fn),
                ("Monte Carlo", monte_carlo_var_fn),
            ]
            for name, fn in methods:
                breaches, _ = rolling_var_breaches(returns, args.weights, fn, window=args.backtest_window, confidence=confidence)
                result = kupiec_test(breaches, confidence)
                verdict = "REJECTED" if result["reject_model"] else "not rejected"
                print(
                    f"  {name:<14} breaches {result['n_breaches']:>3}/{result['n_obs']:<4} "
                    f"({result['breach_rate']:.2%} vs {result['expected_rate']:.2%} expected), "
                    f"LR p-value {result['p_value']:.3f} -> {verdict}"
                )
        else:
            print(f"\n(Need > {min_history} trading days for a meaningful backtest; raise --outputsize.)")
        print()

    if args.plot:
        save_plot(returns, port_returns)


def save_plot(returns, port_returns):
    import matplotlib.pyplot as plt

    var_95, _ = historical_var_cvar(port_returns, 0.95)
    breach_mask = port_returns < -var_95

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(returns.index, port_returns, color="#333333", linewidth=0.8, label="Portfolio daily return")
    ax.axhline(-var_95, color="#a8672a", linestyle="--", linewidth=1, label="95% historical VaR")
    ax.scatter(returns.index[breach_mask], port_returns[breach_mask], color="#a8672a", zorder=5, s=18, label="Breach")
    ax.set_title("Portfolio returns vs 95% historical VaR")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig("var_breaches.png", dpi=150)
    print("Saved plot to var_breaches.png")


if __name__ == "__main__":
    main()
