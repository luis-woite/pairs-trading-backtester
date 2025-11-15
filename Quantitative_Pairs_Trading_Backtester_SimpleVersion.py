import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from scipy import stats



# =========================================
# helper: download prices
# =========================================
def fetch_prices(ticker_a, ticker_b, start, end):
    """Download daily close prices for 2 tickers and drop missing days."""
    data = yf.download([ticker_a, ticker_b], start=start, end=end, progress=False)["Close"]
    data = data.dropna()
    return data


# =========================================
# helper: regression hedge ratio
# =========================================
def calc_hedge_ratio_regression(data, ticker_a, ticker_b):
    """
    Run OLS: A = beta0 + beta1 * B
    Return intercept and hedge ratio.
    """
    X = sm.add_constant(data[ticker_b])   # predictor with intercept
    y = data[ticker_a]                    # target
    model = sm.OLS(y, X).fit()

    intercept = model.params["const"]
    hedge_ratio = model.params[ticker_b]

    print("\n--- Regression summary ---")
    print(model.summary())
    print("Intercept (beta0):", intercept)
    print("Hedge ratio (beta1):", hedge_ratio)

    return intercept, hedge_ratio


# =========================================
# helper: build spread (residual)
# =========================================
def build_spread(data, ticker_a, ticker_b, intercept, hedge_ratio):
    """Spread = actual A - predicted A."""
    predicted_a = intercept + hedge_ratio * data[ticker_b]
    spread = data[ticker_a] - predicted_a
    return spread


# =========================================
# helper: positions from z-score
# =========================================
def build_positions_from_zscore(spread, entry_z=2.0, exit_z=0.5):
    """
    Turn spread into trading positions using z-score rules.
    entry_z: how far it must go to open a trade
    exit_z: how close to 0 to close the trade
    """
    spread_mean = spread.mean()
    spread_std = spread.std()
    zscore = (spread - spread_mean) / spread_std

    position = pd.Series(0, index=spread.index)
    current_pos = 0

    for date in spread.index:
        z = zscore.loc[date]

        # enter
        if z < -entry_z:
            current_pos = 1
        elif z > entry_z:
            current_pos = -1
        else:
            # exit if we are back near the mean
            if current_pos != 0 and abs(z) < exit_z:
                current_pos = 0

        position.loc[date] = current_pos

    return position, zscore, spread_mean, spread_std


# =========================================
# helper: backtest
# =========================================
def backtest(spread, position, transaction_cost=0.2):
    """
    Use positions and spread changes to compute daily P&L and cumulative P&L.
    """
    spread_change = spread.diff()
    pnl = position.shift(1) * spread_change
    pnl = pnl.fillna(0)

    # transaction costs whenever we change position
    position_change = position.diff().fillna(0)
    trade_days = position_change != 0
    pnl[trade_days] = pnl[trade_days] - transaction_cost

    cumulative_pnl = pnl.cumsum()
    return pnl, cumulative_pnl


# =========================================
# helper: performance metrics
# =========================================
def performance_metrics(daily_pnl: pd.Series):
    """
    Compute simple trading performance metrics, including a t-test
    on the daily P&L mean (H0: mean = 0).
    """
    pnl_values = daily_pnl.values
    n = len(pnl_values)

    if n == 0:
        return {"Error": "No P&L data"}

    # total return in spread points
    total_return = pnl_values.sum()

    # win rate
    win_rate = np.mean(pnl_values > 0)

    # Sharpe ratio
    pnl_std = pnl_values.std()
    sharpe = (pnl_values.mean() / pnl_std) * np.sqrt(252) if pnl_std else 0.0

    # max drawdown
    cum = np.cumsum(pnl_values)
    running_max = np.maximum.accumulate(cum)
    max_drawdown = np.max(running_max - cum) if len(cum) > 0 else 0.0

    # ---- 🧮 NEW: one-sample t-test ----
    t_stat, p_value = stats.ttest_1samp(pnl_values, 0)

    return {
        "Total Return": round(total_return, 2),
        "Win Rate": f"{win_rate:.1%}",
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(float(max_drawdown), 2),
        "t-statistic": round(float(t_stat), 2),
        "p-value": f"{p_value:.4f}",
    }



# =========================================
# helper: plotting
# =========================================
def plot_spread_with_signals(spread, zscore, spread_mean, spread_std, entry_z, ticker_a, ticker_b):
    plt.figure(figsize=(10, 5))
    plt.plot(spread, label="Spread (residual)", color="purple")
    plt.axhline(spread_mean, color="green", linestyle="--", label="Mean")
    plt.axhline(spread_mean + spread_std, color="red", linestyle="--", label="+1 std")
    plt.axhline(spread_mean - spread_std, color="red", linestyle="--", label="-1 std")

    buy_points = zscore < -entry_z
    sell_points = zscore > entry_z
    plt.scatter(spread[buy_points].index, spread[buy_points], color="green", marker="^", label="Buy", alpha=0.9)
    plt.scatter(spread[sell_points].index, spread[sell_points], color="red", marker="v", label="Sell", alpha=0.9)

    plt.title(f"Spread and signals: {ticker_a} vs {ticker_b}")
    plt.legend()
    plt.show()


def plot_pnl(cumulative_pnl):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_pnl, label="Cumulative P&L")
    plt.title("Pairs Trading Backtest")
    plt.xlabel("Date")
    plt.ylabel("P&L (spread points)")
    plt.legend()
    plt.show()


# =========================================
# main function
# =========================================
def run_pairs_backtest(
    ticker_a: str,
    ticker_b: str,
    start: str = "2024-01-01",
    end: str = "2024-06-01",
    entry_z: float = 1.0,
    exit_z: float = 0.3,
    transaction_cost: float = 0.2,
):
    print(f"\n=== Running pairs backtest for {ticker_a} vs {ticker_b} ===")

    # 1) data
    data = fetch_prices(ticker_a, ticker_b, start, end)

    # 2) regression hedge ratio
    intercept, hedge_ratio = calc_hedge_ratio_regression(data, ticker_a, ticker_b)

    # 3) spread (residual)
    spread = build_spread(data, ticker_a, ticker_b, intercept, hedge_ratio)

    # 4) positions
    position, zscore, spread_mean, spread_std = build_positions_from_zscore(
        spread, entry_z=entry_z, exit_z=exit_z
    )

    # 5) backtest
    daily_pnl, cumulative_pnl = backtest(spread, position, transaction_cost=transaction_cost)

    # 6) plots
    plot_spread_with_signals(spread, zscore, spread_mean, spread_std, entry_z, ticker_a, ticker_b)
    plot_pnl(cumulative_pnl)

    # 7) summary
    total_trades = (position.diff().fillna(0) != 0).sum()
    print("\n===== SUMMARY =====")
    print(f"Pair: {ticker_a} vs {ticker_b}")
    print(f"Dates: {start} to {end}")
    print(f"Hedge ratio (beta1): {hedge_ratio:.4f}")
    print(f"Final P&L: {cumulative_pnl.iloc[-1]:.2f}")
    print(f"Position changes (entries + exits): {total_trades}")

    # 8) performance metrics
    metrics = performance_metrics(daily_pnl)
    print("\nPerformance Metrics:")
    for k, v in metrics.items():
        print(f"  {k:<15}: {v}")

    return {
        "data": data,
        "spread": spread,
        "zscore": zscore,
        "position": position,
        "daily_pnl": daily_pnl,
        "cumulative_pnl": cumulative_pnl,
    }


# =========================================
# run from command line
# =========================================
if __name__ == "__main__":
    # tweak these if you like
    run_pairs_backtest(
        "XOM",
        "CVX",
        start="2024-01-01",
        end="2024-06-01",
        entry_z=1.0,
        exit_z=0.3,
        transaction_cost=0.2,
    )
    input("Press Enter to exit...")


    







