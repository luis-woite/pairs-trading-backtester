
---

### B. `pairs-trading-backtester/README.md`

```markdown
# Pairs Trading Backtester (Python)

Simple mean-reversion pairs trading engine built around regression, spreads
and z-score signals.

## What it does

- Downloads daily close prices for two tickers (`yfinance`)
- Estimates a hedge ratio via OLS regression (A on B)
- Builds the residual spread and standardises it to a z-score
- Generates trading signals:
  - enter long/short when z-score moves beyond an entry threshold
  - close the position when z-score mean-reverts
- Backtests the strategy including:
  - position changes
  - simple transaction costs per trade
  - daily P&L and cumulative P&L
- Computes summary metrics:
  - total return, Sharpe ratio, win rate, max drawdown
  - t-test of daily P&L vs zero
- Plots the spread with buy/sell markers and the cumulative P&L.

## Files

- `pairs_backtester.py` – main script and helper functions
- `requirements.txt` – Python dependencies
- `.gitignore` – ignore compiled files, environments and data

## How to run

```bash
pip install -r requirements.txt
python pairs_backtester.py
