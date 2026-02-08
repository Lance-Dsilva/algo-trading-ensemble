# algo-trading-ensemble

This project implements an algorithmic trading backtester that utilizes a ensemble approach. It combines multiple technical and statistical indicators (Moving Averages, RSI, and ARIMA) and dynamically adjusts their voting weights based on the detected market regime (Bull, Bear, or Sideways).

## Project Structure

*   **`mock_trading.py`**: The main entry point. Simulates a portfolio backtest across multiple tickers.
*   **`ensemble_strategy.py`**: Contains the logic for detecting market regimes and combining signals.
*   **`moving_average.py`**: Implements the Moving Average Crossover strategy.
*   **`rsi_strategy.py`**: Implements the Relative Strength Index (RSI) strategy.
*   **`arima_strategy.py`**: Implements ARIMA time-series forecasting.
*   **`fetch_data.py`**: Utility to download stock data via `yfinance`.

## How It Works

1.  **Signal Generation**: For every day and every ticker, three distinct signals are generated:
    *   **MA**: Trend following (Golden Cross).
    *   **RSI**: Mean reversion (Overbought/Oversold).
    *   **ARIMA**: Statistical price prediction.
2.  **Regime Detection**: The market is classified into **Bull**, **Bear**, or **Sideways** based on rolling volatility and returns.
3.  **Weighted Voting**:
    *   *Bull Market*: Favors Moving Averages.
    *   *Bear/Sideways*: Favors RSI and ARIMA.
4.  **Execution**: The script simulates a portfolio, managing cash and positions based on the ensemble's final consensus.

## Usage

Run the `mock_trading.py` script from the terminal. You can specify tickers, date range, initial capital, and allocation.

```bash
python mock_trading.py \
  --tickers AAPL MSFT GOOGL \
  --start_date 2020-01-01 \
  --end_date 2024-01-01 \
  --initial_capital 10000 \
  --alloc AAPL=0.4 MSFT=0.3 GOOGL=0.3
```

### Arguments
*   `--tickers`: List of stock ticker symbols (space-separated).
*   `--start_date`: Backtest start date (YYYY-MM-DD).
*   `--end_date`: Backtest end date (YYYY-MM-DD).
*   `--initial_capital`: Starting cash in USD.
*   `--alloc`: (Optional) Portfolio allocation weights (e.g., `AAPL=0.4`). Defaults to equal weight.
*   `--position_size_pct`: Percentage of allocated budget to use per trade (default 0.95).

## Example Results

Below is an example output from a backtest run on AAPL, MSFT, and GOOGL from 2020 to 2024.

```text
(lab4) (base) lancedsilva@Lances-MacBook-Air week4 % python mock_trading.py --tickers AAPL MSFT GOOGL --start_date 2020-01-01 --end_date 2024-01-01 --initial_capital 10000 --alloc AAPL=0.4 MSFT=0.3 GOOGL=0.3
Fetching data for AAPL from 2020-01-01 to 2024-01-01...
Data fetched: 1006 rows
Fetching data for MSFT from 2020-01-01 to 2024-01-01...
Data fetched: 1006 rows
Fetching data for GOOGL from 2020-01-01 to 2024-01-01...
Data fetched: 1006 rows
...

PORTFOLIO RESULTS (Regime-Based Ensemble)

Tickers         : AAPL, MSFT, GOOGL
Allocation      : AAPL=40.0%, MSFT=30.0%, GOOGL=30.0%
Period          : 2020-01-01 to 2024-01-01
Trading Days    : 1005

Initial Capital  : $   10,000.00
Final Value      : $   13,437.87
Profit / Loss    : $    3,437.87

Total Return     :    34.38%
Annualized Return:     7.69%
Sharpe Ratio     :    0.645
Total Trades     :       68

FINAL HOLDINGS:
  AAPL: 0 shares
  MSFT: 0 shares
  GOOGL: 0 shares
  Cash remaining: $13,437.87
```

## Requirements

*   Python 3.8+
*   pandas
*   numpy
*   yfinance
*   statsmodels