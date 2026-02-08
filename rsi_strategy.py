import argparse
import pandas as pd
import numpy as np
from fetch_data import fetch_stock_data


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_rsi_signals(
    data: pd.DataFrame,
    rsi_window: int = 14,
    buy_threshold: float = 30.0,
    sell_threshold: float = 70.0,
) -> pd.DataFrame:
    out = data.copy()
    out["RSI"] = compute_rsi(out["Close"], window=rsi_window)

    signal = np.zeros(len(out), dtype=int)
    prev = 0
    for i in range(len(out)):
        rsi_val = out["RSI"].iloc[i]
        if np.isnan(rsi_val):
            signal[i] = prev
            continue

        if rsi_val <= buy_threshold:
            prev = 1
        elif rsi_val >= sell_threshold:
            prev = 0
        signal[i] = prev

    out["Signal"] = signal
    out["Position"] = out["Signal"].astype(float).diff()
    return out


def backtest_strategy(data: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
    out = data.copy()
    out["Daily_Return"] = out["Close"].pct_change()
    out["Strategy_Return"] = out["Signal"].shift(1) * out["Daily_Return"]

    out["Market_Cumulative"] = (1 + out["Daily_Return"].fillna(0)).cumprod()
    out["Strategy_Cumulative"] = (1 + out["Strategy_Return"].fillna(0)).cumprod()
    out["Portfolio_Value"] = initial_capital * out["Strategy_Cumulative"]
    return out


def main():
    parser = argparse.ArgumentParser(description="RSI Trading Strategy")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--rsi_window", type=int, default=14)
    parser.add_argument("--buy_threshold", type=float, default=30.0)
    parser.add_argument("--sell_threshold", type=float, default=70.0)
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker, args.start_date, args.end_date)
    df = generate_rsi_signals(df, args.rsi_window, args.buy_threshold, args.sell_threshold)
    df = backtest_strategy(df, args.initial_capital)


if __name__ == "__main__":
    main()
