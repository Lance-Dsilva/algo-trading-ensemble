import argparse
import pandas as pd
import numpy as np
from fetch_data import fetch_stock_data

try:
    from statsmodels.tsa.arima.model import ARIMA
except Exception as e: 
    ARIMA = None


def generate_arima_signals(
    data: pd.DataFrame,
    train_window: int = 252,
    order=(1, 1, 1),
    threshold: float = 0.0,
) -> pd.DataFrame:
    out = data.copy()
    close = out["Close"].astype(float)

    pred_next = np.full(len(out), np.nan, dtype=float)
    signal = np.zeros(len(out), dtype=int)

    for i in range(train_window, len(out) - 1):
        train_series = close.iloc[i - train_window : i].dropna()
        if len(train_series) < train_window * 0.9:
            continue

        try:
            model = ARIMA(train_series.reset_index(drop=True), order=order)
            fit = model.fit()
            forecast = float(fit.forecast(steps=1).iloc[0])
            pred_next[i] = forecast

            last_close = float(close.iloc[i - 1])
            pred_ret = (forecast - last_close) / (last_close + 1e-12)

            signal[i] = 1 if pred_ret > threshold else 0
        except Exception:
            signal[i] = 0

    out["ARIMA_Pred_Next_Close"] = pred_next
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
    parser = argparse.ArgumentParser(description="ARIMA Forecast Trading Strategy")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--train_window", type=int, default=252)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--d", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker, args.start_date, args.end_date)
    df = generate_arima_signals(
        df,
        train_window=args.train_window,
        order=(args.p, args.d, args.q),
        threshold=args.threshold,
    )
    df = backtest_strategy(df, args.initial_capital)

    df.to_csv(f"{args.ticker}_arima_results.csv")


if __name__ == "__main__":
    main()
