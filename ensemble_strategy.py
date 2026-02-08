import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from fetch_data import fetch_stock_data
from moving_average import calculate_moving_averages, generate_signals as generate_ma_signals
from rsi_strategy import generate_rsi_signals
from arima_strategy import generate_arima_signals


def compute_mae_rmse(y_true: pd.Series, y_pred: pd.Series) -> dict:
    aligned = pd.concat([y_true.rename("true"), y_pred.rename("pred")], axis=1).dropna()
    if aligned.empty:
        return {"mae": np.nan, "rmse": np.nan, "n": 0}

    err = aligned["pred"] - aligned["true"]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(aligned))}


def backtest_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    out = df.copy()
    out["Daily_Return"] = out["Close"].pct_change()
    out["Strategy_Return"] = out["Signal"].shift(1) * out["Daily_Return"]
    out["Strategy_Cumulative"] = (1 + out["Strategy_Return"].fillna(0)).cumprod()

    final_value = float(initial_capital * out["Strategy_Cumulative"].iloc[-1])
    profit = float(final_value - initial_capital)
    return_pct = float((out["Strategy_Cumulative"].iloc[-1] - 1) * 100)

    strategy_returns = out["Strategy_Return"].dropna()
    sharpe = float((strategy_returns.mean() / (strategy_returns.std() + 1e-10)) * np.sqrt(252)) if len(strategy_returns) else 0.0

    cumulative = out["Strategy_Cumulative"]
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = float(drawdown.min() * 100)

    nonzero = strategy_returns[strategy_returns != 0]
    win_rate = float((nonzero > 0).mean() * 100) if len(nonzero) else 0.0

    return {
        "final_value": final_value,
        "profit": profit,
        "return_pct": return_pct,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def detect_regime(close: pd.Series, window: int = 50) -> pd.Series:
    rolling_return = close.pct_change(window)
    rolling_std = close.pct_change().rolling(window=window).std()

    regime = pd.Series("sideways", index=close.index, dtype="object")
    regime.loc[rolling_return > 0.05] = "bull"
    regime.loc[rolling_return < -0.05] = "bear"
    regime.loc[(rolling_std > 0.015) & (rolling_return.abs() < 0.05)] = "sideways"
    return regime.ffill().bfill()


def regime_based_ensemble(ma_sig: pd.Series, rsi_sig: pd.Series, arima_sig: pd.Series, regime: pd.Series) -> pd.Series:
    signals = pd.DataFrame({"MA": ma_sig, "RSI": rsi_sig, "ARIMA": arima_sig})

    weights = pd.DataFrame(index=regime.index)
    weights["MA"] = 0.33
    weights["RSI"] = 0.33
    weights["ARIMA"] = 0.34

    bull = regime == "bull"
    weights.loc[bull, "MA"] = 0.45
    weights.loc[bull, "RSI"] = 0.15
    weights.loc[bull, "ARIMA"] = 0.40

    bear = regime == "bear"
    weights.loc[bear, "MA"] = 0.20
    weights.loc[bear, "RSI"] = 0.50
    weights.loc[bear, "ARIMA"] = 0.30

    sideways = regime == "sideways"
    weights.loc[sideways, "MA"] = 0.25
    weights.loc[sideways, "RSI"] = 0.40
    weights.loc[sideways, "ARIMA"] = 0.35

    score = (signals * weights).sum(axis=1)
    return (score >= 0.5).astype(int)


def main():
    parser = argparse.ArgumentParser(description="Regime-Based Ensemble Strategy")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    args = parser.parse_args()

    df = fetch_stock_data(args.ticker, args.start_date, args.end_date)

    close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    returns = close.pct_change()

    ma_df = calculate_moving_averages(df.copy(), 50, 200)
    ma_df = generate_ma_signals(ma_df)
    ma_sig = ma_df["Signal"].astype(float)

    rsi_df = generate_rsi_signals(df.copy(), 14, 30.0, 70.0)
    rsi_sig = rsi_df["Signal"].astype(float)

    arima_df = generate_arima_signals(df.copy(), train_window=252, order=(1, 1, 1))
    arima_sig = arima_df["Signal"].astype(float)

    if "ARIMA_Pred_Next_Close" in arima_df.columns:
        arima_pred_next = arima_df["ARIMA_Pred_Next_Close"].astype(float)
        actual_next = close.shift(-1).astype(float)
        forecast_metrics = compute_mae_rmse(actual_next, arima_pred_next)
    else:
        forecast_metrics = {"mae": np.nan, "rmse": np.nan, "n": 0}

    regime = detect_regime(close, window=50)
    ensemble_signal = regime_based_ensemble(ma_sig, rsi_sig, arima_sig, regime)

    test_df = df.copy()
    test_df["Signal"] = ensemble_signal
    perf = backtest_metrics(test_df, args.initial_capital)

    print("REGIME-BASED ENSEMBLE RESULTS")
    print(f"Ticker: {args.ticker}")
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${perf['final_value']:,.2f}")
    print(f"Profit: ${perf['profit']:,.2f}")
    print(f"Return: {perf['return_pct']:.2f}%")
    print(f"Sharpe Ratio: {perf['sharpe']:.2f}")
    print(f"Max Drawdown: {perf['max_drawdown']:.2f}%")
    print(f"Win Rate: {perf['win_rate']:.2f}%")
    print(f"ARIMA Forecast MAE: {forecast_metrics['mae']:.4f} (N={forecast_metrics['n']})")
    print(f"ARIMA Forecast RMSE: {forecast_metrics['rmse']:.4f} (N={forecast_metrics['n']})")


if __name__ == "__main__":
    main()
