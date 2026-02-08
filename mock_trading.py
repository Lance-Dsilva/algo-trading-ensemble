import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from fetch_data import fetch_stock_data
from moving_average import calculate_moving_averages, generate_signals as generate_ma_signals
from rsi_strategy import generate_rsi_signals
from arima_strategy import generate_arima_signals
from ensemble_strategy import detect_regime, regime_based_ensemble


def generate_regime_ensemble_signal(df: pd.DataFrame) -> pd.Series:
    close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]

    ma_df = calculate_moving_averages(df.copy(), 50, 200)
    ma_df = generate_ma_signals(ma_df)
    ma_sig = ma_df["Signal"].astype(float)

    rsi_df = generate_rsi_signals(df.copy(), 14, 30.0, 70.0)
    rsi_sig = rsi_df["Signal"].astype(float)

    arima_df = generate_arima_signals(df.copy(), train_window=252, order=(1, 1, 1))
    arima_sig = arima_df["Signal"].astype(float)

    regime = detect_regime(close, window=50)
    signal = regime_based_ensemble(ma_sig, rsi_sig, arima_sig, regime)
    return signal.astype(int)

def normalize_allocation(tickers: List[str], allocation: Dict[str, float] | None) -> Dict[str, float]:
    if allocation is None or len(allocation) == 0:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    alloc = {t: float(allocation.get(t, 0.0)) for t in tickers}
    total = sum(alloc.values())
    if total <= 0:
        w = 1.0 / len(tickers)
        return {t: w for t in tickers}

    return {t: alloc[t] / total for t in tickers}

def backtest_portfolio(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float,
    allocation: Dict[str, float],
    position_size_pct: float = 0.95,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    data: Dict[str, pd.DataFrame] = {}
    signals: Dict[str, pd.Series] = {}

    for t in tickers:
        df = fetch_stock_data(t, start_date, end_date)
        df = df.sort_index()
        sig = generate_regime_ensemble_signal(df)
        data[t] = df
        signals[t] = sig.reindex(df.index)

    common_dates = None
    for t in tickers:
        idx = data[t].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    if len(common_dates) < 2:
        raise ValueError("Not enough overlapping dates among tickers to backtest.")

    cash = float(initial_capital)
    shares = {t: 0 for t in tickers}
    alloc = normalize_allocation(tickers, allocation)

    trade_log: List[Dict] = []
    history: List[Dict] = []

    for i in range(1, len(common_dates)):
        date = common_dates[i]
        prev_date = common_dates[i - 1]

        prices, s_now, s_prev = {}, {}, {}
        for t in tickers:
            close = data[t]["Close"]
            close = close.iloc[:, 0] if isinstance(close, pd.DataFrame) else close
            prices[t] = float(close.loc[date])
            s_now[t] = int(signals[t].loc[date])
            s_prev[t] = int(signals[t].loc[prev_date])


        for t in tickers:
            price = prices[t]
            if s_now[t] == 1 and s_prev[t] == 0 and shares[t] == 0:
                total_value = cash + sum(shares[x] * prices[x] for x in tickers)
                target_bucket = total_value * alloc[t]
                budget = min(cash, target_bucket) * position_size_pct

                qty = int(budget / price)
                if qty > 0 and qty * price <= cash:
                    cost = qty * price
                    cash -= cost
                    shares[t] += qty
                    trade_log.append({
                        "date": date, "ticker": t, "action": "BUY",
                        "shares": qty, "price": price,
                        "trade_value": cost, "cash_after": cash,
                    })

            if s_now[t] == 0 and s_prev[t] == 1 and shares[t] > 0:
                qty = shares[t]
                proceeds = qty * price
                cash += proceeds
                shares[t] = 0
                trade_log.append({
                    "date": date, "ticker": t, "action": "SELL",
                    "shares": qty, "price": price,
                    "trade_value": proceeds, "cash_after": cash,
                })

        holdings_value = {t: shares[t] * prices[t] for t in tickers}
        total_value = cash + sum(holdings_value.values())

        row = {"date": date, "cash": cash, "total_value": total_value}
        for t in tickers:
            row[f"{t}_shares"] = shares[t]
            row[f"{t}_value"] = holdings_value[t]
        history.append(row)

    portfolio_history_df = pd.DataFrame(history)
    trade_log_df = pd.DataFrame(trade_log)

    final_state = {
        "cash": cash,
        "shares": dict(shares),
        "allocation": alloc,
        "initial_capital": initial_capital,
    }
    return portfolio_history_df, trade_log_df, final_state

def compute_buy_and_hold(
    tickers: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float,
    allocation: Dict[str, float],
) -> Dict[str, float]:
    alloc = normalize_allocation(tickers, allocation)
    first_prices, last_prices = {}, {}

    for t in tickers:
        df = fetch_stock_data(t, start_date, end_date).sort_index()
        close = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
        first_prices[t] = float(close.iloc[0])
        last_prices[t] = float(close.iloc[-1])

    final_value = 0.0
    for t in tickers:
        invested = initial_capital * alloc[t]
        shares = invested / first_prices[t]
        final_value += shares * last_prices[t]

    total_return_pct = (final_value / initial_capital - 1) * 100
    return {"final_value": final_value, "total_return_pct": total_return_pct}

def compute_portfolio_metrics(
    portfolio_history: pd.DataFrame,
    initial_capital: float,
    trading_days_per_year: int = 252,
) -> Dict[str, float]:
    if portfolio_history.empty:
        return {}

    pv = portfolio_history["total_value"].astype(float)
    rets = pv.pct_change().dropna()

    final_value = float(pv.iloc[-1])
    total_return_pct = (final_value / initial_capital - 1) * 100

    n_days = len(pv)
    years = n_days / trading_days_per_year
    annualized_return_pct = ((final_value / initial_capital) ** (1 / max(years, 1e-6)) - 1) * 100

    sharpe = float((rets.mean() / (rets.std() + 1e-10)) * np.sqrt(trading_days_per_year)) if len(rets) else 0.0
    volatility_pct = float(rets.std() * np.sqrt(trading_days_per_year) * 100) if len(rets) else 0.0

    cum = pv / pv.iloc[0]
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_drawdown_pct = float(dd.min() * 100)

    win_rate_pct = float((rets > 0).mean() * 100) if len(rets) else 0.0

    return {
        "initial_capital": float(initial_capital),
        "final_value": final_value,
        "profit": float(final_value - initial_capital),
        "total_return_pct": total_return_pct,
        "annualized_return_pct": annualized_return_pct,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_drawdown_pct,
        "volatility_pct": volatility_pct,
        "win_rate_pct": win_rate_pct,
        "trading_days": n_days,
    }

def parse_allocation_kv(pairs: List[str]) -> Dict[str, float]:
    alloc = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        alloc[k.strip().upper()] = float(v)
    return alloc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", required=True, help="e.g., AAPL MSFT GOOGL")
    parser.add_argument("--start_date", type=str, default="2020-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--initial_capital", type=float, default=10000.0)
    parser.add_argument("--position_size_pct", type=float, default=0.95)
    parser.add_argument("--alloc", nargs="*", default=None,
                        help="optional e.g., AAPL=0.5 MSFT=0.3 GOOGL=0.2")
    parser.add_argument("--save", action="store_true", help="save CSV outputs")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    allocation = parse_allocation_kv(args.alloc) if args.alloc else None

    portfolio_history, trade_log, final_state = backtest_portfolio(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        allocation=allocation,
        position_size_pct=args.position_size_pct,
    )

    metrics = compute_portfolio_metrics(portfolio_history, args.initial_capital)

    bh = compute_buy_and_hold(tickers, args.start_date, args.end_date,
                              args.initial_capital, allocation)

    alloc_display = final_state["allocation"]

    print("\n")
    print("PORTFOLIO RESULTS (Regime-Based Ensemble)")
    print("\n")
    print(f"Tickers         : {', '.join(tickers)}")
    print(f"Allocation       : {', '.join(f'{t}={alloc_display[t]:.1%}' for t in tickers)}")
    print(f"Period           : {args.start_date} to {args.end_date}")
    print(f"Trading Days     : {metrics['trading_days']}")
    print("\n")
    print(f"Initial Capital  : ${metrics['initial_capital']:>12,.2f}")
    print(f"Final Value      : ${metrics['final_value']:>12,.2f}")
    print(f"Profit / Loss    : ${metrics['profit']:>12,.2f}")
    print("\n")
    print(f"Total Return     : {metrics['total_return_pct']:>8.2f}%")
    print(f"Annualized Return: {metrics['annualized_return_pct']:>8.2f}%")
    print(f"Sharpe Ratio     : {metrics['sharpe_ratio']:>8.3f}")

    print(f"Total Trades     : {len(trade_log):>8d}")
    print("\n")

    print("\nFINAL HOLDINGS:")
    for t in tickers:
        sh = final_state["shares"][t]
        print(f"  {t}: {sh} shares")
    print(f"  Cash remaining: ${final_state['cash']:,.2f}")


if __name__ == "__main__":
    main()