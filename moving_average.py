
import argparse
import pandas as pd
import numpy as np
from fetch_data import fetch_stock_data


def calculate_moving_averages(data, short_window=50, long_window=200):
    data['MA_short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_long'] = data['Close'].rolling(window=long_window).mean()
    
    return data


def generate_signals(data):
    data['Signal'] = 0
    data['Signal'] = np.where(data['MA_short'] > data['MA_long'], 1, 0)
    data['Position'] = data['Signal'].diff()
    return data

def backtest_strategy(data, initial_capital=10000):
    data['Daily_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Signal'].shift(1) * data['Daily_Return']
    data['Market_Cumulative'] = (1 + data['Daily_Return']).cumprod()
    data['Strategy_Cumulative'] = (1 + data['Strategy_Return']).cumprod()
    data['Portfolio_Value'] = initial_capital * data['Strategy_Cumulative']

    final_market_return = (data['Market_Cumulative'].iloc[-1] - 1) * 100
    final_strategy_return = (data['Strategy_Cumulative'].iloc[-1] - 1) * 100
    final_portfolio_value = data['Portfolio_Value'].iloc[-1]
    
    print(f"BACKTEST RESULTS")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"")
    print(f"Buy & Hold Return: {final_market_return:.2f}%")
    print(f"Strategy Return: {final_strategy_return:.2f}%")
    print(f"Difference: {final_strategy_return - final_market_return:.2f}%")
    
    return data


def main():
    """
    Main function to run the moving average trading algorithm
    """
    parser = argparse.ArgumentParser(description='Moving Average Trading Algorithm')
    parser.add_argument('--ticker', type=str, default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    TICKER = args.ticker        
    START_DATE = args.start_date 
    END_DATE = args.end_date    
    SHORT_WINDOW = 50           
    LONG_WINDOW = 200           
    INITIAL_CAPITAL = 10000      
    
    data = fetch_stock_data(TICKER, START_DATE, END_DATE)
    
    data = calculate_moving_averages(data, SHORT_WINDOW, LONG_WINDOW)
    
    data = generate_signals(data)
    
    data = backtest_strategy(data, INITIAL_CAPITAL)


if __name__ == "__main__":
    main()