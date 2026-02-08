
import yfinance as yf

def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    if data.empty:
        print(f"No data found for ticker '{ticker}'.")
        return data

    print(f"Data fetched: {len(data)} rows")
    return data
