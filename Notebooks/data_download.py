import yfinance as yf
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from config import (
    START_DATE,
    END_DATE,
    RAW_DATA_PATH,
    MAX_STOCKS,
)

#All liquid big stocks
def get_universe():
    
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META",
        "NVDA", "TSLA", "JPM", "BAC", "GS",
        "JNJ", "XOM", "UNH", "HD", "PG",
        "AVGO", "CVX", "PEP", "KO", "MRK",
        "COST", "WMT", "DIS", "NFLX", "ADBE",
        "INTC", "AMD", "QCOM", "IBM", "ORCL",
        "CSCO", "CRM", "ABT", "ABBV", "PFE",
        "TMO", "ACN", "MCD", "NKE", "SBUX",
    ]

    return tickers[:MAX_STOCKS]

#downloading data for all stocks using yfinance, if its empty move on. Append all to one list and then concat to get one big df
def download_price_data(ticker):
    
    all_data = []
    
    for ticker in tqdm(ticker, desc='Downloading'):
        
        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False,
            progress=False,
        )
        
        if df.empty:
            continue
        
        df = df.reset_index()
        
        df["ticker"] = ticker
        
        all_data.append(df)
        
    data = pd.concat(all_data, ignore_index=True)
    
    return data

#renaming for easier usage later, datatime since its time series
def standardize(df):

    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    return df

#using parquet files given the size
def save_raw_data(df):
    
    Path(RAW_DATA_PATH).mkdir(parents=True, exist_ok=True)

    file_path = Path(RAW_DATA_PATH) / "us_equities_ohlcv.parquet"
    df.to_parquet(file_path, index=False)

    print(f"Saved raw data to {file_path}")


def main():
    
    tickers = get_universe()
    print(f"Downloading {len(tickers)} stocks")

    df = download_price_data(tickers)
    df = standardize(df)

    save_raw_data(df)

    print("Phase 1 data download complete")


if __name__ == "__main__":
    main()
