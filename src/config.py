from datetime import datetime

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

MAX_STOCKS = 100  

MIN_PRICE = 5.0
MIN_AVG_DOLLAR_VOLUME = 5_000_000  

RAW_DATA_PATH = "/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/raw"
PROCESSED_DATA_PATH = "/Users/sudhanvabharadwaj/Documents/Full_Quant_Pipeline/Data/processed"

MARKET_TICKER = "SPY"

TRAIN_START = "2010-01-01"
TRAIN_END = "2021-12-31"    # 12 years training

VAL_START = "2022-01-01"
VAL_END = "2023-06-30"      # 1.5 years validation

TEST_START = "2023-07-01"
TEST_END = "2024-12-31"     # 1.5 years test