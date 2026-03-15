import pandas as pd
from pathlib import Path
from config import PROCESSED_DATA_PATH


# Temporal splits
TRAIN_START = "2010-01-01"
TRAIN_END = "2021-12-31"    # 12 years training

VAL_START = "2022-01-01"
VAL_END = "2023-06-30"      # 1.5 years validation

TEST_START = "2023-07-01"
TEST_END = "2024-12-31"     # 1.5 years test


def create_temporal_splits(df):


    df['date'] = pd.to_datetime(df['date'])
    
    # Creating splits
    train = df[
        (df['date'] >= TRAIN_START) & 
        (df['date'] <= TRAIN_END)
    ].copy()
    
    val = df[
        (df['date'] >= VAL_START) & 
        (df['date'] <= VAL_END)
    ].copy()
    
    test = df[
        (df['date'] >= TEST_START) & 
        (df['date'] <= TEST_END)
    ].copy()
    
    # Verifying the created splits
    print("TEMPORAL SPLITS CREATED")
    
    print(f"\nTrain:")
    print(f"  Dates: {train['date'].min()} to {train['date'].max()}")
    print(f"  Shape: {train.shape}")
    print(f"  Unique dates: {train['date'].nunique()}")
    print(f"  Unique tickers: {train['ticker'].nunique()}")
    
    print(f"\nValidation:")
    print(f"  Dates: {val['date'].min()} to {val['date'].max()}")
    print(f"  Shape: {val.shape}")
    print(f"  Unique dates: {val['date'].nunique()}")
    print(f"  Unique tickers: {val['ticker'].nunique()}")
    
    print(f"\nTest:")
    print(f"  Dates: {test['date'].min()} to {test['date'].max()}")
    print(f"  Shape: {test.shape}")
    print(f"  Unique dates: {test['date'].nunique()}")
    print(f"  Unique tickers: {test['ticker'].nunique()}")
    
    # Check for gaps
    print("\n" + "="*60)
    print("SPLIT VERIFICATION")
    print("="*60)
    
    # No overlap
    assert train['date'].max() < val['date'].min(), "Train/Val overlap!"
    assert val['date'].max() < test['date'].min(), "Val/Test overlap!"
    print("No temporal overlap between splits")
    
    # No gaps
    train_end = pd.to_datetime(TRAIN_END)
    val_start = pd.to_datetime(VAL_START)
    gap_days = (val_start - train_end).days
    print(f"Gap between train and val: {gap_days} days")
    
    # Same tickers in all splits 
    train_tickers = set(train['ticker'].unique())
    val_tickers = set(val['ticker'].unique())
    test_tickers = set(test['ticker'].unique())
    
    common_tickers = train_tickers & val_tickers & test_tickers
    print(f"Tickers in all splits: {len(common_tickers)}")
    
    if len(common_tickers) < len(train_tickers):
        missing = train_tickers - common_tickers
        print(f"Warning: {len(missing)} tickers missing from val/test: {missing}")
    
    return train, val, test


def save_splits(train, val, test):

    split_path = Path(PROCESSED_DATA_PATH) / "splits"
    split_path.mkdir(parents=True, exist_ok=True)
    
    # Save each split
    train.to_parquet(split_path / "train.parquet", index=False)
    val.to_parquet(split_path / "val.parquet", index=False)
    test.to_parquet(split_path / "test.parquet", index=False)
    
    print(f"\nSplits saved to {split_path}")
    print(f"   - train.parquet")
    print(f"   - val.parquet")
    print(f"   - test.parquet")


def load_splits():

    
    split_path = Path(PROCESSED_DATA_PATH) / "splits"
    
    train = pd.read_parquet(split_path / "train.parquet")
    val = pd.read_parquet(split_path / "val.parquet")
    test = pd.read_parquet(split_path / "test.parquet")
    
    return train, val, test


def main():


    processed_path = Path(PROCESSED_DATA_PATH) / "features_df.parquet"
    
    if not processed_path.exists():
        print("❌ Error: Run feature engineering first")
        print(f"   Expected file: {processed_path}")
        return
    
    df = pd.read_parquet(processed_path)

    train, val, test = create_temporal_splits(df)

    save_splits(train, val, test)

if __name__ == "__main__":
    main()