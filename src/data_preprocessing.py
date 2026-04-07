import pandas as pd
import logging
from src.utils import load_csv_data, auto_detect_dataset

logger = logging.getLogger(__name__)

def merge_datasets(data_dir: str) -> pd.DataFrame:
    """Detects, loads, and merges train, stores, and features datasets."""
    dataset_map = auto_detect_dataset(data_dir)
    
    if not all(k in dataset_map for k in ['train', 'stores', 'features']):
        logger.error(f"Required datasets missing: {dataset_map}")
        raise ValueError("Must provide train, stores, and features CSV files.")

    train = load_csv_data(dataset_map['train'])
    stores = load_csv_data(dataset_map['stores'])
    features = load_csv_data(dataset_map['features'])

    # Standard Walmart Schema Merge
    logger.info("Merging datasets...")
    df = train.merge(stores, on='Store', how='left')
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic data cleaning."""
    logger.info("Cleaning data...")
    
    # 1. Date conversion
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Order Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Order Date'])
    else:
        logger.warning("No standard Date column found for parsing.")

    # 2. Handle missing MarkDowns (Fill with 0 as per competition standard)
    markdown_cols = [c for c in df.columns if 'MarkDown' in c]
    df[markdown_cols] = df[markdown_cols].fillna(0)

    # 3. Handle other missing values
    # Fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())
                
    logger.info(f"Cleaning complete. Shape: {df.shape}")
    return df

if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging()
    data_path = 'data'
    try:
        merged_df = merge_datasets(data_path)
        cleaned_df = clean_data(merged_df)
        print(cleaned_df.head())
    except Exception as e:
        print(f"Failed to preprocess: {e}")
