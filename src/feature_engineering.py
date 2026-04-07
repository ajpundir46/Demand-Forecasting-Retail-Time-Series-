import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extracts temporal features from the 'Date' column."""
    logger.info("Extracting features...")
    
    # 1. Temporal features
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)

    # 2. Difference features
    # df['Delivery_Days'] = (df['Ship Date'] - df['Order Date']).dt.days if 'Ship Date' in df.columns else np.nan

    logger.info("Feature extraction complete.")
    return df

def time_aware_train_test_split(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Sorts data by date and performs a split for time-series context."""
    logger.info(f"Performing time-aware split with test_size={test_size}")
    
    # Sort by date to avoid data leakage
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Define split point
    split_idx = int(len(df) * (1 - test_size))
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Separate features and target
    X_train = train_df.drop([target, 'Date'], axis=1)
    y_train = train_df[target]
    
    X_test = test_df.drop([target, 'Date'], axis=1)
    y_test = test_df[target]
    
    logger.info(f"Split complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def get_cat_num_columns(X: pd.DataFrame) -> tuple:
    """Automatically identifies categorical and numerical columns."""
    cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    num_cols = X.select_dtypes(include=['int', 'float']).columns.tolist()
    
    # Remove 'Date' if it somehow snuck in
    if 'Date' in cat_cols: cat_cols.remove('Date')
    
    return cat_cols, num_cols
