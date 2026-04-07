import os
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

# Setup logging
def setup_logging(log_level=logging.INFO):
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=log_level
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def get_project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).parent.parent

def ensure_dir(dir_path: str):
    """Ensures a directory exists."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def get_timestamp() -> str:
    """Returns a formatted timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def load_csv_data(file_path: str) -> pd.DataFrame:
    """Loads a CSV file safely and logs the shape."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        raise e

def save_dataframe(df: pd.DataFrame, file_path: str):
    """Saves a dataframe to a CSV file."""
    ensure_dir(os.path.dirname(file_path))
    df.to_csv(file_path, index=False)
    logger.info(f"Saved dataframe to {file_path}")

def save_json(data: dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    ensure_dir(os.path.dirname(file_path))
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved JSON data to {file_path}")

def auto_detect_dataset(data_dir: str) -> dict:
    """Automatically detects train, stores, and features CSV files."""
    if not os.path.exists(data_dir):
        logger.error(f"Data directory not found: {data_dir}")
        return {}
        
    files = os.listdir(data_dir)
    dataset_map = {}
    for f in files:
        if 'train' in f.lower():
            dataset_map['train'] = os.path.join(data_dir, f)
        elif 'store' in f.lower():
            dataset_map['stores'] = os.path.join(data_dir, f)
        elif 'feature' in f.lower():
            dataset_map['features'] = os.path.join(data_dir, f)
    
    logger.info(f"Detected datasets: {dataset_map}")
    return dataset_map
