import os
import joblib
import logging
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from src.data_preprocessing import merge_datasets, clean_data
from src.feature_engineering import extract_features, time_aware_train_test_split, get_cat_num_columns
from src.utils import setup_logging, ensure_dir

logger = setup_logging()

def build_preprocessing_pipeline(cat_cols, num_cols):
    """Builds a scikit-learn preprocessing pipeline."""
    logger.info(f"Building preprocessing pipeline for {len(cat_cols)} categorical and {len(num_cols)} numerical columns.")
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    
    return preprocessor

def get_models():
    """Returns a dictionary of models to benchmark."""
    models = {
        'LinearRegression': LinearRegression(),
        'PolynomialRegression': Pipeline([
            ('poly', PolynomialFeatures(degree=2)),
            ('model', LinearRegression())
        ]),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    return models

def train_and_save_models(data_path='data', model_dir='models', target='Weekly_Sales'):
    """Main training pipeline."""
    # 1. Preprocess and Split
    df_raw = merge_datasets(data_path)
    df_clean = clean_data(df_raw)
    df_features = extract_features(df_clean)
    
    X_train, X_test, y_train, y_test = time_aware_train_test_split(df_features, target)
    cat_cols, num_cols = get_cat_num_columns(X_train)
    
    preprocessor = build_preprocessing_pipeline(cat_cols, num_cols)
    
    models = get_models()
    trained_models = {}
    
    # 2. Benchmarking
    ensure_dir(model_dir)

    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            full_pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            full_pipeline.fit(X_train, y_train)
            
            # Save model
            model_path = os.path.join(model_dir, f"{name}.pkl")
            joblib.dump(full_pipeline, model_path)
            trained_models[name] = full_pipeline
            logger.info(f"Successfully saved {name} to {model_path}")
        except Exception as e:
            logger.error(f"Error training {name}: {e}")

    # Save X_test and y_test for evaluation
    joblib.dump((X_test, y_test), os.path.join(model_dir, "test_data.pkl"))
    logger.info("Training pipeline complete.")
    return trained_models

if __name__ == "__main__":
    train_and_save_models()
