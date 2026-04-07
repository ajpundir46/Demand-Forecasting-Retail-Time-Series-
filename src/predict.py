import os
import joblib
import logging
import pandas as pd
import json
from src.utils import setup_logging, load_csv_data, ensure_dir

logger = setup_logging()

def load_best_model(model_path='models/best_model.pkl'):
    """Loads the production-ready best model."""
    if not os.path.exists(model_path):
        logger.error(f"Best model not found at {model_path}. Run training first.")
        raise FileNotFoundError(f"Model file missing: {model_path}")
    
    return joblib.load(model_path)

def predict_batch(csv_path: str, output_path='reports/predictions/batch_predictions.csv'):
    """Performs batch prediction on a CSV file."""
    logger.info(f"Performing batch prediction on {csv_path}")
    model = load_best_model()
    df = load_csv_data(csv_path)
    
    # Preprocessing (ensure features match training)
    # In a full production system, we'd have a specific prediction preprocessing pipeline
    # For now, we assume the input CSV has the required features
    predictions = model.predict(df)
    df['Predicted_Sales'] = predictions
    
    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)
    logger.info(f"Batch predictions saved to {output_path}")
    return df

def predict_single(input_data: dict):
    """Performs prediction on a single row (dictionary)."""
    model = load_best_model()
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction

if __name__ == "__main__":
    # Example Single Prediction (requires feature engineering columns)
    # This is a placeholder; in practice, you'd provide the engineered features
    example_input = {
        'Store': 1,
        'Dept': 1,
        'IsHoliday': False,
        'Type': 'C',
        'Size': 150000,
        'Temperature': 65.0,
        'Fuel_Price': 3.5,
        'MarkDown1': 1000,
        'MarkDown2': 0,
        'CPI': 210,
        'Unemployment': 7.5,
        'Year': 2024,
        'Month': 4,
        'Day': 6,
        'DayOfWeek': 5,
        'Is_Weekend': 1,
        'Quarter': 2,
        'WeekOfYear': 14
    }
    
    try:
        res = predict_single(example_input)
        print(f"Predicted Weekly Sales: ${res:.2f}")
    except Exception as e:
        print(f"Error in single prediction: {e}")
