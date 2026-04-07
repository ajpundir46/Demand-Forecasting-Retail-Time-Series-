import os
import joblib
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

from src.utils import setup_logging, ensure_dir, save_json

logger = setup_logging()

def evaluate_models(model_dir='models', report_dir='reports'):
    """Evaluates all saved models and selects the best one."""
    logger.info("Evaluating models...")
    
    # Load test data
    test_data_path = os.path.join(model_dir, "test_data.pkl")
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found at {test_data_path}")
        return

    X_test, y_test = joblib.load(test_data_path)
    
    # Find all trained models
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl') and f not in ['test_data.pkl', 'best_model.pkl']]
    
    metrics_list = []
    best_rmse = float('inf')
    best_model_name = None
    best_model = None
    
    # Prepare reports directories
    fig_dir = os.path.join(report_dir, 'figures')
    ensure_dir(fig_dir)

    for model_file in model_files:
        name = model_file.split('.')[0]
        logger.info(f"Evaluating {name}...")
        
        model = joblib.load(os.path.join(model_dir, model_file))
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics_list.append({
            'Model': name,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        })
        
        # Select best model based on RMSE
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = model

        # Plot Actual vs Predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        plt.xlabel('Actual Sales')
        plt.ylabel('Predicted Sales')
        plt.title(f'Actual vs Predicted - {name}')
        plt.savefig(os.path.join(fig_dir, f'actual_vs_pred_{name}.png'))
        plt.close()

    # Create Comparison DataFrame
    df_comparison = pd.DataFrame(metrics_list).sort_values('RMSE')
    comparison_path = os.path.join(report_dir, 'model_comparison.csv')
    df_comparison.to_csv(comparison_path, index=False)
    logger.info(f"Model comparison saved to {comparison_path}")

    # Save Best Model explicitly
    if best_model:
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        joblib.dump(best_model, best_model_path)
        logger.info(f"Best model selected: {best_model_name} with RMSE: {best_rmse:.2f}")

        # Calculate feature ranges for UI limits
        feature_ranges = {}
        for col in X_test.columns:
            if X_test[col].dtype != 'object' and X_test[col].dtype != 'bool':
                feature_ranges[col] = {
                    'min': float(X_test[col].min()),
                    'max': float(X_test[col].max()),
                    'mean': float(X_test[col].mean())
                }

        # Save Best Model Metadata
        metadata = {
            'best_model': best_model_name,
            'rmse': best_rmse,
            'training_mean': float(y_test.mean()),
            'feature_ranges': feature_ranges,
            'timestamp': datetime.now().isoformat(),
            'metrics': df_comparison.to_dict(orient='records'),
            'features_used': list(X_test.columns)
        }
        save_json(metadata, os.path.join(report_dir, 'model_metrics.json'))
        
        # Write summary text
        with open(os.path.join(report_dir, 'model_summary.txt'), 'w') as f:
            f.write(f"--- Production Model Summary ---\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"RMSE: {best_rmse:.2f}\n")
            f.write(f"R2: {df_comparison[df_comparison['Model']==best_model_name]['R2'].values[0]:.4f}\n")
            f.write(f"Features: {list(X_test.columns)}\n")

    return df_comparison

if __name__ == "__main__":
    evaluate_models()
