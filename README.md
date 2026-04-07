# 🛍️ Retail Sales Forecasting & Benchmarking

An end-to-end, production-style machine learning project for retail demand forecasting. This project benchmarks multiple regression models, selects the optimal one for deployment, and provides a Streamlit-based interface for real-time sales prediction.

## 📖 Project Overview
The objective is to predict weekly sales for multiple stores and departments based on historical data, regional features (temperature, fuel price, unemployment), and promotional activities (MarkDowns).

### Key Features
- **4-Model Benchmarking**: Compares Linear Regression, Decision Tree, Random Forest, and XGBoost.
- **Automated Selection**: Automatically selects and saves the model with the lowest **RMSE**.
- **Time-Aware Splitting**: Uses chronological data splitting to prevent data leakage.
- **Production Pipeline**: Modular code structure (`src/`) for preprocessing, feature engineering, and evaluation.
- **Interactive UI**: A Streamlit application for human-in-the-loop inference.

## 🛠️ Project Structure
```text
├── app/
│   └── app.py              # Streamlit Application
├── data/
│   ├── train.csv           # Training Dataset
│   ├── stores.csv          # Store Metadata
│   └── features.csv        # Economic & Environmental Features
├── models/
│   ├── best_model.pkl      # Optimal model for production
│   └── *.pkl               # Individual benchmarked models
├── reports/
│   ├── figures/            # Actual vs Predicted plots
│   ├── model_comparison.csv # Performance metrics table
│   └── model_metrics.json  # Comprehensive metadata
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict.py
│   └── utils.py
└── README.md
```

## 🚀 Getting Started

### 1. Installation
Install the required dependencies using the project virtual environment:
```bash
pip install -r requirements.txt
```

### 2. Training the Models
Run the training pipeline to merge data, extract features, and train the benchmark models:
```bash
python -m src.train_models
```

### 3. Evaluation & Best Model Selection
Run the evaluation script to generate comparison reports and select the production model:
```bash
python -m src.evaluate_models
```

### 4. Running the Streamlit App
Launch the interactive dashboard:
```bash
streamlit run app/app.py
```

## 📊 Evaluation Metrics
Models are evaluated using:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error) - *Primary selection metric*
- **R² Score** (Coefficient of Determination)

## 🔮 Inference
You can perform batch predictions using the CLI:
```bash
python -m src.predict --path path/to/data.csv
```

---
*Note: This project is designed to be robust and modular, making it easy to swap datasets or add new regression models to the benchmarking suite.*