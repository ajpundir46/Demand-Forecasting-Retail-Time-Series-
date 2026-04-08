# 📚 Retail Demand Forecasting: Learning Guide

Welcome to your learning repository! This project is a practical application of high-performance Time-Series Forecasting. Below are the key concepts used in this system.

---

## 1. 🕒 Time-Aware Data Splitting
In standard ML, we use random shuffling. In Retail Forecasting, shuffling is a **disaster** because it leaks future info into the past.
- **What we used**: We sorted data by Date and took the **last 20%** for testing.
- **Why**: This simulates a real-world scenario where you train on historical data to predict the future.

## 2. 🔩 Feature Engineering for Seasonality
Raw dates can't be fed into a model. We extracted:
- **Week of Year**: Captures the recurring peaks (like the Holiday Season).
- **Quarter**: Captures quarterly business cycles.
- **Is_Weekend**: Retail sales spike significantly on Saturdays and Sundays.

## 3. 🌳 Model Trade-offs
We benchmarked several algorithms:
- **Linear Regression**: Fast but failed here because retail sales are non-linear (spiky).
- **Random Forest**: Great at handling outliers (MarkDowns), but creates very large model files (~15MB).
- **XGBoost (Winner)**: Highly optimized gradient boosting. It handles missing values (MarkDowns) natively and provides the best precision (R² ~0.96) with a smaller file footprint (~0.4MB).

## 📊 How to read Metrics (R² vs RMSE)
- **R² Score**: How much of the variance is captured? (0.96 means the model explains 96% of the sales fluctuations).
- **RMSE**: The average dollar amount the prediction is "off" by. In this project, an RMSE of ~3000 on a $30k average is excellent!

---
*Keep exploring and adding your own notes here!* 🏙️
