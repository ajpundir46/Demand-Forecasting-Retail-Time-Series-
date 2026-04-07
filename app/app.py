import streamlit as st
import pandas as pd
import joblib
import os
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Retail Sales Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Resources ---
@st.cache_resource
def load_metadata():
    metrics_path = 'reports/model_metrics.json'
    return json.load(open(metrics_path, 'r')) if os.path.exists(metrics_path) else None

@st.cache_resource
def load_specific_model(model_name):
    path = f'models/{model_name}.pkl'
    return joblib.load(path) if os.path.exists(path) else None

metadata = load_metadata()

# --- Sidebar Configuration ---
st.sidebar.title("🛠️ Model Selection")
st.sidebar.markdown("---")

if metadata:
    model_choices = [m['Model'] for m in metadata['metrics']]
    active_model_name = st.sidebar.selectbox(
        "Choose Forecast Engine:", 
        model_choices, 
        index=model_choices.index(metadata['best_model'])
    )
    # Load selected model
    model = load_specific_model(active_model_name)
    
    # Metadata display for selected model
    model_m = next(m for m in metadata['metrics'] if m['Model'] == active_model_name)
    st.sidebar.write(f"**Accuracy**: {model_m['RMSE']:.2f} (RMSE)")
    st.sidebar.write(f"**Precision**: {model_m['R2']:.2%}")
else:
    st.sidebar.error("⚠️ Metadata missing. Using default model.")
    model = joblib.load('models/best_model.pkl') if os.path.exists('models/best_model.pkl') else None

st.sidebar.markdown("---")
st.sidebar.write(f"**Training Period**: Feb 2010 - Oct 2012")
st.sidebar.info("This system allows you to toggle between different trained algorithms to compare performance.")

st.title("🛍️ Retail Sales Demand Forecasting")
st.markdown("---")

# --- Main Interface ---
col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.header("📋 Input Features")
    
    # Helper to get range safely
    def get_range(key, default_min, default_max):
        if metadata and 'feature_ranges' in metadata and key in metadata['feature_ranges']:
            return (
                metadata['feature_ranges'][key]['min'], 
                metadata['feature_ranges'][key]['max'],
                metadata['feature_ranges'][key]['mean']
            )
        return default_min, default_max, (default_min + default_max) / 2

    sub1, sub2 = st.columns(2)
    with sub1:
        st.subheader("Store Info")
        s_min, s_max, s_mean = get_range('Store', 1, 45)
        store = st.number_input("Store ID", int(s_min), int(s_max), int(s_mean))
        
        d_min, d_max, d_mean = get_range('Dept', 1, 99)
        dept = st.number_input("Dept ID", int(d_min), int(d_max), int(d_mean))
        
        store_type = st.selectbox("Store Type", ["A", "B", "C"])
        
        sz_min, sz_max, sz_mean = get_range('Size', 30000, 250000)
        size = st.number_input("Size (sq ft)", int(sz_min), int(sz_max), int(sz_mean))
    
    with sub2:
        st.subheader("Regional Context")
        t_min, t_max, t_mean = get_range('Temperature', 0.0, 100.0)
        temp = st.slider("Temp (°F)", t_min, t_max, t_mean)
        
        f_min, f_max, f_mean = get_range('Fuel_Price', 2.0, 5.0)
        fuel = st.slider("Fuel Price", f_min, f_max, f_mean)
        
        c_min, c_max, c_mean = get_range('CPI', 150.0, 250.0)
        cpi = st.number_input("CPI", c_min, c_max, c_mean)
        
        u_min, u_max, u_mean = get_range('Unemployment', 0.0, 20.0)
        unemp = st.slider("Unemployment %", u_min, u_max, u_mean)

with col2:
    st.header("📅 Deployment")
    date_val = st.date_input("Select Date", datetime.now())
    is_holiday = st.checkbox("Is Holiday Week?", value=False)
    
    st.subheader("🏷️ Promotional Markdowns")
    m1 = st.number_input("MarkDown 1", value=1000.0)
    m2 = st.number_input("MarkDown 2", value=0.0)

# Feature Construction
input_df = pd.DataFrame([{
    'Store': store, 'Dept': dept, 'IsHoliday': bool(is_holiday), 'Type': store_type, 'Size': size,
    'Temperature': temp, 'Fuel_Price': fuel, 'MarkDown1': m1, 'MarkDown2': m2, 'CPI': cpi, 'Unemployment': unemp,
    'Year': date_val.year, 'Month': date_val.month, 'Day': date_val.day, 'DayOfWeek': date_val.weekday(),
    'Is_Weekend': 1 if date_val.weekday() >= 5 else 0, 'Quarter': (date_val.month-1)//3+1, 'WeekOfYear': int(date_val.isocalendar()[1])
}])

st.markdown("---")

# --- State Management ---
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# --- Forecast Execution ---
if st.button("🚀 RUN FORECAST", use_container_width=True):
    st.session_state.prediction_result = model.predict(input_df)[0]
    st.toast("Forecasting successful!", icon='📊')

if st.session_state.prediction_result is not None:
    prediction = st.session_state.prediction_result
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Weekly Sales", f"${prediction:,.2f}")
    
    avg_sales = metadata['training_mean'] if metadata and 'training_mean' in metadata else 31353.50
    m2.metric("Dataset Average", f"${avg_sales:,.2f}", delta=f"{(prediction-avg_sales)/avg_sales:+.1f}% vs Avg")
    
    active_r2 = model_m['R2'] if 'model_m' in locals() else 0.967
    m3.metric("Model Precision (R²)", f"{active_r2:.2%}")

# --- Diagnostics ---
if metadata:
    st.markdown("---")
    with st.expander("📊 Detailed Model Performance"):
        tabs = st.tabs(["Metrics Comparison", "Validation Plots"])
        with tabs[0]:
            st.table(pd.DataFrame(metadata['metrics']))
        with tabs[1]:
            models = [m['Model'] for m in metadata['metrics']]
            sel = st.selectbox("Inspect Algorithm:", models, index=models.index(metadata['best_model']))
            path = f"reports/figures/actual_vs_pred_{sel}.png"
            if os.path.exists(path):
                _, img_c, _ = st.columns([1, 2, 1])
                with img_c:
                    st.image(path, caption=f"Validation Flow - {sel}", use_container_width=True)

# --- Historical Insights Dashboard ---
st.markdown("---")
with st.expander("🏗️ Historical Data Insights (2010-2012)"):
    st.write("Explore the foundational patterns the model was trained on.")
    
    # Load aggregated summaries
    trend_path = 'reports/sales_trend_benchmark.csv'
    type_path = 'reports/sales_by_type_benchmark.csv'
    
    if os.path.exists(trend_path) and os.path.exists(type_path):
        dash_col1, dash_col2 = st.columns(2)
        
        with dash_col1:
            st.subheader("📅 Weekly Sales Trend")
            df_trend = pd.read_csv(trend_path)
            df_trend['Date'] = pd.to_datetime(df_trend['Date'])
            st.line_chart(df_trend.set_index('Date'))
            st.caption("Average weekly sales across all stores over time.")
            
        with dash_col2:
            st.subheader("🏬 Performance by Store Type")
            df_type = pd.read_csv(type_path).set_index('Type')
            st.bar_chart(df_type)
            st.caption("Total accumulated sales volume per store classification.")
    else:
        st.info("💡 Insights dashboard is ready. Populate using 'reports/' data CSVs.")
