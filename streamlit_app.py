import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Sales Forecast J+7 (CA)", layout="centered")

@st.cache_resource
def load_model():
    model_path = Path("models") / "linear_regression_pipeline.joblib"
    return joblib.load(model_path)

model = load_model()

st.title("Prévision des ventes à J+7 – Californie (M5)")
st.write("Entrez les variables explicatives puis lancez la prédiction.")

col1, col2 = st.columns(2)

with col1:
    wday = st.number_input("wday (1-7)", min_value=1, max_value=7, value=1)
    month = st.number_input("month (1-12)", min_value=1, max_value=12, value=6)
    year = st.number_input("year (2011-2016)", min_value=2011, max_value=2016, value=2016)
    is_weekend = st.selectbox("is_weekend", [0, 1], index=0)

with col2:
    is_event = st.selectbox("is_event", [0, 1], index=0)
    sales_lag_7 = st.number_input("sales_lag_7", min_value=0.0, value=1500.0, step=10.0)
    sales_rolling_mean_7 = st.number_input("sales_rolling_mean_7", min_value=0.0, value=1400.0, step=10.0)

if st.button("Predict"):
    X = pd.DataFrame([{
        "wday": int(wday),
        "month": int(month),
        "year": int(year),
        "is_weekend": int(is_weekend),
        "is_event": int(is_event),
        "sales_lag_7": float(sales_lag_7),
        "sales_rolling_mean_7": float(sales_rolling_mean_7),
    }])

    pred = model.predict(X)[0]
    st.success(f"Prévision des ventes à J+7 : {pred:.2f}")
