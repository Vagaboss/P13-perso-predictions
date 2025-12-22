from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Charger le modèle
model = joblib.load("models/linear_regression_pipeline.joblib")

# Initialiser l'API
app = FastAPI(title="Sales Forecast API", version="1.0")

# Schéma d'entrée (features)
class SalesFeatures(BaseModel):
    wday: int
    month: int
    year: int
    is_weekend: int
    is_event: int
    sales_lag_7: float
    sales_rolling_mean_7: float

# Endpoint de santé
@app.get("/health")
def health():
    return {"status": "ok"}

# Endpoint de prédiction
@app.post("/predict")
def predict(features: SalesFeatures):
    data = pd.DataFrame([features.dict()])
    prediction = model.predict(data)[0]
    return {
        "prediction_sales_t_plus_7": round(float(prediction), 2)
    }
