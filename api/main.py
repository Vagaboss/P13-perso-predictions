from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

import json
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------
# Paths & model loading (robuste)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]  # racine du projet (dossier parent de /api)
MODEL_PATH = BASE_DIR / "models" / "linear_regression_pipeline.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at: {MODEL_PATH}\n"
        "Make sure 'models/linear_regression_pipeline.joblib' exists at the project root."
    )

model = joblib.load(MODEL_PATH)

# ---------------------------------------------------------------------
# Logging setup (monitoring léger)
# ---------------------------------------------------------------------
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
PRED_LOG = LOG_DIR / "predictions.jsonl"


def log_prediction(payload: dict, prediction: float, latency_ms: float) -> None:
    record = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": payload,
        "prediction_sales_t_plus_7": float(prediction),
        "latency_ms": float(latency_ms),
    }
    with PRED_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------
app = FastAPI(title="Sales Forecast API", version="1.1")


class SalesFeatures(BaseModel):
    # Variables calendaires
    wday: int = Field(ge=1, le=7, description="Day-of-week index in M5 calendar (1-7).")
    month: int = Field(ge=1, le=12, description="Month number (1-12).")
    year: int = Field(ge=2011, le=2016, description="Model trained on years 2011-2016 only.")
    is_weekend: int = Field(ge=0, le=1, description="1 if weekend, else 0.")
    is_event: int = Field(ge=0, le=1, description="1 if event day, else 0.")

    # Historique des ventes
    sales_lag_7: float = Field(ge=0, description="Sales value at t-7.")
    sales_rolling_mean_7: float = Field(ge=0, description="Rolling mean of sales over last 7 days.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(features: SalesFeatures):
    start = time.perf_counter()

    payload = features.model_dump()  # Pydantic v2 (use .dict() if v1)
    data = pd.DataFrame([payload])

    pred = float(model.predict(data)[0])

    latency_ms = (time.perf_counter() - start) * 1000.0
    log_prediction(payload, pred, latency_ms)

    return {"prediction_sales_t_plus_7": round(pred, 2)}


@app.get("/metrics")
def metrics():
    """
    Retourne des métriques simples de monitoring :
    - count : nombre total de prédictions logguées
    - avg_latency_ms : latence moyenne des prédictions (ms)
    """
    if not PRED_LOG.exists():
        return {"count": 0, "avg_latency_ms": None}

    latencies = []
    count = 0

    with PRED_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            lat = rec.get("latency_ms")
            if lat is not None:
                latencies.append(float(lat))
            count += 1

    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    return {"count": count, "avg_latency_ms": avg_latency}
