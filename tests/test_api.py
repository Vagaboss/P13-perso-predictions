from fastapi.testclient import TestClient
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api.main import app

# Initialisation du client de test FastAPI
client = TestClient(app)


def test_health_endpoint():
    """
    Vérifie que l'API est bien démarrée
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_input():
    """
    Vérifie qu'une requête valide retourne une prédiction
    """
    payload = {
        "wday": 6,
        "month": 6,
        "year": 2016,
        "is_weekend": 1,
        "is_event": 0,
        "sales_lag_7": 1500,
        "sales_rolling_mean_7": 1400
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "prediction_sales_t_plus_7" in data
    assert isinstance(data["prediction_sales_t_plus_7"], float)


def test_predict_invalid_type():
    """
    Vérifie qu'un type invalide est rejeté (validation Pydantic)
    """
    payload = {
        "wday": "lundi",  # type invalide
        "month": 6,
        "year": 2016,
        "is_weekend": 1,
        "is_event": 0,
        "sales_lag_7": 1500,
        "sales_rolling_mean_7": 1400
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_missing_field():
    """
    Vérifie qu'un champ manquant est rejeté
    """
    payload = {
        "wday": 6,
        "month": 6,
        "year": 2016,
        "is_weekend": 1,
        # "is_event" manquant
        "sales_lag_7": 1500,
        "sales_rolling_mean_7": 1400
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_out_of_range_values():
    """
    Test logique : valeurs aberrantes mais typées correctement
    (l'API les accepte si aucune contrainte métier n'est définie)
    """
    payload = {
        "wday": 7,
        "month": 12,
        "year": 2016,
        "is_weekend": 1,
        "is_event": 1,
        "sales_lag_7": 999999,          # valeur très élevée
        "sales_rolling_mean_7": 999999
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "prediction_sales_t_plus_7" in response.json()

