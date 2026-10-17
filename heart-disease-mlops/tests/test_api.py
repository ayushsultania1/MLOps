from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200


def test_predict():

    payload = {
        "age":55,
        "sex":1,
        "cp":2,
        "trestbps":140,
        "chol":250,
        "fbs":0,
        "restecg":1,
        "thalach":150,
        "exang":0,
        "oldpeak":1.5,
        "slope":2,
        "ca":0,
        "thal":2
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probability" in data