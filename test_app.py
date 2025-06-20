
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "House Price Prediction" in response.text

def test_predict_price():
    response = client.post("/predict/", json={
        "car_parks": 2,
        "size_num": 1000,
        "location": "ampang",
        "furnishing": "fully_furnished"
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()
