import pytest
from fastapi.testclient import TestClient
from api import app, TumorFeatures

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_predict_benign_mock_data():
    # Provide a dummy payload with shape matching TumorFeatures
    dummy_features = {
        "radius_mean": 10.0, "texture_mean": 10.0, "perimeter_mean": 60.0, "area_mean": 300.0, "smoothness_mean": 0.09, 
        "compactness_mean": 0.05, "concavity_mean": 0.01, "concave_points_mean": 0.01, "symmetry_mean": 0.15, "fractal_dimension_mean": 0.06,
        "radius_se": 0.2, "texture_se": 0.5, "perimeter_se": 1.5, "area_se": 15.0, "smoothness_se": 0.005, 
        "compactness_se": 0.01, "concavity_se": 0.01, "concave_points_se": 0.005, "symmetry_se": 0.01, "fractal_dimension_se": 0.002,
        "radius_worst": 11.0, "texture_worst": 12.0, "perimeter_worst": 65.0, "area_worst": 350.0, "smoothness_worst": 0.1, 
        "compactness_worst": 0.1, "concavity_worst": 0.05, "concave_points_worst": 0.05, "symmetry_worst": 0.2, "fractal_dimension_worst": 0.07
    }
    
    response = client.post("/predict", json=dummy_features)
    assert response.status_code == 200
    json_resp = response.json()
    assert "prediction" in json_resp
    assert "confidence" in json_resp
