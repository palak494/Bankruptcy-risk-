import os
import pytest
import pandas as pd
from fastapi.testclient import TestClient
from project.main import app
import project.main as main
from project.database import DatabaseHandler

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_test_db():
    # Globally mock the database in main.py to use an in-memory database
    test_db = DatabaseHandler(":memory:")
    main.db = test_db
    yield

@pytest.fixture
def sample_payload():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(os.path.dirname(current_dir), "sample_company.csv")
    if not os.path.exists(sample_path):
        sample_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "sample_company.csv")
    df = pd.read_csv(sample_path)
    return df.to_dict(orient="records")

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_endpoint_success(sample_payload):
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    
    result = response.json()
    assert "risk_score" in result
    assert "risk_level" in result
    assert "top_features" in result
    assert "prepared_data" in result
    assert len(result["top_features"]) == 5

def test_predict_and_history_integration(sample_payload):
    # Verify history is initially empty
    history_response_empty = client.get("/history")
    assert history_response_empty.status_code == 200
    assert len(history_response_empty.json()) == 0

    # Run prediction
    predict_response = client.post("/predict", json=sample_payload)
    assert predict_response.status_code == 200
    risk_score = predict_response.json()["risk_score"]
    risk_level = predict_response.json()["risk_level"]

    # Verify run is logged in history
    history_response = client.get("/history")
    assert history_response.status_code == 200
    history_data = history_response.json()
    assert len(history_data) == 1
    assert history_data[0]["risk_score"] == risk_score
    assert history_data[0]["risk_level"] == risk_level

def test_predict_endpoint_empty_payload():
    response = client.post("/predict", json=[])
    assert response.status_code == 400
    assert "detail" in response.json()

def test_predict_endpoint_missing_columns(sample_payload):
    # Drop a column to trigger schema validation failure
    bad_payload = list(sample_payload)
    if bad_payload:
        del bad_payload[0][list(bad_payload[0].keys())[0]]
        
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 400
    assert "Missing columns" in response.json()["detail"]
