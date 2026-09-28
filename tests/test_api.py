"""
Integration tests for FastAPI endpoints (app.py).
"""

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "rag-hallucination-detector"


def test_verify_empty_answer():
    payload = {
        "question": "Sample question?",
        "context": "Sample context.",
        "answer": "   ",
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 400


def test_verify_grounded_answer():
    payload = {
        "question": "Where is the Eiffel Tower?",
        "context": "The Eiffel Tower is located in Paris, France. It was built in 1889.",
        "answer": "The Eiffel Tower is located in Paris, France.",
        "top_k_evidence": 2,
    }
    response = client.post("/verify", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "faithfulness_score" in data
    assert "claims" in data
    assert len(data["claims"]) >= 1
    assert data["claims"][0]["verdict"] == "supported"
    assert data["is_faithful"] is True
