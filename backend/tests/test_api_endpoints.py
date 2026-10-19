import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.services.rag_service import OUT_OF_SCOPE_FALLBACK

client = TestClient(app)

def test_health_endpoint():
    """Verify health endpoint returns openrouter provider info."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["provider"] == "openrouter"
    assert "model" in data

def test_root_endpoint():
    """Verify root documentation and info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "UniAssist AI"
    assert data["provider"] == "OpenRouter"

def test_api_chat_out_of_scope():
    """Verify chat endpoint properly rejects out-of-scope inquiries."""
    response = client.post("/api/chat", json={
        "question": "Can you give me a recipe for chocolate chip cookies?",
        "session_id": "api_test_oos"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == OUT_OF_SCOPE_FALLBACK
    assert data["sources"] == []

def test_api_chat_clear_session():
    """Verify session reset endpoint works."""
    response = client.delete("/api/chat/test_session_123")
    assert response.status_code == 200
    assert response.json()["status"] == "cleared"

def test_api_chat_normal_question_with_mock_rag():
    """Verify chat endpoint returns answer and sources."""
    mock_result = {
        "answer": "The tuition fee payment deadline is August 15.",
        "sources": [{"title": "Tuition Guide", "source": "tuition.md", "section": "Deadlines", "similarity": 0.88}],
        "session_id": "test_sess_api",
        "error": False
    }
    with patch("app.api.routes.rag_service.process_query", return_value=mock_result):
        response = client.post("/api/chat", json={
            "question": "When is tuition due?",
            "session_id": "test_sess_api"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == mock_result["answer"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Tuition Guide"
