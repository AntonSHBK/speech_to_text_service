from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root():
    """Проверяет, что сервер работает."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Speech-to-Text API is running!"}
