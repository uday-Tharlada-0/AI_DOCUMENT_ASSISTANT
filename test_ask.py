from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch, Mock
from main import documents
client = TestClient(app)


def test_ask_returns_404_for_missing_file():
    response = client.post(
        "/ask",
        json={
            "filename": "missing.pdf",
            "question": "What is this document about?"
        }
    )

    assert response.status_code == 404
    assert response.json()["detail"] == (
        "File not found. Please upload the file first."
    )



def test_ask_returns_fallback_for_irrelevant_question():
    documents["test.pdf"] = {
        "chunks": ["The document discusses Python programming."],
        "embeddings": [[1.0, 0.0]]
    }

    with patch("main.get_embedding", return_value=[0.0, 1.0]):
        response = client.post(
            "/ask",
            json={
                "filename": "test.pdf",
                "question": "What is the weather on Mars?"
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "I couldn't find relevant information in this document."
    )

    documents.pop("test.pdf", None)



def test_ask_returns_answer_when_relevant_information_exists():
    documents["test.pdf"] = {
        "chunks": ["Python is a high-level programming language."],
        "embeddings": [[1.0, 0.0]]
    }

    fake_response = Mock()
    fake_response.json.return_value = {
        "response": "Python is a high-level programming language."
    }
    fake_response.raise_for_status.return_value = None

    with patch("main.get_embedding", return_value=[1.0, 0.0]), \
         patch("main.httpx.post", return_value=fake_response):
        response = client.post(
            "/ask",
            json={
                "filename": "test.pdf",
                "question": "What is Python?"
            }
        )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Python is a high-level programming language."
    )

    documents.pop("test.pdf", None)