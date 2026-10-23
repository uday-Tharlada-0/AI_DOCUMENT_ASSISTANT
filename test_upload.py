from fastapi.testclient import TestClient
from main import app 
import pymupdf
from unittest.mock import patch
client = TestClient(app)
def test_upload_rejects_non_pdf():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"Hello, World!", "text/plain")}
    )
    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid file format. Only PDF files are allowed."}



def test_upload_accepts_valid_pdf():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 50), "Python is a programming language.")
    pdf_bytes = pdf.tobytes()
    pdf.close()

    with patch("main.get_embedding", return_value=[0.1, 0.2, 0.3]):
        response = client.post(
            "/upload",
            files={"file": ("notes.pdf", pdf_bytes, "application/pdf")}
        )

    assert response.status_code == 200
    assert response.json()["filename"] == "notes.pdf"
    assert response.json()["pages"] == 1
def test_upload_rejects_empty_pdf():
    pdf = pymupdf.open()
    pdf.new_page()  # Creates a blank page
    pdf_bytes = pdf.tobytes()
    pdf.close()

    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "No extractable text found in this PDF."
    )