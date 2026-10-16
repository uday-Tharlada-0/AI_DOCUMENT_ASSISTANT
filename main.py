from fastapi import FastAPI,File,UploadFile,HTTPException
from pydantic import BaseModel
import pymupdf 
import re
import httpx 
import os
OLLAMA_URL = os.getenv("OLLAMA_URL","http://172.28.240.1:11434/api/generate")
app =FastAPI(title ="AI Document Assistant") 
class QuestionRequest(BaseModel):
    filename: str
    question: str
documents = {}
@app.get("/")
def home():
    return {"message":"AI Document Assistant is running"}
@app.post("/upload")
async def upload_file(file: UploadFile =File(...)):
    if not file.filename.endswith((".pdf")):
        raise HTTPException(status_code=400, detail="Invalid file format. Only PDF files are allowed.")
    content = await file.read()
    try:
        with pymupdf.open(stream=content, filetype="pdf") as document:
            pages = len(document)
            text = "\n".join(page.get_text() for page in document)
    except Exception:
        raise HTTPException(
        status_code=400,
        detail="Unable to read this PDF."
    )

    if not text.strip():
        raise HTTPException(
        status_code=400,
        detail="No extractable text found in this PDF."
    )
    documents[file.filename] = text
    return {
    "filename": file.filename,
    "pages": pages,
    "message": "File uploaded and text extracted successfully."
    }
@app.post("/ask")
def ask_question(request: QuestionRequest):
    if request.filename not in documents:
        raise HTTPException(
            status_code=404,
            detail="File not found. Please upload the file first."
        )

    text = documents[request.filename]

    # Retrieve relevant paragraphs using keyword matching
    stop_words = {
        "what", "is", "the", "a", "an", "of", "in",
        "to", "and", "how", "does", "do", "are", "can"
    }

    def tokenize(value):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", value.lower())
        return set(words) - stop_words

    question_words = tokenize(request.question)
    paragraphs = [
        p.strip() for p in text.split("\n") if p.strip()
    ]

    matches = []

    for paragraph in paragraphs:
        paragraph_words = tokenize(paragraph)
        score = len(question_words & paragraph_words)

        if score > 0:
            matches.append((score, paragraph))

    if not matches:
        return {
            "question": request.question,
            "answer": "I couldn't find relevant information in this document."
        }

    matches.sort(key=lambda item: item[0], reverse=True)

    # Provide the best matching text as context
    context = "\n".join(
        paragraph for _, paragraph in matches[:5]
    )

    prompt = f"""
You are an AI Document Assistant.

Answer the question using only the document context below.
If the context does not contain enough information, say so.
Do not invent facts.

Document context:
{context}

Question:
{request.question}

Answer:
"""

    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["response"].strip()

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Unable to communicate with the local AI model."
        )

    return {
        "question": request.question,
        "answer": answer,
    }