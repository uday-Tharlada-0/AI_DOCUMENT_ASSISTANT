from fastapi import FastAPI,File,UploadFile,HTTPException
from pydantic import BaseModel
import pymupdf 
import re
import httpx 
import os
import math
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks
OLLAMA_URL = os.getenv("OLLAMA_URL","http://172.28.240.1:11434/api/generate")
EMBEDDING_URL = os.getenv(
    "OLLAMA_EMBED_URL",
    "http://172.28.240.1:11434/api/embed"
)

EMBEDDING_MODEL = "nomic-embed-text"
app =FastAPI(title ="AI Document Assistant") 
class QuestionRequest(BaseModel):
    filename: str
    question: str
documents = {}
def get_embedding(text: str) -> list[float]:
    try:
        response = httpx.post(
            EMBEDDING_URL,
            json={
                "model": EMBEDDING_MODEL,
                "input": text
            },
            timeout=120
        )
        response.raise_for_status()

        return response.json()["embeddings"][0]

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate text embeddings."
        ) from error
def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float]
) -> float:
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)
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
    documents[file.filename] =  chunk_text(text)
    chunks = chunk_text(text)

    chunk_embeddings = []

    for chunk in chunks:
        embedding = get_embedding(chunk)
        chunk_embeddings.append(embedding)

    documents[file.filename] = {
    "chunks": chunks,
    "embeddings": chunk_embeddings
}
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

    # Retrieve the stored chunks and their embeddings
    document = documents[request.filename]
    chunks = document["chunks"]
    chunk_embeddings = document["embeddings"]

    # Convert the question into an embedding
    question_embedding = get_embedding(request.question)

    # Calculate similarity between the question and each chunk
    scored_chunks = []

    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(
            question_embedding,
            chunk_embedding
        )

        scored_chunks.append((score, chunk))

    # Rank chunks by similarity, highest first
    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    # Select the three most relevant chunks
    if not scored_chunks or scored_chunks[0][0] < 0.45:
        return {
        "question": request.question,
        "answer": "I couldn't find relevant information in this document."
    }

    top_chunks = scored_chunks[:3]

    context = "\n\n".join(
        chunk for score, chunk in top_chunks
    )

    # Prepare the prompt for Gemma
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

    # Ask Gemma to generate the answer
    try:
        response = httpx.post(
            OLLAMA_URL,
            json={
                "model": "gemma:2b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        answer = response.json()["response"].strip()

    except httpx.HTTPError as error:
        raise HTTPException(
            status_code=502,
            detail="Unable to communicate with the local AI model."
        ) from error

    return {
        "question": request.question,
        "answer": answer
    }