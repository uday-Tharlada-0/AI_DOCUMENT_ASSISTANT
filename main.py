from fastapi import FastAPI,File,UploadFile,HTTPException
from pydantic import BaseModel
import pymupdf 
import re
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

    stop_words = {
        "what", "is", "the", "a", "an", "of", "in",
        "to", "and", "how", "does", "do", "are", "can"
    }

    def tokenize(text):
        words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        return set(words) - stop_words

    question_words = tokenize(request.question)

    # Split the document into sentences instead of individual lines
    text = documents[request.filename]
    sentences = re.split(r"(?<=[.!?])\s+", text)

    matches = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        sentence_words = tokenize(sentence)
        score = len(question_words.intersection(sentence_words))

        if score > 0:
            # Prefer informative sentences over short headings
            if len(sentence.split()) < 5:
                score -= 0.5

            matches.append((score, sentence))

    if not matches:
        return {
            "question": request.question,
            "answer": "No relevant information found in the document."
        }

    matches.sort(key=lambda item: item[0], reverse=True)

    return {
        "question": request.question,
        "answer": matches[0][1]
    }
