from fastapi import FastAPI,File,UploadFile,HTTPException
import pymupdf 
app =FastAPI(title ="AI Document Assistant") 
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

    return {
    "filename": file.filename,
    "pages": pages,
    "text": text
    }