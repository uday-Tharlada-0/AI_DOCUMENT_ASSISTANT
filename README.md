# AI Document Assistant

A Python-based document question-answering application built with FastAPI.

## Features

- Upload PDF documents.
- Extract text using PyMuPDF.
- Ask questions about uploaded documents.
- Retrieve relevant text using keyword matching.
- Validate requests using Pydantic.
- Interactive API documentation with Swagger UI.

## Tech Stack

- Python
- FastAPI
- Uvicorn
- PyMuPDF
- Pydantic
- uv

## Installation

Clone the repository:

    git clone https://github.com/uday-Tharlada-0/AI_DOCUMENT_ASSISTANT.git

Navigate into the project:

    cd ai-document-assistant

Install dependencies:

    uv sync

Start the application:

    uv run uvicorn main:app --reload

Open the API documentation:

    http://127.0.0.1:8000/docs

## Current Limitations

- Documents are stored in memory.
- Search uses basic keyword matching.
- AI-generated answers and semantic retrieval are not implemented yet.

## Future Improvements

- Integrate an LLM for natural-language answers.
- Implement embeddings and Retrieval-Augmented Generation (RAG).
- Add persistent document storage.
- Deploy the application.
