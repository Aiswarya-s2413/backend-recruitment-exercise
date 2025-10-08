import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
import fitz  # PyMuPDF
from sqlalchemy.orm import Session
from . import models, schemas, crud, database

# Create uploads directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize DB
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="PDF Service")

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# POST /upload
@app.post("/pdf/upload", response_model=list[schemas.DocumentResponse])
async def upload_pdfs(files: list[UploadFile] = File(...), db: Session = Depends(get_db)):
    uploaded_docs = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")
        
        # Save locally
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Extract text
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Store metadata
        db_doc = crud.create_document(db, filename=file.filename, extracted_text=text)
        uploaded_docs.append(db_doc)

    return uploaded_docs

# GET /documents/{doc_id}
@app.get("/pdf/documents/{doc_id}", response_model=schemas.DocumentResponse)
def read_document(doc_id: str, db: Session = Depends(get_db)):
    db_doc = crud.get_document(db, doc_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_doc

# GET /documents?page=1&limit=10
@app.get("/pdf/documents", response_model=list[schemas.DocumentResponse])
def read_documents(page: int = Query(1, ge=1), limit: int = Query(10, ge=1), db: Session = Depends(get_db)):
    skip = (page - 1) * limit
    docs = crud.get_documents(db, skip=skip, limit=limit)
    return docs
