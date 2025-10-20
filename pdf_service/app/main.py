import os
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
import fitz  # PyMuPDF
import boto3
from sqlalchemy.orm import Session
from . import models, schemas, crud, database, auth
from dotenv import load_dotenv

load_dotenv()

# Create uploads directory
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="PDF Service")

# S3 client setup
localstack_endpoint = os.getenv("LOCALSTACK_ENDPOINT")
s3_client = boto3.client(
    's3',
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "test"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "test"),
    endpoint_url=localstack_endpoint if localstack_endpoint else None
)
s3_bucket = os.getenv("S3_BUCKET", "pdf-service-bucket")

@app.on_event("startup")
def on_startup():
    # Create DB tables
    models.Base.metadata.create_all(bind=database.engine)

def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# POST /upload
@app.post("/pdf/upload", response_model=list[schemas.DocumentResponse])
async def upload_pdfs(files: list[UploadFile] = File(...), current_user: str = Depends(auth.verify_token), db: Session = Depends(get_db)):
    uploaded_docs = []

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{file.filename} is not a PDF")

        # Read file content
        content = await file.read()

        # Extract text from bytes
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        # Generate doc_id
        doc_id = str(uuid.uuid4())

        # Store file in S3 (or LocalStack S3 if configured)
        try:
            key = f"{doc_id}/{file.filename}"
            s3_client.put_object(Bucket=s3_bucket, Key=key, Body=content)
            file_location = f"s3:{key}"
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

        # Store metadata
        db_doc = crud.create_document(db, doc_id=doc_id, filename=file.filename, extracted_text=text, file_location=file_location)
        uploaded_docs.append(db_doc)

    return uploaded_docs

# GET /documents/{doc_id}
@app.get("/pdf/documents/{doc_id}", response_model=schemas.DocumentResponse)
def read_document(doc_id: str, current_user: str = Depends(auth.verify_token), db: Session = Depends(get_db)):
    db_doc = crud.get_document(db, doc_id)
    if not db_doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return db_doc

# GET /documents?page=1&limit=10
@app.get("/pdf/documents", response_model=list[schemas.DocumentResponse])
def read_documents(page: int = Query(1, ge=1), limit: int = Query(10, ge=1), current_user: str = Depends(auth.verify_token), db: Session = Depends(get_db)):
    skip = (page - 1) * limit
    docs = crud.get_documents(db, skip=skip, limit=limit)
    return docs
