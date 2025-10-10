import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.main import app, get_db
from app import models
import uuid
import io

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_pdf_metadata.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)

@pytest.fixture(scope="function", autouse=True)
def setup_database():
    # Create tables
    Base.metadata.create_all(bind=engine)
    yield
    # Drop tables and clean up
    Base.metadata.drop_all(bind=engine)
    # Remove test db file if exists
    test_db_path = "./test_pdf_metadata.db"
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except PermissionError:
            pass  # May be locked on Windows

@pytest.fixture
def auth_token():
    # Use the dummy credentials from auth.py
    response = client.post("/auth/login", data={"username": "admin", "password": "password"})
    assert response.status_code == 200
    return response.json()["access_token"]

def create_sample_pdf():
    """Create a minimal PDF for testing"""
    # Simple PDF bytes - this is a valid minimal PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 5 0 R
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 750 Td
(This is a test PDF document.) Tj
ET
endstream
endobj
5 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000274 00000 n
0000000368 00000 n
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
459
%%EOF"""
    from io import BytesIO
    return BytesIO(pdf_content)

def test_upload_pdf(auth_token):
    """Test uploading a sample PDF and verifying doc_id is returned"""
    pdf_content = create_sample_pdf()

    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = client.post("/pdf/upload", files=files, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    doc = data[0]
    assert "doc_id" in doc
    assert isinstance(doc["doc_id"], str)
    # Validate UUID format
    uuid.UUID(doc["doc_id"])
    assert doc["filename"] == "test.pdf"
    assert "upload_timestamp" in doc
    assert "extracted_text" in doc
    assert len(doc["extracted_text"]) > 0  # Basic check that text was extracted

def test_retrieve_document_metadata(auth_token):
    """Test retrieving metadata and extracted text for a doc_id"""
    # First upload a PDF to get a doc_id
    pdf_content = create_sample_pdf()

    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    headers = {"Authorization": f"Bearer {auth_token}"}

    upload_response = client.post("/pdf/upload", files=files, headers=headers)
    assert upload_response.status_code == 200
    doc_id = upload_response.json()[0]["doc_id"]

    # Now retrieve the document
    response = client.get(f"/pdf/documents/{doc_id}", headers=headers)

    assert response.status_code == 200
    doc = response.json()
    assert doc["doc_id"] == doc_id
    assert doc["filename"] == "test.pdf"
    assert "upload_timestamp" in doc
    assert "extracted_text" in doc
    assert len(doc["extracted_text"]) > 0

def test_retrieve_nonexistent_document(auth_token):
    """Test retrieving a document that doesn't exist"""
    fake_doc_id = str(uuid.uuid4())
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = client.get(f"/pdf/documents/{fake_doc_id}", headers=headers)

    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]

def test_upload_non_pdf_file(auth_token):
    """Test uploading a non-PDF file should fail"""
    txt_content = b"This is not a PDF file."

    files = {"files": ("test.txt", txt_content, "text/plain")}
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = client.post("/pdf/upload", files=files, headers=headers)

    assert response.status_code == 400
    assert "is not a PDF" in response.json()["detail"]

def test_unauthorized_access():
    """Test accessing endpoints without authentication"""
    response = client.get("/pdf/documents/some-id")
    assert response.status_code == 401

    pdf_content = create_sample_pdf()
    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    response = client.post("/pdf/upload", files=files)
    assert response.status_code == 401