import pytest
import os
from io import BytesIO
from unittest.mock import MagicMock, patch
import uuid

# Step 1: Create test database and patch BEFORE any app imports
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

TEST_DATABASE_URL = "sqlite:///test.db"
test_engine = create_engine(
    TEST_DATABASE_URL, 
    connect_args={"check_same_thread": False},
    echo=False
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Step 2: Patch at import time using environment or direct module patching
# Import and immediately patch the database module
import pdf_service.app.database as db_module
original_engine = db_module.engine
original_session = db_module.SessionLocal

db_module.engine = test_engine
db_module.SessionLocal = TestingSessionLocal

# Step 3: Now import everything else
from pdf_service.app import models

# Step 5: Import main after everything is set up
from pdf_service.app.main import app, get_db
import pdf_service.app.main as main_module

# Also patch the engine on the main module itself
main_module.database.engine = test_engine

# Step 6: Override database dependency
def override_get_db():
    """Override database dependency to use test database"""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Step 7: Mock S3 client
mock_s3_client = MagicMock()
mock_s3_client.put_object.return_value = None
mock_s3_client.upload_fileobj.return_value = None
main_module.s3_client = mock_s3_client

# Step 8: Create test client
from fastapi.testclient import TestClient
client = TestClient(app)


import os

@pytest.fixture(autouse=True)
def setup_database():
    """Create and drop database tables for each test"""
    # Create tables
    models.Base.metadata.create_all(bind=test_engine)
    yield
    # Drop tables
    models.Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="module")
def basic_auth_creds():
    """Provides the username/password tuple for HTTP Basic Auth."""
    return ("admin", "password")


def create_sample_pdf():
    """Create a minimal PDF for testing"""
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
    return BytesIO(pdf_content)


def test_upload_pdf(basic_auth_creds):
    """Test uploading a sample PDF and verifying doc_id is returned"""
    pdf_content = create_sample_pdf()

    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    response = client.post("/pdf/upload", files=files, auth=basic_auth_creds)

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


def test_retrieve_document_metadata(basic_auth_creds):
    """Test retrieving metadata and extracted text for a doc_id"""
    # First upload a PDF to get a doc_id
    pdf_content = create_sample_pdf()

    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    upload_response = client.post("/pdf/upload", files=files, auth=basic_auth_creds)
    assert upload_response.status_code == 200
    doc_id = upload_response.json()[0]["doc_id"]

    # Now retrieve the document
    response = client.get(f"/pdf/documents/{doc_id}", auth=basic_auth_creds)

    assert response.status_code == 200
    doc = response.json()
    assert doc["doc_id"] == doc_id
    assert doc["filename"] == "test.pdf"
    assert "upload_timestamp" in doc
    assert "extracted_text" in doc
    assert len(doc["extracted_text"]) > 0


def test_retrieve_nonexistent_document(basic_auth_creds):
    """Test retrieving a document that doesn't exist"""
    fake_doc_id = str(uuid.uuid4())
    response = client.get(f"/pdf/documents/{fake_doc_id}", auth=basic_auth_creds)

    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]


def test_upload_non_pdf_file(basic_auth_creds):
    """Test uploading a non-PDF file should fail"""
    txt_content = b"This is not a PDF file."

    files = {"files": ("test.txt", txt_content, "text/plain")}
    response = client.post("/pdf/upload", files=files, auth=basic_auth_creds)

    assert response.status_code == 400
    assert "is not a PDF" in response.json()["detail"]


def test_unauthorized_access():
    """Test accessing endpoints without authentication"""
    response = client.get("/pdf/documents/some-id")  # No auth
    assert response.status_code == 401  # Should be 401 Unauthorized

    pdf_content = create_sample_pdf()
    files = {"files": ("test.pdf", pdf_content, "application/pdf")}
    response = client.post("/pdf/upload", files=files)
    assert response.status_code == 401