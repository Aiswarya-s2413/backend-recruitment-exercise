import pytest
import boto3
from fastapi.testclient import TestClient
from moto import mock_dynamodb, mock_s3
import os
from unittest.mock import patch

# Set a dummy service token before the app is imported
os.environ["SERVICE_TOKEN"] = "test-service-token"

# Set dummy AWS credentials for moto
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

from aws_service.app.main import app

@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@pytest.fixture(scope="function")
def setup_mocks(aws_credentials):
    """Set up mocked AWS services"""
    with mock_dynamodb(), mock_s3():
        # Create mocked DynamoDB table
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="DocumentsMetadata",
            KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )

        # Create mocked S3 bucket
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="pdf-service-bucket")

        yield

@pytest.fixture(scope="function")
def client(setup_mocks):
    """Test client with mocked AWS services"""
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="module")
def basic_auth_creds():
    """Provides the username/password tuple for HTTP Basic Auth."""
    return ("admin", "password")

def test_create_document(client, basic_auth_creds):
    """Test creating a document in DynamoDB"""
    doc_data = {
        "doc_id": "test-doc-123",
        "filename": "test.pdf",
    }

    response = client.post("/aws/documents", json=doc_data, auth=basic_auth_creds)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["doc_id"] == "test-doc-123"

def test_get_document(client, basic_auth_creds):
    """Test retrieving a document from DynamoDB"""
    # First create a document
    doc_data = {
        "doc_id": "test-doc-456",
        "filename": "test.pdf",
        "tags": {"category": "test"}
    }
    client.post("/aws/documents", json=doc_data, auth=basic_auth_creds)

    # Now retrieve it
    response = client.get("/aws/documents/test-doc-456", auth=basic_auth_creds)

    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "test-doc-456"
    assert data["filename"] == "test.pdf"
    assert data["tags"] == {"category": "test"}

def test_update_document(client, basic_auth_creds):
    """Test updating a document in DynamoDB"""
    # Create document
    doc_data = {
        "doc_id": "test-doc-789",
        "filename": "test.pdf",
        "tags": {"category": "test"}
    }
    client.post("/aws/documents", json=doc_data, auth=basic_auth_creds)

    # Update it
    update_data = {
        "tags": {"category": "updated", "new_tag": "value"},
        "s3_key": "new-s3-key.pdf"
    }
    response = client.put("/aws/documents/test-doc-789", json=update_data, auth=basic_auth_creds)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # Verify update
    response = client.get("/aws/documents/test-doc-789", auth=basic_auth_creds)
    data = response.json()
    assert data["tags"] == {"category": "updated", "new_tag": "value"}
    assert data["s3_key"] == "new-s3-key.pdf"

def test_delete_document(client, basic_auth_creds):
    """Test deleting a document from DynamoDB"""
    # Create document
    doc_data = {
        "doc_id": "test-doc-del",
        "filename": "test.pdf",
        "s3_key": "test-doc-del/test.pdf"
    }
    client.post("/aws/documents", json=doc_data, auth=basic_auth_creds)

    # Delete it
    response = client.delete("/aws/documents/test-doc-del", auth=basic_auth_creds)

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    # Verify it's gone
    response = client.get("/aws/documents/test-doc-del", auth=basic_auth_creds)
    assert response.status_code == 404

def test_delete_document_with_s3(client, basic_auth_creds):
    """Test deleting a document with S3 cleanup"""
    # Create document with S3 key
    doc_data = {
        "doc_id": "test-doc-s3",
        "filename": "test.pdf",
        "s3_key": "test-doc-s3/test.pdf"
    }
    client.post("/aws/documents", json=doc_data, auth=basic_auth_creds)

    # Put an object in mocked S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket="pdf-service-bucket", Key="test-doc-s3/test.pdf", Body=b"test content")

    # Delete with S3 cleanup
    response = client.delete("/aws/documents/test-doc-s3?delete_s3=true", auth=basic_auth_creds)

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

def test_get_nonexistent_document(client, basic_auth_creds):
    """Test getting a document that doesn't exist"""
    response = client.get("/aws/documents/nonexistent-doc", auth=basic_auth_creds)

    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]

@patch("requests.post")
def test_index_document(mock_requests, client, basic_auth_creds):
    """Test that POST /aws/documents/{doc_id}/index invokes RAG Module"""
    # Mock the RAG module response
    mock_requests.return_value.json.return_value = {"status": "success"}

    response = client.post("/aws/documents/test-doc-123/index", auth=basic_auth_creds)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "rag_response" in data

    # Verify the HTTP call was made to RAG module
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[0][0] == "http://rag_module:8001/rag/index"
    assert call_args[1]["json"] == {"document_ids": ["test-doc-123"]}
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-service-token"

@patch("requests.post")
def test_aws_query(mock_requests, client, basic_auth_creds):
    """Test that POST /aws/query invokes RAG Module and returns result"""
    # Mock the RAG module response
    rag_response = {
        "run_id": "test-run-123",
        "answer": "Test answer",
        "tokens_consumed": 100,
        "tokens_generated": 50,
        "response_time_ms": 150.5,
        "confidence_score": 0.85
    }
    mock_requests.return_value.json.return_value = rag_response

    query_data = {
        "document_ids": ["doc1", "doc2"],
        "question": "What is the answer?"
    }

    response = client.post("/aws/query", json=query_data, auth=basic_auth_creds)

    assert response.status_code == 200
    data = response.json()
    assert data == rag_response

    # Verify the HTTP call was made to RAG module
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[0][0] == "http://rag_module:8001/rag/query"
    assert call_args[1]["json"] == query_data
    assert call_args[1]["headers"]["Authorization"] == "Bearer test-service-token"

def test_unauthorized_access():
    """Test accessing endpoints without authentication"""
    # Create a separate client for this test that doesn't depend on mocks
    test_client = TestClient(app)
    
    response = test_client.post("/aws/documents", json={"doc_id": "test", "filename": "test.pdf"})
    assert response.status_code == 401

    response = test_client.get("/aws/documents/test")
    assert response.status_code == 401

    response = test_client.put("/aws/documents/test", json={"tags": {}})
    assert response.status_code == 401

    response = test_client.delete("/aws/documents/test")
    assert response.status_code == 401

    response = test_client.post("/aws/documents/test/index")
    assert response.status_code == 401

    response = test_client.post("/aws/query", json={"document_ids": ["doc1"], "question": "test"})
    assert response.status_code == 401