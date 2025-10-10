import pytest
import boto3
from fastapi.testclient import TestClient
from moto import mock_dynamodb, mock_s3
import os
from app.main import app
from unittest.mock import patch

# Mock AWS services
@mock_dynamodb
@mock_s3
@pytest.fixture(scope="function")
def setup_mocks():
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

client = TestClient(app)

@pytest.fixture
def auth_token():
    # Use dummy credentials
    response = client.post("/auth/login", data={"username": "admin", "password": "password"})
    assert response.status_code == 200
    return response.json()["access_token"]

@mock_dynamodb
@mock_s3
def test_create_document(auth_token, setup_mocks):
    """Test creating a document in DynamoDB"""
    headers = {"Authorization": f"Bearer {auth_token}"}
    doc_data = {
        "doc_id": "test-doc-123",
        "filename": "test.pdf",
        "tags": {"category": "test"},
        "s3_key": "test-doc-123/test.pdf"
    }

    response = client.post("/aws/documents", json=doc_data, headers=headers)

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
    assert data["doc_id"] == "test-doc-123"

@mock_dynamodb
@mock_s3
def test_get_document(auth_token, setup_mocks):
    """Test retrieving a document from DynamoDB"""
    # First create a document
    headers = {"Authorization": f"Bearer {auth_token}"}
    doc_data = {
        "doc_id": "test-doc-456",
        "filename": "test.pdf",
        "tags": {"category": "test"}
    }
    client.post("/aws/documents", json=doc_data, headers=headers)

    # Now retrieve it
    response = client.get("/aws/documents/test-doc-456", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "test-doc-456"
    assert data["filename"] == "test.pdf"
    assert data["tags"] == {"category": "test"}

@mock_dynamodb
@mock_s3
def test_update_document(auth_token, setup_mocks):
    """Test updating a document in DynamoDB"""
    # Create document
    headers = {"Authorization": f"Bearer {auth_token}"}
    doc_data = {
        "doc_id": "test-doc-789",
        "filename": "test.pdf",
        "tags": {"category": "test"}
    }
    client.post("/aws/documents", json=doc_data, headers=headers)

    # Update it
    update_data = {
        "tags": {"category": "updated", "new_tag": "value"},
        "s3_key": "new-s3-key.pdf"
    }
    response = client.put("/aws/documents/test-doc-789", json=update_data, headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    # Verify update
    response = client.get("/aws/documents/test-doc-789", headers=headers)
    data = response.json()
    assert data["tags"] == {"category": "updated", "new_tag": "value"}
    assert data["s3_key"] == "new-s3-key.pdf"

@mock_dynamodb
@mock_s3
def test_delete_document(auth_token, setup_mocks):
    """Test deleting a document from DynamoDB"""
    # Create document
    headers = {"Authorization": f"Bearer {auth_token}"}
    doc_data = {
        "doc_id": "test-doc-del",
        "filename": "test.pdf",
        "s3_key": "test-doc-del/test.pdf"
    }
    client.post("/aws/documents", json=doc_data, headers=headers)

    # Delete it
    response = client.delete("/aws/documents/test-doc-del", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    # Verify it's gone
    response = client.get("/aws/documents/test-doc-del", headers=headers)
    assert response.status_code == 404

@mock_dynamodb
@mock_s3
def test_delete_document_with_s3(auth_token, setup_mocks):
    """Test deleting a document with S3 cleanup"""
    # Create document with S3 key
    headers = {"Authorization": f"Bearer {auth_token}"}
    doc_data = {
        "doc_id": "test-doc-s3",
        "filename": "test.pdf",
        "s3_key": "test-doc-s3/test.pdf"
    }
    client.post("/aws/documents", json=doc_data, headers=headers)

    # Put an object in mocked S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.put_object(Bucket="pdf-service-bucket", Key="test-doc-s3/test.pdf", Body=b"test content")

    # Delete with S3 cleanup
    response = client.delete("/aws/documents/test-doc-s3?delete_s3=true", headers=headers)

    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

@mock_dynamodb
@mock_s3
def test_get_nonexistent_document(auth_token, setup_mocks):
    """Test getting a document that doesn't exist"""
    headers = {"Authorization": f"Bearer {auth_token}"}

    response = client.get("/aws/documents/nonexistent-doc", headers=headers)

    assert response.status_code == 404
    assert "Document not found" in response.json()["detail"]

@patch("requests.post")
@mock_dynamodb
@mock_s3
def test_index_document(mock_requests, auth_token, setup_mocks):
    """Test that POST /aws/documents/{doc_id}/index invokes RAG Module"""
    # Mock the RAG module response
    mock_requests.return_value.json.return_value = {"status": "success"}

    headers = {"Authorization": f"Bearer {auth_token}"}

    response = client.post("/aws/documents/test-doc-123/index", headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "rag_response" in data

    # Verify the HTTP call was made to RAG module
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[0][0] == "http://rag_module:8001/rag/index"
    assert call_args[1]["json"] == ["test-doc-123"]
    assert call_args[1]["headers"]["Authorization"] == "Bearer None"  # SERVICE_TOKEN is None in test

@patch("requests.post")
@mock_dynamodb
@mock_s3
def test_aws_query(mock_requests, auth_token, setup_mocks):
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

    headers = {"Authorization": f"Bearer {auth_token}"}
    query_data = {
        "document_ids": ["doc1", "doc2"],
        "question": "What is the answer?"
    }

    response = client.post("/aws/query", json=query_data, headers=headers)

    assert response.status_code == 200
    data = response.json()
    assert data == rag_response

    # Verify the HTTP call was made to RAG module
    mock_requests.assert_called_once()
    call_args = mock_requests.call_args
    assert call_args[0][0] == "http://rag_module:8001/rag/query"
    assert call_args[1]["json"] == query_data
    assert call_args[1]["headers"]["Authorization"] == "Bearer None"

def test_unauthorized_access():
    """Test accessing endpoints without authentication"""
    response = client.post("/aws/documents", json={"doc_id": "test", "filename": "test.pdf"})
    assert response.status_code == 401

    response = client.get("/aws/documents/test")
    assert response.status_code == 401

    response = client.put("/aws/documents/test", json={"tags": {}})
    assert response.status_code == 401

    response = client.delete("/aws/documents/test")
    assert response.status_code == 401

    response = client.post("/aws/documents/test/index")
    assert response.status_code == 401

    response = client.post("/aws/query", json={"document_ids": ["doc1"], "question": "test"})
    assert response.status_code == 401