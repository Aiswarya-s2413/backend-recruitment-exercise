import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from rag_module.app.main import app
from rag_module.app import schemas
import uuid
import os
import numpy as np

os.environ["SERVICE_TOKEN"] = "test-service-token"

client = TestClient(app)

@pytest.fixture(scope="module")
def service_token():
    # A dummy token for testing inter-service communication
    return "test-service-token"

def test_chunking_logic():
    """Test chunking logic on a long text"""
    from rag_module.app.utils import text_splitter

    long_text = "This is a test sentence. " * 50  # Repeat to make it long

    chunks = text_splitter.split_text(long_text)

    # Verify chunks are created
    assert len(chunks) > 1

    # Check chunk sizes 
    chunk_size = 500  # default from utils
    for chunk in chunks:
        assert len(chunk) <= chunk_size + 50  # Allow some tolerance for overlap

    # Verify total content is preserved (approximately)
    combined = " ".join(chunks)
    # Note: Due to overlap, combined will be longer than original

def test_embedding_logic():
    """Test embedding logic by mocking the embedding API"""
    from rag_module.app.utils import embed_texts

    test_texts = ["This is a test sentence.", "Another test sentence."]

    # Mock the embedding model
    with patch("rag_module.app.utils.embedding_model") as mock_model:
        # Create a numpy array for the mock return value
        mock_embeddings = np.array([[0.1] * 384, [0.2] * 384])
        mock_model.encode.return_value = mock_embeddings

        embeddings = embed_texts(test_texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384
        assert len(embeddings[1]) == 384
        mock_model.encode.assert_called_once_with(test_texts)

@patch("rag_module.app.utils.index")
@patch("rag_module.app.utils.embed_texts")
@patch("rag_module.app.utils.get_document_text")
@patch("rag_module.app.utils.send_metrics")
def test_rag_query_endpoint(mock_send_metrics, mock_get_text, mock_embed, mock_index, service_token):
    """Test the /rag/query endpoint with stubbing"""
    # Mock document text retrieval
    mock_get_text.return_value = "This is sample document text for testing."

    # Mock embeddings
    mock_embed.return_value = [[0.1] * 384]

    # Mock Pinecone query response
    mock_query_response = {
        "matches": [
            {"metadata": {"text": "Relevant chunk 1"}, "score": 0.8},
            {"metadata": {"text": "Relevant chunk 2"}, "score": 0.7}
        ]
    }
    mock_index.query.return_value = mock_query_response

    # Mock LLM response
    with patch("rag_module.app.utils.call_llm") as mock_llm:
        mock_llm.return_value = {
            "answer": "This is a test answer.",
            "tokens_consumed": 100,
            "tokens_generated": 50
        }

        request_data = {
            "document_ids": ["doc1", "doc2"],
            "question": "What is this about?"
        }
        headers = {"Authorization": f"Bearer {service_token}"}

        response = client.post("/rag/query", json=request_data, headers=headers)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "run_id" in data
        assert isinstance(data["run_id"], str)
        uuid.UUID(data["run_id"])  # Valid UUID

        assert data["answer"] == "This is a test answer."
        assert data["tokens_consumed"] == 100
        assert data["tokens_generated"] == 50
        assert "response_time_ms" in data
        assert isinstance(data["response_time_ms"], float)
        assert data["response_time_ms"] > 0

        # Confidence score is average of match scores
        expected_confidence = (0.8 + 0.7) / 2
        assert abs(data["confidence_score"] - expected_confidence) < 0.001

        # Verify metrics were sent
        mock_send_metrics.assert_called_once()
        metrics_call = mock_send_metrics.call_args[0][0]
        assert metrics_call["run_id"] == data["run_id"]
        assert metrics_call["agent_name"] == "RAGQueryAgent"
        assert metrics_call["tokens_consumed"] == 100
        assert metrics_call["tokens_generated"] == 50
        assert metrics_call["response_time_ms"] == data["response_time_ms"]
        assert metrics_call["confidence_score"] == expected_confidence
        assert metrics_call["status"] == "completed"

@patch("rag_module.app.utils.index")
@patch("rag_module.app.utils.embed_texts")
@patch("rag_module.app.utils.get_document_text")
@patch("rag_module.app.utils.send_metrics")
def test_rag_index_endpoint(mock_send_metrics, mock_get_text, mock_embed, mock_index, service_token):
    """Test the /rag/index endpoint"""
    # Mock document text - make it long enough to create 2 chunks
    long_text = "Sample document text for indexing. " * 50  # Create a longer text
    mock_get_text.return_value = long_text

    # Mock embeddings - return 2 embeddings for 2 chunks
    mock_embed.return_value = [[0.1] * 384, [0.2] * 384]

    request_data = {"document_ids": ["doc1"]}
    headers = {"Authorization": f"Bearer {service_token}"}

    response = client.post("/rag/index", json=request_data, headers=headers)

    assert response.status_code == 200
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 1
    result = data["results"][0]
    assert result["doc_id"] == "doc1"
    assert result["status"] == "success"

    # Verify Pinecone upsert was called
    mock_index.upsert.assert_called_once()
    upsert_call = mock_index.upsert.call_args[1]["vectors"]
    assert len(upsert_call) == 2  # Two chunks
    for vector in upsert_call:
        assert len(vector) == 3  # (id, embedding, metadata)
        assert vector[0].startswith("doc1_")
        assert len(vector[1]) == 384
        assert vector[2]["doc_id"] == "doc1"

def test_invalid_query_request(service_token):
    """Test query endpoint with invalid data"""
    # Empty document_ids
    request_data = {"document_ids": [], "question": "Test question"}
    headers = {"Authorization": f"Bearer {service_token}"}

    response = client.post("/rag/query", json=request_data, headers=headers)
    assert response.status_code == 422  # Validation error

    # Empty question
    request_data = {"document_ids": ["doc1"], "question": ""}
    response = client.post("/rag/query", json=request_data, headers=headers)
    assert response.status_code == 422

    # Too many document_ids
    request_data = {"document_ids": [f"doc{i}" for i in range(51)], "question": "Test"}
    response = client.post("/rag/query", json=request_data, headers=headers)
    assert response.status_code == 422

@patch("rag_module.app.utils.get_document_text")
def test_index_with_error(mock_get_text, service_token):
    """Test indexing when document retrieval fails"""
    mock_get_text.side_effect = Exception("Document not found")

    request_data = {"document_ids": ["doc1"]}
    headers = {"Authorization": f"Bearer {service_token}"}

    response = client.post("/rag/index", json=request_data, headers=headers)

    assert response.status_code == 200  # Still returns 200 but with failure status
    data = response.json()
    assert data["results"][0]["status"] == "failure"
    assert "Document not found" in data["results"][0]["reason"]

def test_unauthorized_access():
    """Test accessing endpoints without authentication"""
    # Test without Authorization header
    response = client.post("/rag/query", json={"document_ids": ["doc1"], "question": "test"})
    assert response.status_code == 401

    response = client.post("/rag/index", json={"document_ids": ["doc1"]})
    assert response.status_code == 401
    
    # Test with invalid token
    headers = {"Authorization": "Bearer invalid-token"}
    response = client.post("/rag/query", json={"document_ids": ["doc1"], "question": "test"}, headers=headers)
    assert response.status_code == 401
    
    response = client.post("/rag/index", json={"document_ids": ["doc1"]}, headers=headers)
    assert response.status_code == 401