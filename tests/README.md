# Tests

This directory contains unit tests for all services in the backend recruitment exercise.

## Test Structure

- `test_pdf_service.py`: Tests for PDF Service (upload, retrieval)
- `test_rag_module.py`: Tests for RAG Module (chunking, embedding, query)
- `test_aws_service.py`: Tests for AWS Service (CRUD operations, indexing, querying)
- `test_metrics_lambda.py`: Tests for Metrics Lambda (DynamoDB writes)

## Running Tests

### Prerequisites

1. Install dependencies for each service:
   ```bash
   # PDF Service
   cd ../pdf_service
   poetry install

   # RAG Module
   cd ../rag_module
   poetry install

   # AWS Service
   cd ../aws_service
   poetry install

   # Metrics Lambda
   cd ../metrics_lambda
   poetry install
   ```

### Run Individual Service Tests

```bash
# PDF Service tests
cd ../pdf_service
pytest ../../tests/test_pdf_service.py -v

# RAG Module tests
cd ../rag_module
pytest ../../tests/test_rag_module.py -v

# AWS Service tests
cd ../aws_service
pytest ../../tests/test_aws_service.py -v

# Metrics Lambda tests
cd ../metrics_lambda
pytest ../../tests/test_metrics_lambda.py -v
```

### Run All Tests

From the project root:
```bash
# Run all tests (requires all services to have dependencies installed)
pytest tests/ -v
```

## Test Coverage

The tests cover all requirements from the testing & validation section:

### PDF Service
-  Upload a sample PDF, verify that a valid doc_id is returned
-  Retrieve metadata and extracted text for that doc_id

### RAG Module
-  Test chunking logic on a long text: ensure correct chunk sizes/counts
-  Test embedding logic by mocking Pinecone or the embedding API
-  Test the /rag/query endpoint by stubbing Pinecone and the LLM: verify response includes run_id, answer, tokens_consumed, tokens_generated, response_time_ms, and confidence_score
-  Verify that an HTTP call is made to the Lambda's URL with the correct metrics payload

### AWS Service
-  Test CRUD operations against DynamoDB (using moto): Create, retrieve, update, and delete items in DocumentsMetadata
-  Test that POST /aws/documents/{doc_id}/index invokes the RAG Module's /rag/index endpoint (mock the HTTP call)
-  Test that POST /aws/query invokes the RAG Module's /rag/query endpoint and returns its result

### Metrics Lambda
-  Test that, given a valid JSON event, the Lambda writes one item to AgentMetrics (using moto to mock DynamoDB)

## Mocking Strategy

- **AWS Services**: Using `moto` library to mock DynamoDB and S3
- **External APIs**: Using `unittest.mock` to patch HTTP calls and API responses
- **Authentication**: Using dummy credentials ("admin"/"password") as defined in each service's auth.py
- **Database**: Using in-memory SQLite for PDF service tests

## Notes

- Tests are designed to run independently
- All external dependencies are mocked to ensure reliable test execution
- Authentication is handled using the dummy user credentials from each service
- Database operations use test-specific databases that are cleaned up after each test