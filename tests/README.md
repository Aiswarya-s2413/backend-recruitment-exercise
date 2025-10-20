# Tests

This directory contains unit tests for all services in the backend recruitment exercise.

## Test Structure

- `test_pdf_service.py`: Tests for PDF Service (upload, retrieval)
- `test_rag_module.py`: Tests for RAG Module (chunking, embedding, query)
- `test_aws_service.py`: Tests for AWS Service (CRUD operations, indexing, querying)
- `test_metrics_lambda.py`: Tests for Metrics Lambda (DynamoDB writes)

## Running Tests

The recommended way to run all tests is to use the `run_all_tests.sh` script in the project's root directory. This script handles dependency installation and execution for each service automatically.

From the project root:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

If you need to run tests for a single service, you can do so by navigating to its directory and using `poetry run pytest`. For detailed instructions, please refer to the main `README.md` in the project root.

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