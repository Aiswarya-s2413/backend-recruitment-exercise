# Backend Recruitment Exercise

A multi-service RAG (Retrieval-Augmented Generation) system with AWS integration, consisting of PDF service, RAG module, AWS service, and Metrics Lambda.

## Services

- **PDF Service**: Handles PDF document uploads and metadata storage.
- **RAG Module**: Processes documents for indexing and querying using Pinecone and embeddings.
- **AWS Service**: Provides AWS integrations (DynamoDB, S3) for document management.
- **Metrics Lambda**: Collects and stores performance metrics in DynamoDB.

## Development Setup

1. Clone the repository.
2. Copy `.env.example` to `.env` and configure environment variables.
3. Run with Docker Compose: `docker-compose up`

## Testing

The project has a comprehensive test suite for each service. To run all tests for the entire project, use the provided helper script. This script will automatically navigate into each service's directory, install its dependencies, and run its tests.

### Prerequisites

Ensure you have `bash` and `poetry` installed on your system.

### Run All Tests

From the project root directory, make the script executable and run it:
```bash
chmod +x run_all_tests.sh
./run_all_tests.sh
```

## Production Deployment

For production AWS deployments:

1. **AWS Credentials**: Ensure AWS credentials are configured (IAM roles for ECS/EC2 or environment variables).
2. **Environment Variables**:
   - Remove `LOCALSTACK_ENDPOINT` to use real AWS services.
   - Set `METRICS_LAMBDA_NAME` for direct Lambda invocation.
   - Configure Pinecone, DynamoDB table names, S3 bucket, etc.
3. **Deploy Services**:
   - Deploy each service to ECS or EKS.
   - Deploy Metrics Lambda to AWS Lambda.
4. **Database Setup**:
   - Ensure DynamoDB tables exist.
   - Configure S3 bucket.
   - Pinecone index is created automatically.

## Features Implemented

- Comprehensive error handling with custom exceptions and logging.
- Cloud configuration for production AWS beyond LocalStack.
- Metrics collection and storage with robust error handling.