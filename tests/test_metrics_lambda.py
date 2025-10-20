import pytest
import json
from moto import mock_dynamodb
from botocore.exceptions import ClientError
from decimal import Decimal
import boto3

@mock_dynamodb
def test_store_agent_metrics_valid_event():
    """Test that the Lambda writes one item to AgentMetrics given a valid JSON event"""
    # Create mocked DynamoDB table
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        # Use the same table name as in the handler
        # to ensure consistency
        TableName="AgentMetrics",
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"}
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"}
        ],
        BillingMode="PAY_PER_REQUEST"
    )

    # Test event
    test_event = {
        "run_id": "test-run-123",
        "agent_name": "TestAgent",
        "tokens_consumed": 150,
        "tokens_generated": 75,
        "response_time_ms": 250.5,
        "confidence_score": 0.92,
        "status": "completed"
    }

    # Mock context (minimal)
    context = {}

    # Import handler here to ensure mocks are active
    # FIXED: Use absolute import from the project root
    from metrics_lambda.handler import store_agent_metrics

    # Call the handler
    response = store_agent_metrics(test_event, context)

    # Verify response
    assert response["statusCode"] == 200
    response_body = json.loads(response["body"])
    assert response_body["status"] == "ok"
    assert response_body["run_id"] == "test-run-123"

    # Verify item was written to DynamoDB
    items = table.scan()["Items"]
    assert len(items) == 1
    item = items[0]

    assert item["run_id"] == "test-run-123"
    assert "timestamp" in item
    assert item["agent_name"] == "TestAgent"
    assert item["tokens_consumed"] == 150
    assert item["tokens_generated"] == 75
    assert item["response_time_ms"] == Decimal("250.5")
    assert item["confidence_score"] == Decimal("0.92")
    assert item["status"] == "completed"

def test_store_agent_metrics_missing_run_id():
    """Test handling of missing run_id"""
    test_event = {
        "agent_name": "TestAgent",
        # missing run_id
    }

    context = {}
    # Import handler here to ensure mocks are active
    # FIXED: Use absolute import from the project root
    from metrics_lambda.handler import store_agent_metrics

    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 400
    response_body = json.loads(response["body"])
    assert "Missing run_id" in response_body["error"]

def test_store_agent_metrics_string_body():
    """Test handling of string body (JSON) - FIXED"""
    # This test was previously unreliable. Now it properly mocks DynamoDB.
    with mock_dynamodb():
        # Create mocked DynamoDB table
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.create_table(
            TableName="AgentMetrics",
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}, {"AttributeName": "timestamp", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}, {"AttributeName": "timestamp", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST"
        )

        test_event = {
            "body": json.dumps({
                "run_id": "test-run-456",
                "agent_name": "TestAgent",
                "status": "completed"
            })
        }
        context = {}

        # Import handler here to ensure mocks are active
        # FIXED: Use absolute import from the project root
        from metrics_lambda.handler import store_agent_metrics

        response = store_agent_metrics(test_event, context)

        assert response["statusCode"] == 200
        body = json.loads(response["body"])
        assert body["status"] == "ok"
        assert body["run_id"] == "test-run-456"

        # Verify item was written
        items = table.scan()["Items"]
        assert len(items) == 1
        assert items[0]["run_id"] == "test-run-456"

def test_store_agent_metrics_invalid_json():
    """Test handling of invalid JSON in body"""
    test_event = {
        "body": "{invalid json"
    }

    context = {}

    # Import handler here to ensure mocks are active
    # FIXED: Use absolute import from the project root
    from metrics_lambda.handler import store_agent_metrics

    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 400
    response_body = json.loads(response["body"])
    assert "Invalid JSON payload" in response_body["error"]

@mock_dynamodb
def test_store_agent_metrics_with_defaults():
    """Test that missing optional fields get default values"""
    # Mock DynamoDB
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="AgentMetrics",
        KeySchema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"}
        ],
        AttributeDefinitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"}
        ],
        BillingMode="PAY_PER_REQUEST"
    )

    test_event = {
        "run_id": "test-run-789",
        "agent_name": "TestAgent"
        # Missing other fields
    }

    context = {}

    # Import handler here to ensure mocks are active
    # FIXED: Use absolute import from the project root
    from metrics_lambda.handler import store_agent_metrics

    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 200

    # Check defaults were set
    items = table.scan()["Items"]
    assert len(items) == 1
    item = items[0]

    assert item["tokens_consumed"] == 0
    assert item["tokens_generated"] == 0
    assert item["response_time_ms"] == Decimal("0.0")
    assert item["confidence_score"] == Decimal("0.0")
    assert item["status"] == "unknown"

from unittest.mock import patch
@patch("boto3.resource")
def test_store_agent_metrics_exception_handling(mock_boto_resource):
    """Test exception handling in the Lambda - FIXED"""
    # This test was previously unreliable. Now it explicitly mocks a failure.
    
    # Configure the mock to raise an error when put_item is called
    mock_table = mock_boto_resource.return_value.Table.return_value
    mock_table.put_item.side_effect = ClientError(
        {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": "Test Exception"}},
        "PutItem"
    )

    test_event = {
        "run_id": "test-run-fail",
        "agent_name": "TestAgent"
    }
    context = {}

    # FIXED: Use absolute import from the project root
    from metrics_lambda.handler import store_agent_metrics
    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 500
    assert "Internal server error" in json.loads(response["body"])["error"]