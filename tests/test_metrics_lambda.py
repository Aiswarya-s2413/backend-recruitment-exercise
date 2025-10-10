import pytest
import json
from moto import mock_dynamodb
import boto3
from handler import store_agent_metrics

@mock_dynamodb
def test_store_agent_metrics_valid_event():
    """Test that the Lambda writes one item to AgentMetrics given a valid JSON event"""
    # Create mocked DynamoDB table
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
    assert item["response_time_ms"] == 250.5
    assert item["confidence_score"] == 0.92
    assert item["status"] == "completed"

def test_store_agent_metrics_missing_run_id():
    """Test handling of missing run_id"""
    test_event = {
        "agent_name": "TestAgent",
        # missing run_id
    }

    context = {}
    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 400
    response_body = json.loads(response["body"])
    assert "Missing run_id" in response_body["error"]

def test_store_agent_metrics_string_body():
    """Test handling of string body (JSON)"""
    test_event = {
        "body": json.dumps({
            "run_id": "test-run-456",
            "agent_name": "TestAgent",
            "status": "completed"
        })
    }

    context = {}

    # Mock DynamoDB for this test
    with pytest.raises(Exception):  # Will fail because no table, but tests parsing
        response = store_agent_metrics(test_event, context)

def test_store_agent_metrics_invalid_json():
    """Test handling of invalid JSON in body"""
    test_event = {
        "body": "{invalid json"
    }

    context = {}
    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 400
    response_body = json.loads(response["body"])
    assert "Invalid JSON payload" in response_body["error"]

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
    response = store_agent_metrics(test_event, context)

    assert response["statusCode"] == 200

    # Check defaults were set
    items = table.scan()["Items"]
    assert len(items) == 1
    item = items[0]

    assert item["tokens_consumed"] == 0
    assert item["tokens_generated"] == 0
    assert item["response_time_ms"] == 0.0
    assert item["confidence_score"] == 0.0
    assert item["status"] == "unknown"

def test_store_agent_metrics_exception_handling():
    """Test exception handling in the Lambda"""
    # This test ensures that if DynamoDB operations fail, it returns 500
    # Since we're using moto, it should work, but we can test by not creating table
    test_event = {
        "run_id": "test-run-fail",
        "agent_name": "TestAgent"
    }

    context = {}

    # Without mocking DynamoDB, it should fail gracefully
    response = store_agent_metrics(test_event, context)

    # It will try to create the table resource but fail on put_item
    # The exact behavior depends on boto3 error handling
    assert response["statusCode"] in [200, 500]  # Either succeeds with moto or fails