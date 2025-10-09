import os
import json
from datetime import datetime
import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566")
METRICS_TABLE = os.getenv("METRICS_TABLE", "AgentMetrics")

dynamodb = boto3.resource("dynamodb", endpoint_url=LOCALSTACK_ENDPOINT, region_name=AWS_REGION)

def lambda_handler(event, context):
    # event can be dict or may contain "body" (string) depending on invocation method
    payload = event.get("body") if isinstance(event, dict) and "body" in event else event
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except:
            payload = {}

    run_id = payload.get("run_id")
    timestamp = datetime.utcnow().isoformat()
    item = {
        "run_id": run_id,
        "timestamp": timestamp,
        "agent_name": payload.get("agent_name"),
        "tokens_consumed": payload.get("tokens_consumed"),
        "tokens_generated": payload.get("tokens_generated"),
        "response_time_ms": payload.get("response_time_ms"),
        "confidence_score": payload.get("confidence_score"),
        "status": payload.get("status")
    }

    table = dynamodb.Table(METRICS_TABLE)
    table.put_item(Item=item)
    return {"statusCode": 200, "body": json.dumps({"status": "ok", "run_id": run_id})}
