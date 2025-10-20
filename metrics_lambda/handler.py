import os
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
import boto3

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT")
USE_LOCALSTACK = bool(LOCALSTACK_ENDPOINT)
METRICS_TABLE = os.getenv("METRICS_TABLE", "AgentMetrics")

boto3_kwargs = {"region_name": AWS_REGION}
if USE_LOCALSTACK:
    boto3_kwargs["endpoint_url"] = LOCALSTACK_ENDPOINT

dynamodb = boto3.resource("dynamodb", **boto3_kwargs)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def store_agent_metrics(event, context):
    try:
        logger.info(f"Received metrics event: {event}")

        # event can be dict or may contain "body" (string) depending on invocation method
        payload = event.get("body") if isinstance(event, dict) and "body" in event else event
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse payload: {e}")
                return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON payload"})}

        run_id = payload.get("run_id")
        if not run_id:
            logger.warning("Missing run_id in payload")
            return {"statusCode": 400, "body": json.dumps({"error": "Missing run_id"})}

        timestamp = datetime.now(timezone.utc).isoformat()
        item = {
            "run_id": run_id,
            "timestamp": timestamp,
            "agent_name": payload.get("agent_name"),
            "tokens_consumed": payload.get("tokens_consumed", 0),
            "tokens_generated": payload.get("tokens_generated", 0),
            "response_time_ms": Decimal(str(payload.get("response_time_ms", 0.0))),
            "confidence_score": Decimal(str(payload.get("confidence_score", 0.0))),
            "status": payload.get("status", "unknown")
        }

        table = dynamodb.Table(METRICS_TABLE)
        table.put_item(Item=item)
        logger.info(f"Successfully stored metrics for run_id: {run_id}")
        return {"statusCode": 200, "body": json.dumps({"status": "ok", "run_id": run_id})}

    except Exception as e:
        logger.error(f"Error processing metrics: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": "Internal server error"})}
