import os
from fastapi import FastAPI, Request
from datetime import datetime
import boto3

app = FastAPI(title="metrics_stub")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566")
METRICS_TABLE = os.getenv("METRICS_TABLE", "AgentMetrics")

dynamodb = boto3.resource("dynamodb", endpoint_url=LOCALSTACK_ENDPOINT, region_name=AWS_REGION)

@app.post("/metrics")
async def receive_metrics(req: Request):
    payload = await req.json()
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
    return {"status": "ok", "run_id": run_id}
