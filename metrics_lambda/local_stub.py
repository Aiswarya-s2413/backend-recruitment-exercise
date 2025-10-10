import os
import logging
from fastapi import FastAPI, Request, HTTPException
from datetime import datetime
import boto3

app = FastAPI(title="metrics_stub")

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566")
METRICS_TABLE = os.getenv("METRICS_TABLE", "AgentMetrics")

dynamodb = boto3.resource("dynamodb", endpoint_url=LOCALSTACK_ENDPOINT, region_name=AWS_REGION)

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

@app.post("/metrics")
async def receive_metrics(req: Request):
    try:
        payload = await req.json()
        logger.info(f"Received metrics payload: {payload}")

        run_id = payload.get("run_id")
        if not run_id:
            logger.warning("Missing run_id in payload")
            raise HTTPException(status_code=400, detail="Missing run_id")

        timestamp = datetime.utcnow().isoformat()
        item = {
            "run_id": run_id,
            "timestamp": timestamp,
            "agent_name": payload.get("agent_name"),
            "tokens_consumed": payload.get("tokens_consumed", 0),
            "tokens_generated": payload.get("tokens_generated", 0),
            "response_time_ms": payload.get("response_time_ms", 0.0),
            "confidence_score": payload.get("confidence_score", 0.0),
            "status": payload.get("status", "unknown")
        }
        table = dynamodb.Table(METRICS_TABLE)
        table.put_item(Item=item)
        logger.info(f"Successfully stored metrics for run_id: {run_id}")
        return {"status": "ok", "run_id": run_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
