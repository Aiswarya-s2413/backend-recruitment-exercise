import os
import json
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Query
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError
import requests
from datetime import datetime

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT", "http://localstack:4566")
DYNAMODB_TABLE_DOCUMENTS = os.getenv("DYNAMODB_TABLE_DOCUMENTS", "DocumentsMetadata")
S3_BUCKET = os.getenv("S3_BUCKET", "pdf-service-bucket")
RAG_MODULE_URL = os.getenv("RAG_MODULE_URL", "http://rag_module:8001")
METRICS_LAMBDA_URL = os.getenv("METRICS_LAMBDA_URL", "http://metrics_stub:9000/metrics")

# boto3 clients/resources pointing to LocalStack
boto3_kwargs = {"region_name": AWS_REGION}
# Add endpoint to client/resource to point to LocalStack
dynamodb_client = boto3.client("dynamodb", endpoint_url=LOCALSTACK_ENDPOINT, **boto3_kwargs)
dynamodb = boto3.resource("dynamodb", endpoint_url=LOCALSTACK_ENDPOINT, **boto3_kwargs)
s3 = boto3.client("s3", endpoint_url=LOCALSTACK_ENDPOINT, **boto3_kwargs)

app = FastAPI(title="aws_service")

# Pydantic models
class DocumentItem(BaseModel):
    doc_id: str
    filename: str
    tags: Optional[dict] = None
    s3_key: Optional[str] = None

class UpdateDocument(BaseModel):
    tags: Optional[dict] = None
    s3_key: Optional[str] = None

# Ensure DynamoDB table and S3 bucket exist at startup
def ensure_dynamodb_table(table_name: str):
    existing = dynamodb_client.list_tables().get("TableNames", [])
    if table_name not in existing:
        print(f"Creating DynamoDB table: {table_name}")
        dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        # Wait until table exists
        waiter = dynamodb_client.get_waiter('table_exists')
        waiter.wait(TableName=table_name)
        print("Table ready.")

def ensure_s3_bucket(bucket_name: str):
    try:
        s3.head_bucket(Bucket=bucket_name)
    except ClientError:
        print(f"Creating S3 bucket: {bucket_name}")
        s3.create_bucket(Bucket=bucket_name)

@app.on_event("startup")
def startup_event():
    ensure_dynamodb_table(DYNAMODB_TABLE_DOCUMENTS)
    ensure_dynamodb_table(os.getenv("METRICS_TABLE", "AgentMetrics"))
    ensure_s3_bucket(S3_BUCKET)

# CRUD: create metadata
@app.post("/aws/documents", status_code=201)
def create_document(item: DocumentItem = Body(...)):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    now = datetime.utcnow().isoformat()
    doc = {
        "doc_id": item.doc_id,
        "filename": item.filename,
        "upload_timestamp": now,
        "s3_key": item.s3_key if item.s3_key else "",
        "tags": item.tags or {}
    }
    table.put_item(Item=doc)
    return {"status": "ok", "doc_id": item.doc_id}

# Read
@app.get("/aws/documents/{doc_id}")
def get_document(doc_id: str):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    resp = table.get_item(Key={"doc_id": doc_id})
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return item

# Update
@app.put("/aws/documents/{doc_id}")
def update_document(doc_id: str, body: UpdateDocument = Body(...)):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    expression_items = []
    expr_attr_values = {}
    if body.tags is not None:
        expression_items.append("tags = :t")
        expr_attr_values[":t"] = body.tags
    if body.s3_key is not None:
        expression_items.append("s3_key = :s")
        expr_attr_values[":s"] = body.s3_key

    if not expression_items:
        raise HTTPException(status_code=400, detail="No updatable fields provided")

    update_expression = "SET " + ", ".join(expression_items)
    table.update_item(
        Key={"doc_id": doc_id},
        UpdateExpression=update_expression,
        ExpressionAttributeValues=expr_attr_values
    )
    return {"status": "ok", "doc_id": doc_id}

# Delete (with optional S3 delete)
@app.delete("/aws/documents/{doc_id}")
def delete_document(doc_id: str, delete_s3: bool = Query(False)):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    resp = table.get_item(Key={"doc_id": doc_id})
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")

    # Optionally delete S3 object
    if delete_s3 and item.get("s3_key"):
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=item["s3_key"])
        except Exception as e:
            print("S3 delete failed:", e)

    table.delete_item(Key={"doc_id": doc_id})
    return {"status": "deleted", "doc_id": doc_id}

# Trigger RAG indexing for a doc_id
@app.post("/aws/documents/{doc_id}/index")
def index_document(doc_id: str):
    url = f"{RAG_MODULE_URL}/rag/index"
    payload = [doc_id]
    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        return {"status": "ok", "rag_response": resp.json()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Forward query to RAG module
@app.post("/aws/query")
def aws_query(body: Dict[str, Any] = Body(...)):
    url = f"{RAG_MODULE_URL}/rag/query"
    try:
        resp = requests.post(url, json=body, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
