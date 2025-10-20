import os
import json
import time
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Body, Query, Depends
from pydantic import BaseModel
import boto3
from botocore.exceptions import ClientError
import requests
from datetime import datetime, timezone
from . import auth

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT")
USE_LOCALSTACK = bool(LOCALSTACK_ENDPOINT)
DYNAMODB_TABLE_DOCUMENTS = os.getenv("DYNAMODB_TABLE_DOCUMENTS", "DocumentsMetadata")
S3_BUCKET = os.getenv("S3_BUCKET", "pdf-service-bucket")
RAG_MODULE_URL = os.getenv("RAG_MODULE_URL", "http://rag_module:8001")
METRICS_LAMBDA_URL = os.getenv("METRICS_LAMBDA_URL", "http://metrics_stub:9000/metrics")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")

# Headers for inter-service communication
# FIXED: Ensure headers are always set, even if SERVICE_TOKEN is None
def get_service_headers():
    """Get headers for inter-service communication"""
    if SERVICE_TOKEN:
        return {"Authorization": f"Bearer {SERVICE_TOKEN}"}
    return {}

# boto3 clients/resources
boto3_kwargs = {"region_name": AWS_REGION}
if USE_LOCALSTACK:
    boto3_kwargs["endpoint_url"] = LOCALSTACK_ENDPOINT

dynamodb_client = boto3.client("dynamodb", **boto3_kwargs)
dynamodb = boto3.resource("dynamodb", **boto3_kwargs)
s3 = boto3.client("s3", **boto3_kwargs)

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
def ensure_dynamodb_table(table_name: str, key_schema=None, attribute_definitions=None):
    if key_schema is None:
        key_schema = [{"AttributeName": "doc_id", "KeyType": "HASH"}]
    if attribute_definitions is None:
        attribute_definitions = [{"AttributeName": "doc_id", "AttributeType": "S"}]
    existing = dynamodb_client.list_tables().get("TableNames", [])
    if table_name not in existing:
        print(f"Creating DynamoDB table: {table_name}")
        dynamodb_client.create_table(
            TableName=table_name,
            KeySchema=key_schema,
            AttributeDefinitions=attribute_definitions,
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
    # ADDED: Log SERVICE_TOKEN status for debugging
    if SERVICE_TOKEN:
        print(f"SERVICE_TOKEN is configured (length: {len(SERVICE_TOKEN)})")
    else:
        print("WARNING: SERVICE_TOKEN is not set - inter-service calls may fail if RAG module requires auth")
    
    ensure_dynamodb_table(DYNAMODB_TABLE_DOCUMENTS)
    ensure_dynamodb_table(
        os.getenv("METRICS_TABLE", "AgentMetrics"),
        key_schema=[
            {"AttributeName": "run_id", "KeyType": "HASH"},
            {"AttributeName": "timestamp", "KeyType": "RANGE"}
        ],
        attribute_definitions=[
            {"AttributeName": "run_id", "AttributeType": "S"},
            {"AttributeName": "timestamp", "AttributeType": "S"}
        ]
    )
    ensure_s3_bucket(S3_BUCKET)

# CRUD: create metadata
@app.post("/aws/documents", status_code=201)
def create_document(item: DocumentItem = Body(...), current_user: str = Depends(auth.verify_token)):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    now = datetime.now(timezone.utc).isoformat()
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
def get_document(doc_id: str, current_user: str = Depends(auth.verify_token)):
    table = dynamodb.Table(DYNAMODB_TABLE_DOCUMENTS)
    resp = table.get_item(Key={"doc_id": doc_id})
    item = resp.get("Item")
    if not item:
        raise HTTPException(status_code=404, detail="Document not found")
    return item

# Update
@app.put("/aws/documents/{doc_id}")
def update_document(doc_id: str, body: UpdateDocument = Body(...), current_user: str = Depends(auth.verify_token)):
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
def delete_document(doc_id: str, delete_s3: bool = Query(False), current_user: str = Depends(auth.verify_token)):
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
def index_document(doc_id: str, current_user: str = Depends(auth.verify_token)):
    url = f"{RAG_MODULE_URL}/rag/index"
    payload = {"document_ids": [doc_id]}
    try:
        # FIXED: Use function to get headers dynamically
        headers = get_service_headers()
        print(f"Calling RAG module at {url} with headers: {list(headers.keys())}")
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return {"status": "ok", "rag_response": resp.json()}
    except requests.exceptions.HTTPError as e:
        print(f"RAG module returned error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"RAG module error: {e.response.text}")
    except Exception as e:
        print(f"Error calling RAG module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Forward query to RAG module
@app.post("/aws/query")
def aws_query(body: Dict[str, Any] = Body(...), current_user: str = Depends(auth.verify_token)):
    url = f"{RAG_MODULE_URL}/rag/query"
    try:
        # FIXED: Use function to get headers dynamically
        headers = get_service_headers()
        resp = requests.post(url, json=body, headers=headers, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        print(f"RAG module returned error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"RAG module error: {e.response.text}")
    except Exception as e:
        print(f"Error calling RAG module: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))