import os
import time
import uuid
import requests
import json
import base64
import boto3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Load environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
METRICS_LAMBDA_URL = os.getenv("METRICS_LAMBDA_URL")
METRICS_LAMBDA_NAME = os.getenv("METRICS_LAMBDA_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT")
USE_LOCALSTACK = bool(LOCALSTACK_ENDPOINT)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://pdf_service:8000")

# Basic auth for inter-service calls
SERVICE_AUTH = "Basic " + base64.b64encode(b"admin:password").decode()
SERVICE_HEADERS = {"Authorization": SERVICE_AUTH}

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize AWS Lambda client
lambda_kwargs = {"region_name": AWS_REGION}
if USE_LOCALSTACK:
    lambda_kwargs["endpoint_url"] = LOCALSTACK_ENDPOINT
lambda_client = boto3.client("lambda", **lambda_kwargs) if METRICS_LAMBDA_NAME else None

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists and create if needed
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,  # MiniLM dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'  
        )
    )
    # Wait for index to be ready
    time.sleep(1)

# Get index
index = pc.Index(PINECONE_INDEX_NAME)

# Text chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Metrics sender
def send_metrics(metrics_payload: dict):
    try:
        if METRICS_LAMBDA_NAME and lambda_client:
            # Invoke Lambda directly via boto3
            payload_json = json.dumps(metrics_payload)
            response = lambda_client.invoke(
                FunctionName=METRICS_LAMBDA_NAME,
                InvocationType='Event',  # Asynchronous
                Payload=payload_json
            )
            if response['StatusCode'] != 202:
                print(f"Metrics Lambda invocation failed with status: {response['StatusCode']}")
        elif METRICS_LAMBDA_URL:
            # Fallback to HTTP POST
            response = requests.post(METRICS_LAMBDA_URL, json=metrics_payload)
            response.raise_for_status()
        else:
            print("No metrics endpoint configured")
    except Exception as e:
        print(f"Metrics Lambda call failed: {e}")

# Function to generate embeddings
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts).tolist()

# Function to get document text from PDF service
def get_document_text(doc_id: str) -> str:
    try:
        response = requests.get(f"{PDF_SERVICE_URL}/pdf/documents/{doc_id}", headers=SERVICE_HEADERS)
        response.raise_for_status()
        doc_data = response.json()
        return doc_data["extracted_text"]
    except requests.RequestException as e:
        raise Exception(f"Failed to retrieve document {doc_id}: {str(e)}")

# Function to call LLM
def call_llm(prompt: str) -> dict:
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        tokens_consumed = response.usage.prompt_tokens
        tokens_generated = response.usage.completion_tokens
        return {
            "answer": answer,
            "tokens_consumed": tokens_consumed,
            "tokens_generated": tokens_generated
        }
    except Exception as e:
        raise Exception(f"LLM call failed: {str(e)}")