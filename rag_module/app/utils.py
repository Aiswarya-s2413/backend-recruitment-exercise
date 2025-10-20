import os
import time
import logging
import uuid
import requests
import json
import base64
import boto3
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient

from . import auth
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)

PDF_SERVICE_USER = os.getenv("PDF_SERVICE_USER", "admin")
PDF_SERVICE_PASSWORD = os.getenv("PDF_SERVICE_PASSWORD", "password")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
METRICS_LAMBDA_URL = os.getenv("METRICS_LAMBDA_URL")
METRICS_LAMBDA_NAME = os.getenv("METRICS_LAMBDA_NAME")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
LOCALSTACK_ENDPOINT = os.getenv("LOCALSTACK_ENDPOINT")
USE_LOCALSTACK = bool(LOCALSTACK_ENDPOINT)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
if not HUGGINGFACE_API_KEY:
    raise ValueError("HUGGINGFACE_API_KEY environment variable not set. Please provide a valid Hugging Face API key.")

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://pdf_service:8000")
SERVICE_TOKEN = os.getenv("SERVICE_TOKEN")

# LLM model to use - Must support chat completion
# Available free models that support chat completion:
LLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"  # Fast, good for Q&A
# Alternatives:
# LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
# LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# LLM_MODEL = "google/gemma-2-2b-it"

# Basic auth for inter-service calls
SERVICE_HEADERS = {"Authorization": f"Bearer {SERVICE_TOKEN}"} if SERVICE_TOKEN else {}

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Hugging Face Inference Client
hf_client = InferenceClient(token=HUGGINGFACE_API_KEY)

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
    url = f"{PDF_SERVICE_URL}/pdf/documents/{doc_id}"
    resp = requests.get(url, auth=HTTPBasicAuth(PDF_SERVICE_USER, PDF_SERVICE_PASSWORD))
    resp.raise_for_status()
    return resp.json()["extracted_text"]

# Function to call LLM - FIXED: Using chat_completion instead of text_generation
def call_llm(prompt: str) -> dict:
    try:
        # Use chat_completion API (the new standard as of July 2025)
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        response = hf_client.chat_completion(
            messages=messages,
            model=LLM_MODEL,
            max_tokens=500,
            temperature=0.7,
        )
        
        # Extract the answer from the response
        answer = response.choices[0].message.content
        
        # The Inference API does not return token counts in the free tier, so we'll use 0
        return {
            "answer": answer,
            "tokens_consumed": 0,
            "tokens_generated": 0
        }
    except Exception as e:
        logger.error(f"LLM call failed: {e}", exc_info=True)
        raise