import os
import time
import uuid
import requests
from sentence_transformers import SentenceTransformer
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))
METRICS_LAMBDA_URL = os.getenv("METRICS_LAMBDA_URL")

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(PINECONE_INDEX_NAME, dimension=384)  # MiniLM dimension
index = pinecone.Index(PINECONE_INDEX_NAME)

# Text chunker
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Metrics sender
def send_metrics(metrics_payload: dict):
    try:
        response = requests.post(METRICS_LAMBDA_URL, json=metrics_payload)
        response.raise_for_status()
    except Exception as e:
        print(f"Metrics Lambda call failed: {e}")

# Function to generate embeddings
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_model.encode(texts).tolist()
