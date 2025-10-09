from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException
from app import schemas, utils
import uuid

app = FastAPI(title="RAG Module")

# For demo, assume we have a local document storage
# Replace this with real DB or S3 retrieval
DOCUMENTS = {
    "doc1": "This is the first PDF text...",
    "doc2": "This is the second PDF text..."
}

@app.post("/rag/index")
def index_documents(request: schemas.IndexRequest):
    results = []
    for doc_id in request.document_ids:
        if doc_id not in DOCUMENTS:
            results.append({"doc_id": doc_id, "status": "failure", "reason": "Document not found"})
            continue
        
        text = DOCUMENTS[doc_id]
        chunks = utils.text_splitter.split_text(text)
        embeddings = utils.embed_texts(chunks)
        
        # Upsert to Pinecone
        to_upsert = [(f"{doc_id}_{i}", emb, {"doc_id": doc_id, "text": chunk}) for i, (emb, chunk) in enumerate(zip(embeddings, chunks))]
        utils.index.upsert(vectors=to_upsert)
        results.append({"doc_id": doc_id, "status": "success"})
    
    return {"results": results}

@app.post("/rag/query", response_model=schemas.QueryResponse)
def query_rag(request: schemas.QueryRequest):
    start_time = time.time()
    run_id = str(uuid.uuid4())
    
    # Embed question
    question_embedding = utils.embed_texts([request.question])[0]
    
    # Query Pinecone
    query_results = utils.index.query(
        vector=question_embedding,
        top_k=utils.TOP_K,
        filter={"doc_id": {"$in": request.document_ids}},
        include_metadata=True
    )
    
    # Combine context
    context_text = "\n".join([match['metadata']['text'] for match in query_results['matches']])
    
    # Create prompt
    prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context_text}\n\nQuestion:\n{request.question}\n\nAnswer:"
    
    # Call LLM API (pseudo-code, replace with real API)
    answer = f"Simulated answer to: {request.question}"
    tokens_consumed = 50  # placeholder
    tokens_generated = 25  # placeholder
    confidence_score = 1.0
    
    response_time_ms = (time.time() - start_time) * 1000
    
    # Send metrics
    metrics_payload = {
        "run_id": run_id,
        "agent_name": "RAGQueryAgent",
        "tokens_consumed": tokens_consumed,
        "tokens_generated": tokens_generated,
        "response_time_ms": response_time_ms,
        "confidence_score": confidence_score,
        "status": "completed"
    }
    utils.send_metrics(metrics_payload)
    
    return schemas.QueryResponse(
        run_id=run_id,
        answer=answer,
        tokens_consumed=tokens_consumed,
        tokens_generated=tokens_generated,
        response_time_ms=response_time_ms,
        confidence_score=confidence_score
    )
