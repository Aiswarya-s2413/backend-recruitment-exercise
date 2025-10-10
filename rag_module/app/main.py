from dotenv import load_dotenv

load_dotenv()
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from app import schemas, utils, auth, exceptions
from app.logger import get_logger
import uuid
import time

app = FastAPI(title="RAG Module")

# Logger
logger = get_logger(__name__)

# Global exception handlers
@app.exception_handler(exceptions.RAGException)
async def rag_exception_handler(request: Request, exc: exceptions.RAGException):
    logger.error(f"RAG Exception: {exc.error_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "message": exc.detail,
            "details": getattr(exc, 'details', None)
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": f"HTTP_{exc.status_code}",
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred"
        }
    )





@app.post("/rag/index")
def index_documents(request: schemas.IndexRequest):
    logger.info(f"Indexing documents: {request.document_ids}")
    results = []

    try:
        for doc_id in request.document_ids:
            try:
                text = utils.get_document_text(doc_id)
                logger.info(f"Processing document: {doc_id}")

                chunks = utils.text_splitter.split_text(text)
                logger.info(f"Split document {doc_id} into {len(chunks)} chunks")

                embeddings = utils.embed_texts(chunks)
                logger.info(f"Generated embeddings for {len(embeddings)} chunks")

                # Upsert to Pinecone
                to_upsert = [(f"{doc_id}_{i}", emb, {"doc_id": doc_id, "text": chunk}) for i, (emb, chunk) in enumerate(zip(embeddings, chunks))]
                utils.index.upsert(vectors=to_upsert)
                logger.info(f"Successfully indexed document: {doc_id}")
                results.append({"doc_id": doc_id, "status": "success"})
            except Exception as e:
                logger.error(f"Error indexing document {doc_id}: {str(e)}")
                results.append({"doc_id": doc_id, "status": "failure", "reason": str(e)})
                continue
        logger.info(f"Indexing completed. Success: {len([r for r in results if r["status"] == "success"])}, Failures: {len([r for r in results if r["status"] == "failure"])}")
        return {"results": results}

    except Exception as e:
        logger.error(f"Unexpected error during indexing: {str(e)}")
        raise exceptions.IndexError("Failed to index documents")

@app.post("/rag/query", response_model=schemas.QueryResponse)
def query_rag(request: schemas.QueryRequest):
    logger.info(f"RAG query request: {request.question[:50]}...")
    start_time = time.time()
    run_id = str(uuid.uuid4())
    
    try:
        # Embed question
        logger.info("Generating question embedding")
        question_embedding = utils.embed_texts([request.question])[0]
        
        # Query Pinecone
        logger.info(f"Querying vector database with top_k={utils.TOP_K}")
        query_results = utils.index.query(
            vector=question_embedding,
            top_k=utils.TOP_K,
            filter={"doc_id": {"$in": request.document_ids}},
            include_metadata=True
        )
        
        # Combine context
        context_text = "\n".join([match['metadata']['text'] for match in query_results['matches']])
        logger.info(f"Retrieved {len(query_results['matches'])} relevant chunks")
        
        # Create prompt
        prompt = f"Based on the following context, please answer the question.\n\nContext:\n{context_text}\n\nQuestion:\n{request.question}\n\nAnswer:"
        
        # Call LLM API
        logger.info("Calling LLM API")
        llm_response = utils.call_llm(prompt)
        answer = llm_response["answer"]
        tokens_consumed = llm_response["tokens_consumed"]
        tokens_generated = llm_response["tokens_generated"]
        confidence_score = sum(match['score'] for match in query_results['matches']) / len(query_results['matches']) if query_results['matches'] else 0.0
        
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
        
        logger.info(f"Query completed successfully in {response_time_ms:.2f}ms")
        return schemas.QueryResponse(
            run_id=run_id,
            answer=answer,
            tokens_consumed=tokens_consumed,
            tokens_generated=tokens_generated,
            response_time_ms=response_time_ms,
            confidence_score=confidence_score
        )
    
    except Exception as e:
        logger.error(f"Error during RAG query: {str(e)}")
        # Send failure metrics
        try:
            failure_metrics = {
                "run_id": run_id,
                "agent_name": "RAGQueryAgent",
                "tokens_consumed": 0,
                "tokens_generated": 0,
                "response_time_ms": (time.time() - start_time) * 1000,
                "confidence_score": 0.0,
                "status": "failed"
            }
            utils.send_metrics(failure_metrics)
        except Exception as metrics_error:
            logger.error(f"Failed to send failure metrics: {str(metrics_error)}")
        
        raise exceptions.LLMError("Failed to process RAG query")
