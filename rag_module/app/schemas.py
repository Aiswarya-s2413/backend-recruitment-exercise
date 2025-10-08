from pydantic import BaseModel
from typing import List

class IndexRequest(BaseModel):
    document_ids: List[str]

class QueryRequest(BaseModel):
    document_ids: List[str]
    question: str

class QueryResponse(BaseModel):
    run_id: str
    answer: str
    tokens_consumed: int
    tokens_generated: int
    response_time_ms: float
    confidence_score: float
