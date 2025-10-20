from pydantic import BaseModel, Field, field_validator
from typing import List
import uuid

class IndexRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1, max_length=100)

    @field_validator('document_ids')
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        if len(v) > 100:
            raise ValueError("Cannot index more than 100 documents at once")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("document_ids must be unique")
        # Validate format (basic check)
        for doc_id in v:
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"Invalid document ID: {doc_id}")
        return v

class QueryRequest(BaseModel):
    document_ids: List[str] = Field(..., min_length=1, max_length=50)
    question: str = Field(..., min_length=1, max_length=1000)

    @field_validator('document_ids')
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("document_ids cannot be empty")
        if len(v) > 50:
            raise ValueError("Cannot query more than 50 documents at once")
        # Check for duplicates
        if len(v) != len(set(v)):
            raise ValueError("document_ids must be unique")
        # Validate format
        for doc_id in v:
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError(f"Invalid document ID: {doc_id}")
        return v

    @field_validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty")
        return v.strip()

class QueryResponse(BaseModel):
    run_id: str
    answer: str
    tokens_consumed: int = Field(..., ge=0)
    tokens_generated: int = Field(..., ge=0)
    response_time_ms: float = Field(..., ge=0)
    confidence_score: float = Field(..., ge=0.0, le=1.0)

class ErrorResponse(BaseModel):
    error_code: str
    message: str
    details: dict = None
