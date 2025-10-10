from fastapi import HTTPException
from typing import Any, Dict, Optional

class RAGException(HTTPException):
    """Base exception for RAG module errors"""

    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = None,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code or f"RAG_{status_code}"

class AuthenticationError(RAGException):
    """Authentication related errors"""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail, error_code="AUTH_ERROR")

class AuthorizationError(RAGException):
    """Authorization related errors"""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=403, detail=detail, error_code="AUTHZ_ERROR")

class ValidationError(RAGException):
    """Input validation errors"""

    def __init__(self, detail: str = "Invalid input data"):
        super().__init__(status_code=400, detail=detail, error_code="VALIDATION_ERROR")

class DocumentNotFoundError(RAGException):
    """Document not found errors"""

    def __init__(self, doc_id: str):
        super().__init__(
            status_code=404,
            detail=f"Document '{doc_id}' not found",
            error_code="DOCUMENT_NOT_FOUND"
        )

class EmbeddingError(RAGException):
    """Embedding generation errors"""

    def __init__(self, detail: str = "Failed to generate embeddings"):
        super().__init__(status_code=500, detail=detail, error_code="EMBEDDING_ERROR")

class IndexError(RAGException):
    """Vector index operation errors"""

    def __init__(self, detail: str = "Vector index operation failed"):
        super().__init__(status_code=500, detail=detail, error_code="INDEX_ERROR")

class LLMError(RAGException):
    """LLM API errors"""

    def __init__(self, detail: str = "LLM service error"):
        super().__init__(status_code=500, detail=detail, error_code="LLM_ERROR")

class MetricsError(RAGException):
    """Metrics collection errors"""

    def __init__(self, detail: str = "Metrics collection failed"):
        super().__init__(status_code=500, detail=detail, error_code="METRICS_ERROR")