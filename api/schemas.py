from typing import Any, Optional
from pydantic import BaseModel


class IngestUploadResponse(BaseModel):
    """Returned after a user uploads + ingests a PDF."""
    status: str
    doc_id: str
    filename: str
    saved_to: str
    collection_name: str
    force: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    latency: float
    retrieval_confidence: Optional[float] = None
    validation_score: Optional[float] = None
    route: Optional[str] = None
    chunks: Optional[list] = None
    planner_decision: Optional[dict] = None
    validation_result: Optional[dict] = None
    retrieval_analysis: Optional[list] = None
    retrieved_image_path: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
