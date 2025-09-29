"""
Pydantic models for the Knowledge Base Search & Enrichment system.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"

class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class EnrichmentType(str, Enum):
    DOCUMENT = "document"
    WEB_SEARCH = "web_search"
    API_FETCH = "api_fetch"
    USER_INPUT = "user_input"

class DocumentInfo(BaseModel):
    """Information about an uploaded document."""
    id: str
    filename: str
    content_type: str
    document_type: DocumentType
    upload_date: datetime
    size_bytes: int
    chunk_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SearchRequest(BaseModel):
    """Request for searching documents."""
    query: str = Field(..., description="Natural language search query")
    include_enrichment: bool = Field(default=True, description="Include enrichment suggestions")
    max_results: int = Field(default=5, description="Maximum number of relevant documents to retrieve")

class MissingInfo(BaseModel):
    """Information about what's missing from an answer."""
    topic: str = Field(..., description="The topic or aspect that's missing")
    description: str = Field(..., description="Description of what information is needed")
    importance: str = Field(..., description="How important this information is (high/medium/low)")

class EnrichmentSuggestion(BaseModel):
    """Suggestion for enriching the knowledge base."""
    type: EnrichmentType
    description: str
    missing_info: List["MissingInfo"]
    suggested_actions: List[str]
    confidence: float = Field(..., ge=0.0, le=1.0)
    auto_enrichable: bool = Field(default=False)

class SearchResponse(BaseModel):
    """Response from document search with AI-generated answer."""
    query: str
    answer: str
    confidence: ConfidenceLevel
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    sources: List["DocumentInfo"]
    missing_info: List["MissingInfo"]
    enrichment_suggestions: List["EnrichmentSuggestion"]
    processing_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.now)

class RatingRequest(BaseModel):
    """Request for rating an answer."""
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (poor) to 5 (excellent)")
    feedback: Optional[str] = Field(None, description="Optional feedback text")

class EnrichmentResult(BaseModel):
    """Result of an enrichment operation."""
    success: bool
    new_documents: List[DocumentInfo]
    enriched_topics: List[str]
    sources: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)
