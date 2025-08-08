from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class DocumentType(str, Enum):
    KNOWLEDGE = "knowledge"
    PRODUCT = "product"

class NamespaceType(str, Enum):
    KNOWLEDGE = "dog-health-knowledge"
    PRODUCTS = "marshee-products"

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    document_type: DocumentType
    namespace: str
    original_namespace: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentCreate(BaseModel):
    filename: str
    content: str
    document_type: DocumentType
    namespace: str
    original_namespace: str
    metadata: Optional[Dict[str, Any]] = {}

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    document_type: str
    namespace: str
    original_namespace: str
    content_length: int
    chunk_count: Optional[int] = None
    metadata: Dict[str, Any]
    created_at: datetime
    indexed_at: Optional[datetime] = None

class EmbeddingRequest(BaseModel):
    text: str
    model: str = "models/embedding-001"
    output_dimensionality: int = 768

class SimilaritySearchRequest(BaseModel):
    query: str
    k: int = 5
    filter_metadata: Optional[Dict[str, Any]] = {}

class SimilaritySearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total_results: int