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

class ProcessDocumentsRequest(BaseModel):
    document_type: DocumentType
    target_namespace: NamespaceType  # New field for namespace selection
    source_folder: Optional[str] = None  # Optional custom folder path
    force_reindex: bool = False
    include_subfolders: bool = False

class BulkProcessRequest(BaseModel):
    folder_path: str
    target_namespace: NamespaceType  # Choose where to send the files
    document_type: DocumentType = DocumentType.KNOWLEDGE  # Default type
    force_reindex: bool = False
    file_pattern: str = "*.txt"  # Pattern to match files

class DocumentChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    document_type: DocumentType
    namespace: str  # Actual namespace where it will be stored
    original_namespace: str  # Original intended namespace
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DocumentCreate(BaseModel):
    filename: str
    content: str
    document_type: DocumentType
    namespace: str
    original_namespace: str  # Track original vs selected namespace
    metadata: Optional[Dict[str, Any]] = {}

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
