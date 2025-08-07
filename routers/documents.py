import logging
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.knowledge_base_service import KnowledgeBaseService
from services.auth_service import AuthService
from modals.document import SimilaritySearchRequest, SimilaritySearchResponse

router = APIRouter(prefix="/documents", tags=["Document Management"])
security = HTTPBearer()
knowledge_service = KnowledgeBaseService()
auth_service = AuthService()

logger = logging.getLogger(__name__)

@router.post("/process/flexible")
async def process_documents_flexible(
    document_type: DocumentType,
    target_namespace: NamespaceType,
    source_folder: Optional[str] = None,
    force_reindex: bool = False,
    include_subfolders: bool = False,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Process documents with flexible namespace assignment
    
    - **document_type**: Type of documents ('knowledge' or 'product')
    - **target_namespace**: Which namespace to store in ('dog-health-knowledge' or 'marshee-products')
    - **source_folder**: Optional custom source folder (defaults to type-based folder)
    - **force_reindex**: Clear existing vectors before processing
    - **include_subfolders**: Process files in subfolders too
    
    Examples:
    - Send product docs to knowledge namespace
    - Send knowledge docs to products namespace
    - Process custom folder to any namespace
    """
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        
        request = ProcessDocumentsRequest(
            document_type=document_type,
            target_namespace=target_namespace,
            source_folder=source_folder,
            force_reindex=force_reindex,
            include_subfolders=include_subfolders
        )
        
        results = knowledge_service.process_documents_flexible(request)
        
        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results["error"]
            )
        
        return {
            "message": f"Flexible processing completed: {document_type.value} → {target_namespace.value}",
            "user_id": current_user.user_id,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Flexible processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Flexible document processing failed"
        )

@router.post("/process/bulk")
async def process_documents_bulk(
    folder_path: str,
    target_namespace: NamespaceType,
    document_type: DocumentType = DocumentType.KNOWLEDGE,
    file_pattern: str = "*.txt",
    force_reindex: bool = False,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Process all files from any folder to any namespace
    
    - **folder_path**: Source folder path (e.g., 'data/all_files', 'data/mixed')
    - **target_namespace**: Target namespace for all files
    - **document_type**: How to categorize the documents
    - **file_pattern**: File pattern to match (default: '*.txt')
    - **force_reindex**: Clear existing vectors before processing
    
    Perfect for:
    - Processing mixed content files
    - Bulk uploading to specific namespace
    - Converting document types
    """
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        
        request = BulkProcessRequest(
            folder_path=folder_path,
            target_namespace=target_namespace,
            document_type=document_type,
            file_pattern=file_pattern,
            force_reindex=force_reindex
        )
        
        results = knowledge_service.process_documents_bulk(request)
        
        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results["error"]
            )
        
        return {
            "message": f"Bulk processing completed: {folder_path} → {target_namespace.value}",
            "user_id": current_user.user_id,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Bulk processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Bulk document processing failed"
        )

@router.get("/folders/info")
async def get_folder_info(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get information about available folders and namespaces
    
    Returns:
    - Available folders and file counts
    - Namespace options
    - Usage examples
    - Current vector database stats
    """
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        folder_info = knowledge_service.get_folder_info()
        
        return {
            "user_id": current_user.user_id,
            "folder_info": folder_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Folder info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve folder information"
        )

@router.post("/process")
async def process_documents(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Process all text files in the data folder and create embeddings
    
    This endpoint will:
    1. Read all .txt files from the data folder
    2. Split them into chunks
    3. Create embeddings using Gemini
    4. Store in vector database
    """
    try:
        # Verify authentication
        current_user = auth_service.get_current_user(credentials.credentials)
        
        # Process documents
        results = knowledge_service.process_and_index_documents()
        
        if "error" in results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=results["error"]
            )
        
        return {
            "message": "Documents processed successfully",
            "user_id": current_user.user_id,
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Document processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document processing failed"
        )

@router.post("/search", response_model=SimilaritySearchResponse)
async def search_documents(
    request: SimilaritySearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Search the knowledge base using semantic similarity
    
    - **query**: Text to search for
    - **k**: Number of results to return (default: 5)
    - **filter_metadata**: Optional metadata filters
    """
    try:
        # Verify authentication
        current_user = auth_service.get_current_user(credentials.credentials)
        
        # Search knowledge base
        results = knowledge_service.search_knowledge_base(request)
        
        logger.info(f"🔍 User {current_user.user_id} searched: '{request.query}' - {results.total_results} results")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Search failed"
        )

@router.get("/status")
async def get_system_status(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get comprehensive system status including file count, database stats, and vector index info"""
    try:
        # Verify authentication
        current_user = auth_service.get_current_user(credentials.credentials)
        
        # Get system status
        status_info = knowledge_service.get_system_status()
        
        return {
            "user_id": current_user.user_id,
            "system_status": status_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Status error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve system status"
        )

@router.get("/health")
async def documents_health_check():
    """Health check for document processing services"""
    return {
        "service": "Document Processing",
        "status": "healthy",
        "components": {
            "document_processor": "ready",
            "embedding_service": "ready",
            "vector_database": "ready",
            "mongodb": "ready"
        }
    }

@router.get("/examples/usage")
async def get_usage_examples():
    """Get examples of how to use the flexible processing features"""
    return {
        "flexible_processing_examples": [
            {
                "scenario": "Send product files to knowledge namespace",
                "endpoint": "/documents/process/flexible",
                "params": {
                    "document_type": "product",
                    "target_namespace": "dog-health-knowledge",
                    "source_folder": "data/products"
                }
            },
            {
                "scenario": "Process custom folder to products namespace",
                "endpoint": "/documents/process/flexible",
                "params": {
                    "document_type": "knowledge",
                    "target_namespace": "marshee-products",
                    "source_folder": "data/custom"
                }
            }
        ],
        "bulk_processing_examples": [
            {
                "scenario": "Process all files from mixed folder to knowledge",
                "endpoint": "/documents/process/bulk",
                "params": {
                    "folder_path": "data/all_files",
                    "target_namespace": "dog-health-knowledge",
                    "document_type": "knowledge"
                }
            },
            {
                "scenario": "Process specific files to products namespace",
                "endpoint": "/documents/process/bulk",
                "params": {
                    "folder_path": "data/mixed",
                    "target_namespace": "marshee-products",
                    "file_pattern": "*product*.txt"
                }
            }
        ],
        "folder_structure_suggestions": {
            "data/all_files/": "Put all your files here and choose namespace dynamically",
            "data/mixed/": "Mixed content that you can separate by namespace",
            "data/custom/": "Custom organized content",
            "data/knowledge/": "Default knowledge files",
            "data/products/": "Default product files"
        },
        "namespace_options": [
            "dog-health-knowledge",
            "marshee-products"
        ]
    }
    
