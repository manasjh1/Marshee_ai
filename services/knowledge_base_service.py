import logging
from typing import List, Dict, Any, Optional
from services.document_service import DocumentProcessor
from services.embedding_service import GeminiEmbeddingService
from services.vector_db_service import PineconeVectorDB
from repositories.document_repository import DocumentRepository
from modals.document import (
    DocumentType, NamespaceType, SimilaritySearchRequest, SimilaritySearchResponse,
    ProcessDocumentsRequest, BulkProcessRequest
)

logger = logging.getLogger(__name__)

class KnowledgeBaseService:
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.embedding_service = GeminiEmbeddingService()
        self.vector_db = PineconeVectorDB()
        self.document_repo = DocumentRepository()

    def process_documents_flexible(self, request: ProcessDocumentsRequest) -> Dict[str, Any]:
        """Process documents with flexible namespace assignment"""
        try:
            logger.info(f"🚀 Starting flexible document processing")
            logger.info(f"📄 Type: {request.document_type.value}")
            logger.info(f"🎯 Target namespace: {request.target_namespace.value}")
            
            # Read text files with flexible namespace
            documents = self.document_processor.read_text_files_flexible(request)
            if not documents:
                return {
                    "error": f"No documents found to process",
                    "request": {
                        "document_type": request.document_type.value,
                        "target_namespace": request.target_namespace.value,
                        "source_folder": request.source_folder
                    }
                }
            
            # Clear namespace if force reindex
            if request.force_reindex:
                namespace = documents[0].namespace
                logger.info(f"🔄 Force reindex: clearing namespace '{namespace}'")
                self.vector_db.delete_namespace(namespace)
            
            return self._process_document_list(documents, request)
            
        except Exception as e:
            logger.error(f"❌ Flexible processing error: {e}")
            return {"error": str(e)}

    def process_documents_bulk(self, request: BulkProcessRequest) -> Dict[str, Any]:
        """Process documents in bulk from any folder to any namespace"""
        try:
            logger.info(f"🔄 Starting bulk document processing")
            logger.info(f"📁 Source: {request.folder_path}")
            logger.info(f"🎯 Target namespace: {request.target_namespace.value}")
            logger.info(f"📝 Pattern: {request.file_pattern}")
            
            # Read text files in bulk
            documents = self.document_processor.read_text_files_bulk(request)
            if not documents:
                return {
                    "error": f"No documents found matching pattern '{request.file_pattern}' in {request.folder_path}",
                    "request": {
                        "folder_path": request.folder_path,
                        "target_namespace": request.target_namespace.value,
                        "file_pattern": request.file_pattern
                    }
                }
            
            # Clear namespace if force reindex
            if request.force_reindex:
                namespace = documents[0].namespace
                logger.info(f"🔄 Force reindex: clearing namespace '{namespace}'")
                self.vector_db.delete_namespace(namespace)
            
            # Convert to ProcessDocumentsRequest for compatibility
            process_request = ProcessDocumentsRequest(
                document_type=request.document_type,
                target_namespace=request.target_namespace,
                force_reindex=False  # Already handled above
            )
            
            return self._process_document_list(documents, process_request)
            
        except Exception as e:
            logger.error(f"❌ Bulk processing error: {e}")
            return {"error": str(e)}

    def _process_document_list(
        self, 
        documents: List, 
        request: ProcessDocumentsRequest
    ) -> Dict[str, Any]:
        """Common processing logic for document lists"""
        results = {
            "request_info": {
                "document_type": request.document_type.value,
                "target_namespace": request.target_namespace.value,
                "source_folder": getattr(request, 'source_folder', 'default'),
                "force_reindex": request.force_reindex
            },
            "processing_results": {
                "namespace": documents[0].namespace if documents else None,
                "processed_documents": 0,
                "total_chunks": 0,
                "embedded_chunks": 0,
                "indexed_chunks": 0,
                "files": []
            }
        }
        
        for document in documents:
            try:
                # Save document metadata
                document_id = self.document_repo.save_document(document)
                
                # Create chunks
                chunks = self.document_processor.create_chunks(document)
                if not chunks:
                    continue
                
                # Create embeddings
                embedded_chunks = self.embedding_service.embed_document_chunks(chunks)
                if not embedded_chunks:
                    continue
                
                # Save to MongoDB
                saved = self.document_repo.save_chunks(embedded_chunks)
                
                # Upload to vector database
                indexed = self.vector_db.upsert_chunks(
                    embedded_chunks, 
                    namespace=document.namespace
                )
                
                # Update results
                results["processing_results"]["processed_documents"] += 1
                results["processing_results"]["total_chunks"] += len(chunks)
                results["processing_results"]["embedded_chunks"] += len(embedded_chunks)
                if indexed:
                    results["processing_results"]["indexed_chunks"] += len(embedded_chunks)
                
                results["processing_results"]["files"].append({
                    "filename": document.filename,
                    "document_id": document_id,
                    "document_type": document.document_type.value,
                    "target_namespace": document.namespace,
                    "original_namespace": document.original_namespace,
                    "namespace_override": document.namespace != document.original_namespace,
                    "chunks": len(chunks),
                    "embedded": len(embedded_chunks),
                    "indexed": indexed
                })
                
                logger.info(f"✅ Processed: {document.filename} → {document.namespace}")
                
            except Exception as e:
                logger.error(f"❌ Error processing {document.filename}: {e}")
                continue
        
        logger.info(f"🎉 Flexible processing complete: {results['processing_results']['processed_documents']} documents")
        return results

    def get_folder_info(self) -> Dict[str, Any]:
        """Get information about available folders and namespaces"""
        try:
            folder_info = self.document_processor.get_available_folders()
            vector_stats = self.vector_db.get_namespace_stats()
            
            return {
                "folders": folder_info,
                "vector_stats": vector_stats,
                "usage_guide": {
                    "flexible_processing": "Use /process/flexible to choose namespace for any document type",
                    "bulk_processing": "Use /process/bulk to process entire folders to any namespace",
                    "namespace_options": list(folder_info.get("available_namespaces", [])),
                    "example_workflows": [
                        "Put all files in data/all_files/ and send to knowledge namespace",
                        "Process products from data/mixed/ to products namespace",
                        "Override default namespace assignment for any document type"
                    ]
                }
            }
            
        except Exception as e:
            return {"error": str(e)}