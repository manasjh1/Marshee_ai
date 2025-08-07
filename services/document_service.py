import os
import re
import logging
from typing import List, Dict, Any, Optional  # Add Optional here
from datetime import datetime
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Add these missing imports
from modals.document import (
    DocumentChunk, 
    DocumentCreate, 
    DocumentType,           # Add this
    NamespaceType,         # Add this
    ProcessDocumentsRequest,  # Add this (note: correct spelling)
    BulkProcessRequest     # Add this
)

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        # Default folders
        self.knowledge_folder = os.getenv("KNOWLEDGE_DATA_FOLDER", "data/knowledge")
        self.products_folder = os.getenv("PRODUCTS_DATA_FOLDER", "data/products")
        
        # Namespace mapping
        self.namespace_map = {
            NamespaceType.KNOWLEDGE: os.getenv("KNOWLEDGE_NAMESPACE", "dog-health-knowledge"),
            NamespaceType.PRODUCTS: os.getenv("PRODUCTS_NAMESPACE", "marshee-products")
        }
        
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create default folders
        self._ensure_folders_exist()
        
        logger.info(f"✅ Enhanced DocumentProcessor initialized")
        logger.info(f"📚 Available namespaces: {list(self.namespace_map.keys())}")

    def _ensure_folders_exist(self):
        """Create default folders if they don't exist"""
        folders = [
            self.knowledge_folder,
            self.products_folder,
            "data/custom",
            "data/mixed",
            "data/all_files"
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)

    def read_text_files_flexible(
        self, 
        request: ProcessDocumentsRequest
    ) -> List[DocumentCreate]:
        """Read text files with flexible namespace assignment"""
        documents = []
        
        # Determine source folder
        if request.source_folder:
            folder_path = request.source_folder
        else:
            # Use default folder based on document type
            if request.document_type == DocumentType.KNOWLEDGE:
                folder_path = self.knowledge_folder
            else:
                folder_path = self.products_folder
        
        # Get target namespace
        target_namespace = self.namespace_map[request.target_namespace]
        original_namespace = self.namespace_map.get(
            NamespaceType.KNOWLEDGE if request.document_type == DocumentType.KNOWLEDGE 
            else NamespaceType.PRODUCTS
        )
        
        logger.info(f"📁 Reading from: {folder_path}")
        logger.info(f"🎯 Target namespace: {target_namespace}")
        logger.info(f"📝 Document type: {request.document_type.value}")
        
        documents = self._process_folder(
            folder_path=folder_path,
            document_type=request.document_type,
            target_namespace=target_namespace,
            original_namespace=original_namespace,
            include_subfolders=request.include_subfolders
        )
        
        return documents

    def read_text_files_bulk(self, request: BulkProcessRequest) -> List[DocumentCreate]:
        """Read text files from any folder and assign to specified namespace"""
        target_namespace = self.namespace_map[request.target_namespace]
        original_namespace = target_namespace  # Same as target in bulk mode
        
        logger.info(f"🔄 Bulk processing from: {request.folder_path}")
        logger.info(f"🎯 Target namespace: {target_namespace}")
        logger.info(f"📝 Document type: {request.document_type.value}")
        logger.info(f"📄 File pattern: {request.file_pattern}")
        
        documents = self._process_folder_with_pattern(
            folder_path=request.folder_path,
            document_type=request.document_type,
            target_namespace=target_namespace,
            original_namespace=original_namespace,
            file_pattern=request.file_pattern
        )
        
        return documents

    def _process_folder(
        self,
        folder_path: str,
        document_type: DocumentType,
        target_namespace: str,
        original_namespace: str,
        include_subfolders: bool = False
    ) -> List[DocumentCreate]:
        """Process all text files in a folder"""
        documents = []
        data_path = Path(folder_path)
        
        if not data_path.exists():
            logger.warning(f"Folder {folder_path} does not exist")
            return documents
        
        # Get text files
        if include_subfolders:
            txt_files = list(data_path.rglob("*.txt"))
        else:
            txt_files = list(data_path.glob("*.txt"))
        
        if not txt_files:
            logger.warning(f"No .txt files found in {folder_path}")
            return documents
        
        for txt_file in txt_files:
            document = self._process_single_file(
                txt_file=txt_file,
                document_type=document_type,
                target_namespace=target_namespace,
                original_namespace=original_namespace
            )
            if document:
                documents.append(document)
        
        logger.info(f"📊 Processed {len(documents)} files from {folder_path}")
        return documents

    def _process_folder_with_pattern(
        self,
        folder_path: str,
        document_type: DocumentType,
        target_namespace: str,
        original_namespace: str,
        file_pattern: str
    ) -> List[DocumentCreate]:
        """Process files matching a specific pattern"""
        documents = []
        data_path = Path(folder_path)
        
        if not data_path.exists():
            logger.warning(f"Folder {folder_path} does not exist")
            return documents
        
        # Get files matching pattern
        files = list(data_path.glob(file_pattern))
        
        if not files:
            logger.warning(f"No files matching '{file_pattern}' found in {folder_path}")
            return documents
        
        for file_path in files:
            if file_path.suffix.lower() == '.txt':
                document = self._process_single_file(
                    txt_file=file_path,
                    document_type=document_type,
                    target_namespace=target_namespace,
                    original_namespace=original_namespace
                )
                if document:
                    documents.append(document)
        
        logger.info(f"📊 Processed {len(documents)} files matching '{file_pattern}'")
        return documents

    def _process_single_file(
        self,
        txt_file: Path,
        document_type: DocumentType,
        target_namespace: str,
        original_namespace: str
    ) -> Optional[DocumentCreate]:
        """Process a single text file"""
        try:
            with open(txt_file, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Clean the content
                content = self._clean_text(content)
                
                if not content.strip():
                    logger.warning(f"Empty file skipped: {txt_file.name}")
                    return None
                
                # Create enhanced metadata
                metadata = self._create_file_metadata(
                    txt_file=txt_file,
                    document_type=document_type,
                    target_namespace=target_namespace,
                    original_namespace=original_namespace
                )
                
                document = DocumentCreate(
                    filename=txt_file.name,
                    content=content,
                    document_type=document_type,
                    namespace=target_namespace,
                    original_namespace=original_namespace,
                    metadata=metadata
                )
                
                logger.info(f"✅ Processed: {txt_file.name} → {target_namespace}")
                return document
                
        except Exception as e:
            logger.error(f"❌ Error processing {txt_file.name}: {e}")
            return None

    def _create_file_metadata(
        self,
        txt_file: Path,
        document_type: DocumentType,
        target_namespace: str,
        original_namespace: str
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for a file"""
        base_metadata = {
            "file_path": str(txt_file),
            "file_name": txt_file.name,
            "file_size": txt_file.stat().st_size,
            "created_at": datetime.fromtimestamp(txt_file.stat().st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(txt_file.stat().st_mtime).isoformat(),
            "document_type": document_type.value,
            "target_namespace": target_namespace,
            "original_namespace": original_namespace,
            "namespace_override": target_namespace != original_namespace
        }
        
        # Add type-specific metadata
        if document_type == DocumentType.KNOWLEDGE:
            base_metadata.update({
                "content_type": "dog_health_knowledge",
                "category": self._extract_knowledge_category(txt_file.name),
                "source": "veterinary_literature",
                "knowledge_type": self._detect_knowledge_type(txt_file.name)
            })
        else:  # PRODUCT
            base_metadata.update({
                "content_type": "marshee_product",
                "product_category": self._extract_product_category(txt_file.name),
                "brand": "Marshee",
                "product_type": self._extract_product_type_from_filename(txt_file.name)
            })
        
        # Add folder-based metadata
        folder_parts = txt_file.parts
        if len(folder_parts) > 1:
            base_metadata["source_folder"] = folder_parts[-2]
            base_metadata["folder_path"] = "/".join(folder_parts[:-1])
        
        return base_metadata

    def create_chunks(self, document: DocumentCreate) -> List[DocumentChunk]:
        """Create chunks with enhanced metadata"""
        try:
            chunks = self.text_splitter.split_text(document.content)
            
            document_chunks = []
            for i, chunk_content in enumerate(chunks):
                if chunk_content.strip():
                    chunk_metadata = {
                        **document.metadata,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk_content),
                        "processing_timestamp": datetime.utcnow().isoformat()
                    }
                    
                    document_chunk = DocumentChunk(
                        content=chunk_content.strip(),
                        metadata=chunk_metadata,
                        document_type=document.document_type,
                        namespace=document.namespace,
                        original_namespace=document.original_namespace
                    )
                    document_chunks.append(document_chunk)
            
            logger.info(f"📄 Split {document.filename} into {len(document_chunks)} chunks")
            return document_chunks
            
        except Exception as e:
            logger.error(f"❌ Error creating chunks for {document.filename}: {e}")
            return []

    def _detect_knowledge_type(self, filename: str) -> str:
        """Detect the type of knowledge from filename"""
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['disease', 'illness', 'condition']):
            return 'medical_conditions'
        elif any(word in filename_lower for word in ['treatment', 'therapy', 'medication']):
            return 'treatments'
        elif any(word in filename_lower for word in ['prevention', 'vaccine', 'preventive']):
            return 'prevention'
        elif any(word in filename_lower for word in ['emergency', 'first_aid', 'urgent']):
            return 'emergency_care'
        else:
            return 'general_knowledge'

    def _extract_product_type_from_filename(self, filename: str) -> str:
        """Extract product type from filename"""
        filename_lower = filename.lower()
        if any(word in filename_lower for word in ['dry', 'kibble']):
            return 'dry_food'
        elif any(word in filename_lower for word in ['wet', 'canned', 'pouch']):
            return 'wet_food'
        elif any(word in filename_lower for word in ['treat', 'snack', 'biscuit']):
            return 'treats'
        elif any(word in filename_lower for word in ['supplement', 'vitamin', 'health']):
            return 'supplements'
        elif any(word in filename_lower for word in ['toy', 'play']):
            return 'toys'
        else:
            return 'general_product'

    # ... include your existing helper methods (_clean_text, _extract_knowledge_category, etc.)

    def get_available_folders(self) -> Dict[str, Any]:
        """Get list of available folders with file counts"""
        folders = {}
        
        # Check default folders
        default_folders = {
            "knowledge": self.knowledge_folder,
            "products": self.products_folder,
            "custom": "data/custom",
            "mixed": "data/mixed",
            "all_files": "data/all_files"
        }
        
        for folder_name, folder_path in default_folders.items():
            path = Path(folder_path)
            if path.exists():
                txt_files = list(path.glob("*.txt"))
                folders[folder_name] = {
                    "path": str(path),
                    "exists": True,
                    "file_count": len(txt_files),
                    "files": [f.name for f in txt_files[:10]]  # Show first 10 files
                }
            else:
                folders[folder_name] = {
                    "path": str(path),
                    "exists": False,
                    "file_count": 0,
                    "files": []
                }
        
        return {
            "available_folders": folders,
            "available_namespaces": list(self.namespace_map.keys()),
            "namespace_mapping": self.namespace_map
        }
