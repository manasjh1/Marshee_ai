import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pymongo.errors import DuplicateKeyError
from database.connection import db_connection
from modals.document import DocumentCreate, DocumentResponse, DocumentChunk

class DocumentRepository:
    def __init__(self):
        self.documents_collection = db_connection.database["documents"]
        self.chunks_collection = db_connection.database["document_chunks"]

    def save_document(self, document: DocumentCreate) -> str:
        """Save document metadata to MongoDB"""
        try:
            document_id = str(uuid.uuid4())
            doc_record = {
                "document_id": document_id,
                "filename": document.filename,
                "content_length": len(document.content),
                "metadata": document.metadata,
                "created_at": datetime.utcnow(),
                "indexed_at": None
            }
            
            result = self.documents_collection.insert_one(doc_record)
            if result.inserted_id:
                return document_id
            else:
                raise Exception("Failed to save document")
                
        except Exception as e:
            print(f"❌ Error saving document: {e}")
            raise

    def save_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Save document chunks to MongoDB"""
        try:
            chunk_records = []
            for chunk in chunks:
                record = {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "embedding": chunk.embedding,
                    "created_at": chunk.created_at
                }
                chunk_records.append(record)
            
            if chunk_records:
                result = self.chunks_collection.insert_many(chunk_records)
                print(f"✅ Saved {len(result.inserted_ids)} chunks to MongoDB")
                return True
            return False
            
        except Exception as e:
            print(f"❌ Error saving chunks: {e}")
            return False

    def get_all_documents(self) -> List[Dict]:
        """Get all document metadata"""
        try:
            return list(self.documents_collection.find({}, {"_id": 0}))
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")
            return []

    def get_document_stats(self) -> Dict:
        """Get document statistics"""
        try:
            total_docs = self.documents_collection.count_documents({})
            total_chunks = self.chunks_collection.count_documents({})
            
            return {
                "total_documents": total_docs,
                "total_chunks": total_chunks,
                "collection_names": ["documents", "document_chunks"]
            }
        except Exception as e:
            return {"error": str(e)}
