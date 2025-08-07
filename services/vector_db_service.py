import os
import logging
import pinecone
from typing import List, Dict, Any, Optional
from modals.document import DocumentChunk, SimilaritySearchRequest, SimilaritySearchResponse

logger = logging.getLogger(__name__)

class PineconeVectorDB:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION"))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.api_key,
            environment=self.environment
        )
        
        self.index = None
        self._setup_index()

    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                logger.info(f"🔄 Creating new Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    pods=1,
                    replicas=1,
                    pod_type="s1.x1"
                )
                logger.info(f"✅ Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up Pinecone index: {e}")
            raise

    def upsert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """Upload document chunks to Pinecone"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                if chunk.embedding:
                    vector = {
                        "id": chunk.chunk_id,
                        "values": chunk.embedding,
                        "metadata": {
                            **chunk.metadata,
                            "content": chunk.content[:1000],  # Limit content size
                            "created_at": chunk.created_at.isoformat()
                        }
                    }
                    vectors.append(vector)
            
            if not vectors:
                logger.warning("No vectors to upsert")
                return False
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"📈 Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
            
            logger.info(f"✅ Successfully upserted {len(vectors)} vectors to Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error upserting to Pinecone: {e}")
            return False

    def similarity_search(self, query_embedding: List[float], top_k: int = 5, filter_metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Perform similarity search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata or {}
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                    "content": match.metadata.get("content", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "content"}
                }
                formatted_results.append(result)
            
            logger.info(f"🔍 Found {len(formatted_results)} similar documents")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {e}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting index stats: {e}")
            return {"error": str(e)}

    def delete_all_vectors(self) -> bool:
        """Delete all vectors from the index (use with caution)"""
        try:
            if not self.index:
                return False
            
            self.index.delete(delete_all=True)
            logger.info("🗑️ Deleted all vectors from Pinecone index")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting vectors: {e}")
            return False
