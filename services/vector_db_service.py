import os
import logging
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from modals.document import DocumentChunk, SimilaritySearchRequest, SimilaritySearchResponse

logger = logging.getLogger(__name__)

class PineconeVectorDB:
    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "marshee")
        self.dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        # Initialize Pinecone with new client
        self.pc = Pinecone(api_key=self.api_key)
        
        self.index = None
        self._setup_index()

    def _setup_index(self):
        """Create or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"🔄 Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                logger.info(f"✅ Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"❌ Error setting up Pinecone index: {e}")
            raise

    def upsert_chunks(self, chunks: List[DocumentChunk], namespace: str = None) -> bool:
        """Upload document chunks to Pinecone with namespace support"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Use provided namespace or default
            target_namespace = namespace or "default"
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                if chunk.embedding:
                    # Ensure content length is within limits
                    content = chunk.content[:1000] if len(chunk.content) > 1000 else chunk.content
                    
                    vector = {
                        "id": chunk.chunk_id,
                        "values": chunk.embedding,
                        "metadata": {
                            **{k: v for k, v in chunk.metadata.items() if isinstance(v, (str, int, float, bool))},
                            "content": content,
                            "created_at": chunk.created_at.isoformat(),
                            "namespace": target_namespace,
                            "document_type": chunk.document_type.value if hasattr(chunk.document_type, 'value') else str(chunk.document_type)
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
                self.index.upsert(vectors=batch, namespace=target_namespace)
                logger.info(f"📈 Upserted batch {i//batch_size + 1}: {len(batch)} vectors to namespace '{target_namespace}'")
            
            logger.info(f"✅ Successfully upserted {len(vectors)} vectors to namespace '{target_namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error upserting to Pinecone: {e}")
            return False

    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filter_metadata: Optional[Dict] = None,
        namespace: str = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone with namespace support"""
        try:
            if not self.index:
                raise Exception("Pinecone index not initialized")
            
            # Use provided namespace or default
            search_namespace = namespace or "default"
            
            # Perform similarity search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata or {},
                namespace=search_namespace
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": float(match.score),
                    "content": match.metadata.get("content", ""),
                    "metadata": {k: v for k, v in match.metadata.items() if k != "content"},
                    "namespace": search_namespace
                }
                formatted_results.append(result)
            
            logger.info(f"🔍 Found {len(formatted_results)} similar documents in namespace '{search_namespace}'")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {e}")
            return []

    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors from a specific namespace"""
        try:
            if not self.index:
                return False
            
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"🗑️ Deleted all vectors from namespace '{namespace}'")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting namespace '{namespace}': {e}")
            return False

    def get_namespace_stats(self) -> Dict[str, Any]:
        """Get statistics about all namespaces"""
        try:
            if not self.index:
                return {"error": "Index not initialized"}
            
            stats = self.index.describe_index_stats()
            
            namespace_stats = {}
            if hasattr(stats, 'namespaces') and stats.namespaces:
                for ns_name, ns_stats in stats.namespaces.items():
                    namespace_stats[ns_name] = {
                        "vector_count": ns_stats.vector_count if hasattr(ns_stats, 'vector_count') else 0
                    }
            
            return {
                "index_name": self.index_name,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count,
                "namespaces": namespace_stats
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting namespace stats: {e}")
            return {"error": str(e)}

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the Pinecone index"""
        return self.get_namespace_stats()

    def list_namespaces(self) -> List[str]:
        """List all available namespaces"""
        try:
            stats = self.get_namespace_stats()
            if "namespaces" in stats:
                return list(stats["namespaces"].keys())
            return []
        except Exception as e:
            logger.error(f"❌ Error listing namespaces: {e}")
            return []

    def delete_all_vectors(self, namespace: str = None) -> bool:
        """Delete all vectors from the index or specific namespace"""
        try:
            if not self.index:
                return False
            
            if namespace:
                return self.delete_namespace(namespace)
            else:
                self.index.delete(delete_all=True)
                logger.info("🗑️ Deleted all vectors from Pinecone index")
                return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting vectors: {e}")
            return False