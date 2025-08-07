import os
import logging
import google.generativeai as genai
from typing import List, Optional, Dict, Any
import numpy as np
from modals.document import DocumentChunk, EmbeddingRequest

logger = logging.getLogger(__name__)

class GeminiEmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001")
        self.embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "768"))
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info(f"✅ Gemini Embedding Service initialized with model: {self.model_name}")

    def create_single_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a single text"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            
            # Create embedding using Gemini
            result = genai.embed_content(
                model=self.model_name,
                content=text.strip(),
                output_dimensionality=self.embedding_dimension
            )
            
            if result and hasattr(result, 'embedding') and result.embedding:
                embedding = result.embedding
                logger.debug(f"✅ Created embedding with {len(embedding)} dimensions")
                return embedding
            else:
                logger.error("No embedding returned from Gemini API")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {e}")
            return None

    def create_batch_embeddings(self, texts: List[str], batch_size: int = 5) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts in batches"""
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"🔄 Creating embeddings for {total_texts} texts in batches of {batch_size}")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.create_single_embedding(text)
                batch_embeddings.append(embedding)
            
            embeddings.extend(batch_embeddings)
            logger.info(f"📈 Progress: {min(i + batch_size, total_texts)}/{total_texts} embeddings created")
        
        return embeddings

    def embed_document_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to document chunks"""
        logger.info(f"🎯 Creating embeddings for {len(chunks)} document chunks")
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.create_batch_embeddings(texts)
        
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            if embedding is not None:
                chunk.embedding = embedding
                embedded_chunks.append(chunk)
                logger.debug(f"✅ Embedded chunk: {chunk.chunk_id[:8]}...")
            else:
                logger.warning(f"⚠️ Skipping chunk {chunk.chunk_id[:8]} - no embedding created")
        
        logger.info(f"✅ Successfully embedded {len(embedded_chunks)}/{len(chunks)} chunks")
        return embedded_chunks

    def similarity_search(self, query_text: str, stored_embeddings: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search using cosine similarity"""
        try:
            # Create embedding for query
            query_embedding = self.create_single_embedding(query_text)
            if not query_embedding:
                logger.error("Could not create embedding for query")
                return []
            
            query_vector = np.array(query_embedding)
            
            # Calculate similarities
            similarities = []
            for item in stored_embeddings:
                if 'embedding' in item and item['embedding']:
                    stored_vector = np.array(item['embedding'])
                    
                    # Cosine similarity
                    similarity = np.dot(query_vector, stored_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
                    )
                    
                    similarities.append({
                        'content': item.get('content', ''),
                        'metadata': item.get('metadata', {}),
                        'similarity_score': float(similarity),
                        'chunk_id': item.get('chunk_id', '')
                    })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return []

    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        """Validate embedding has correct dimensions"""
        return len(embedding) == self.embedding_dimension
