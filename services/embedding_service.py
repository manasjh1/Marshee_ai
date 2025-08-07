import os
import logging
import google.generativeai as genai
from typing import List, Optional, Dict, Any
import numpy as np
import time
from modals.document import DocumentChunk, EmbeddingRequest

logger = logging.getLogger(__name__)

class GeminiEmbeddingService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model_name = "models/embedding-001"
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        logger.info(f"✅ Gemini Embedding Service initialized with model: {self.model_name}")

    def create_single_embedding(self, text: str) -> Optional[List[float]]:
        """Create embedding for a single text"""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            
            # Clean the text
            cleaned_text = text.strip()
            if len(cleaned_text) > 20000:
                cleaned_text = cleaned_text[:20000]
            
            # The response is a dictionary with 'embedding' key
            result = genai.embed_content(
                model="models/embedding-001",
                content=cleaned_text
            )
            
            # Check if result is a dictionary with 'embedding' key
            if result and isinstance(result, dict) and 'embedding' in result and result['embedding']:
                embedding = result['embedding']
                return embedding
            # Fallback: check if it's an object with embedding attribute
            elif result and hasattr(result, 'embedding') and result.embedding:
                embedding = result.embedding
                return embedding
            else:
                logger.error(f"No embedding found in response")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creating embedding: {e}")
            return None

    def create_batch_embeddings(self, texts: List[str], batch_size: int = 5) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts in larger batches with optimized timing"""
        embeddings = []
        total_texts = len(texts)
        
        print(f"🔄 Creating embeddings for {total_texts} texts in batches of {batch_size}")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            print(f"📦 Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
            
            for j, text in enumerate(batch):
                current_index = i + j + 1
                
                # Show progress every 10 items or for the last item in batch
                if current_index % 10 == 0 or j == len(batch) - 1:
                    print(f"   📊 Processing {current_index}/{total_texts}")
                
                embedding = self.create_single_embedding(text)
                batch_embeddings.append(embedding)
                
                # Shorter delay between requests in the same batch
                if j < len(batch) - 1:
                    time.sleep(0.5)  # Reduced from 1 second
            
            embeddings.extend(batch_embeddings)
            successful_in_batch = sum(1 for e in batch_embeddings if e is not None)
            print(f"   ✅ Batch complete: {successful_in_batch}/{len(batch)} successful")
            
            # Shorter delay between batches
            if i + batch_size < total_texts:
                print(f"   ⏳ Waiting 2 seconds before next batch...")
                time.sleep(2)  # Reduced from longer delays
        
        successful_total = sum(1 for e in embeddings if e is not None)
        success_rate = (successful_total / total_texts * 100) if total_texts > 0 else 0
        print(f"🎯 Total successful embeddings: {successful_total}/{total_texts} ({success_rate:.1f}%)")
        
        return embeddings

    def embed_document_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add embeddings to document chunks with increased limits"""
        total_chunks = len(chunks)
        print(f"🎯 Creating embeddings for {total_chunks} document chunks")
        
        # Increased limits for better processing
        max_chunks_per_session = 200  # Increased from 50
        
        if total_chunks > max_chunks_per_session:
            print(f"📊 Large dataset detected ({total_chunks} chunks)")
            print(f"   Processing first {max_chunks_per_session} chunks in this session")
            print(f"   Remaining {total_chunks - max_chunks_per_session} chunks can be processed in next run")
            chunks_to_process = chunks[:max_chunks_per_session]
        else:
            chunks_to_process = chunks
            print(f"📊 Processing all {total_chunks} chunks")
        
        # Extract texts for embedding
        texts = []
        for i, chunk in enumerate(chunks_to_process):
            # Clean and prepare text
            text = chunk.content.strip()
            if len(text) > 15000:  # Slightly smaller limit for safety
                text = text[:15000] + "..."
            texts.append(text)
        
        print(f"📝 Prepared {len(texts)} texts for embedding")
        
        # Create embeddings with optimized batching
        embeddings = self.create_batch_embeddings(texts, batch_size=8)  # Increased batch size
        
        # Combine chunks with embeddings
        embedded_chunks = []
        failed_chunks = []
        
        for chunk, embedding in zip(chunks_to_process, embeddings):
            if embedding is not None:
                chunk.embedding = embedding
                embedded_chunks.append(chunk)
            else:
                failed_chunks.append(chunk.chunk_id[:8])
        
        success_rate = len(embedded_chunks) / len(chunks_to_process) * 100 if chunks_to_process else 0
        
        print(f"\n📊 EMBEDDING RESULTS:")
        print(f"✅ Successfully embedded: {len(embedded_chunks)}/{len(chunks_to_process)} chunks ({success_rate:.1f}%)")
        
        if failed_chunks:
            print(f"❌ Failed chunks: {len(failed_chunks)}")
            if len(failed_chunks) <= 5:
                print(f"   Failed IDs: {', '.join(failed_chunks)}")
        
        if total_chunks > max_chunks_per_session:
            remaining = total_chunks - max_chunks_per_session
            print(f"\n💡 TO PROCESS REMAINING {remaining} CHUNKS:")
            print(f"   1. Wait 5-10 minutes to avoid rate limits")
            print(f"   2. Update your data folder to contain only unprocessed files")
            print(f"   3. Run the script again")
            print(f"   4. Or increase max_chunks_per_session if needed")
        
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
        return isinstance(embedding, list) and len(embedding) == 768

    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension"""
        return 768