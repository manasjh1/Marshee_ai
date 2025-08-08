import logging
import os
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from services.vector_db_service import PineconeVectorDB
from services.embedding_service import GeminiEmbeddingService
from database.connection import db_connection
from modals.chat import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # Initialize components
        self.vector_db = PineconeVectorDB()
        self.embedding_service = GeminiEmbeddingService()
        
        # Initialize Gemini for text generation
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.chat_model = genai.GenerativeModel('gemini-pro')
        else:
            raise ValueError("GOOGLE_API_KEY not found for RAG service")
        
        # MongoDB collections for user context
        self.chat_sessions_collection = db_connection.chat_sessions_collection

        logger.info("RAG Service initialized")

    async def get_personalized_response(self, request: RAGRequest) -> RAGResponse:
        """
        Generate personalized response using RAG
        Combines vector search with LLM generation
        """
        try:
            # Get relevant context from vector database
            context_docs = await self._get_relevant_context(request)
            
            # Get user's dog information and chat history
            user_context = await self._get_user_context(request.user_id, request.session_id)
            
            # Create prompt for LLM
            prompt = self._create_personalized_prompt(request, context_docs, user_context)
            
            # Generate response using Gemini
            response = await self._generate_llm_response(prompt)
            
            # Create RAG response
            return RAGResponse(
                response=response,
                sources=context_docs,
                confidence=0.85,  # You can implement confidence scoring
                response_type=request.context_type
            )
            
        except Exception as e:
            logger.error(f"RAG service error: {e}")
            return RAGResponse(
                response="I'm sorry, I'm having trouble accessing my knowledge base right now. Please try again in a moment.",
                sources=[],
                confidence=0.0,
                response_type="error"
            )

    async def _get_relevant_context(self, request: RAGRequest) -> List[Dict[str, Any]]:
        """Search vector database for relevant information"""
        try:
            # Create embedding for the query
            query_embedding = self.embedding_service.create_single_embedding(request.query)
            
            if not query_embedding:
                return []
            
            # Determine which namespace to search
            namespace = self._determine_search_namespace(request)
            
            # Search vector database
            search_results = self.vector_db.similarity_search(
                query_embedding=query_embedding,
                top_k=5,  # Get top 5 relevant documents
                filter_metadata={},
                namespace=namespace
            )
            
            # Format results for context
            context_docs = []
            for result in search_results:
                context_docs.append({
                    "content": result["content"],
                    "source": result["metadata"].get("file_name", "Unknown"),
                    "score": result["score"],
                    "document_type": result["metadata"].get("document_type", "general")
                })
            
            logger.info(f"🔍 Found {len(context_docs)} relevant documents in namespace '{namespace}'")
            return context_docs
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return []

    def _determine_search_namespace(self, request: RAGRequest) -> str:
        """Determine which namespace to search based on context"""
        if request.context_type == "health_guidance":
            return "dog-health-knowledge"
        elif request.context_type == "product":
            return "marshee-products"
        else:
            # For general queries, search knowledge namespace
            return "dog-health-knowledge"

    async def _get_user_context(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get user's dog information and conversation history"""
        try:
            user_context = {
                "dog_breed": None,
                "dog_info": {},
                "recent_detections": [],
                "conversation_history": []
            }
            
            # Get current session information
            session_doc = self.chat_sessions_collection.find_one({"session_id": session_id})
            if session_doc:
                user_context["dog_breed"] = session_doc.get("dog_breed")
                user_context["health_condition"] = session_doc.get("health_condition")
                user_context["recent_detections"] = [
                    session_doc.get("breed_detection"),
                    session_doc.get("disease_detection")
                ]
                user_context["conversation_history"] = session_doc.get("conversation_history", [])
            
            return user_context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}

    def _create_personalized_prompt(
        self, 
        request: RAGRequest, 
        context_docs: List[Dict[str, Any]], 
        user_context: Dict[str, Any]
    ) -> str:
        """Create personalized prompt for LLM"""
        
        # Base system instruction
        system_prompt = """You are an expert AI dog care assistant. Provide helpful, accurate, and personalized advice about dog care, health, nutrition, and behavior. Always prioritize the dog's safety and wellbeing.

IMPORTANT GUIDELINES:
- If discussing health issues, always recommend consulting a veterinarian for serious concerns
- Provide practical, actionable advice
- Be empathetic and understanding
- Use the provided context to give specific, relevant information
- Personalize responses based on the dog's breed when known
- Keep responses concise but informative
"""
        
        # Add user's dog information
        dog_info = ""
        if user_context.get("dog_breed"):
            dog_info = f"\nUSER'S DOG INFORMATION:\n"
            dog_info += f"- Breed: {user_context['dog_breed']}\n"
            
            if user_context.get("health_condition"):
                dog_info += f"- Recent Health Condition: {user_context['health_condition']}\n"
        
        # Add relevant context from vector database
        context_section = ""
        if context_docs:
            context_section = "\nRELEVANT INFORMATION:\n"
            for i, doc in enumerate(context_docs, 1):
                context_section += f"{i}. {doc['content'][:300]}...\n"
                context_section += f"   Source: {doc['source']}\n\n"
        
        # Add conversation history if available
        history_section = ""
        if user_context.get("conversation_history"):
            recent_history = user_context["conversation_history"][-3:]  # Last 3 messages
            history_section = "\nRECENT CONVERSATION:\n"
            for msg in recent_history:
                history_section += f"- {msg}\n"
        
        # Add recent detection results
        detection_section = ""
        recent_detections = user_context.get("recent_detections", [])
        if any(recent_detections):
            detection_section = "\nRECENT DETECTION RESULTS:\n"
            
            if recent_detections[0]:  # Breed detection
                breed_det = recent_detections[0]
                detection_section += f"- Breed detected: {breed_det.get('detected_class')} (confidence: {breed_det.get('confidence', 0):.1%})\n"
            
            if recent_detections[1]:  # Disease detection
                disease_det = recent_detections[1]
                detection_section += f"- Health condition detected: {disease_det.get('detected_class')} (confidence: {disease_det.get('confidence', 0):.1%})\n"
        
        # Construct final prompt
        full_prompt = f"""{system_prompt}

{dog_info}
{context_section}
{history_section}
{detection_section}

USER'S QUESTION: {request.query}

Please provide a helpful, personalized response based on the above information. If the question is about health concerns, remind the user to consult with their veterinarian for professional medical advice."""
        
        return full_prompt

    async def _generate_llm_response(self, prompt: str) -> str:
        """Generate response using Gemini LLM"""
        try:
            response = self.chat_model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try rephrasing your question or try again in a moment."

    async def get_breed_specific_advice(self, breed: str, topic: str) -> str:
        """Get breed-specific advice for a particular topic"""
        request = RAGRequest(
            query=f"{breed} {topic}",
            session_id="breed_advice",
            user_id="system",
            dog_breed=breed,
            context_type="breed_info"
        )
        
        response = await self.get_personalized_response(request)
        return response.response

    async def get_health_guidance(self, condition: str, breed: str = None) -> str:
        """Get health guidance for a specific condition"""
        query = f"dog health {condition}"
        if breed:
            query += f" {breed}"
        
        request = RAGRequest(
            query=query,
            session_id="health_guidance",
            user_id="system",
            dog_breed=breed,
            context_type="health_guidance"
        )
        
        response = await self.get_personalized_response(request)
        return response.response

    def get_system_status(self) -> Dict[str, Any]:
        """Get RAG system status"""
        return {
            "vector_db_connected": self.vector_db.index is not None,
            "embedding_service_ready": self.embedding_service is not None,
            "llm_configured": self.chat_model is not None,
            "mongodb_connected": db_connection.client is not None
        }