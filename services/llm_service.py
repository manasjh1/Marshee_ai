import os
import cv2
import numpy as np
import base64
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from groq import Groq
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
from services.embedding_service import GeminiEmbeddingService
from services.vector_db_service import PineconeVectorDB
from modals.chat import ChatSession, YOLODetectionResult

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        # Initialize Groq LLM
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        self.groq_client = Groq(api_key=self.groq_api_key)
        self.groq_model = os.getenv("GROQ_MODEL", "llama3-70b-8192")
        
        # Initialize YOLO models
        self.breed_model_path = os.getenv("YOLO_BREED_MODEL_PATH", "models/breed_detection.pt")
        self.disease_model_path = os.getenv("YOLO_DISEASE_MODEL_PATH", "models/disease_detection.pt")
        
        self.breed_model = None
        self.disease_model = None
        self._load_yolo_models()
        
        # Initialize RAG components
        self.embedding_service = GeminiEmbeddingService()
        self.vector_db = PineconeVectorDB()
        
        logger.info("✅ LLM Service initialized with Groq + YOLO + RAG")

    def _load_yolo_models(self):
        """Load YOLO models"""
        try:
            if os.path.exists(self.breed_model_path):
                self.breed_model = YOLO(self.breed_model_path)
                logger.info(f"✅ Breed model loaded: {self.breed_model_path}")
            else:
                logger.warning(f"⚠️ Breed model not found: {self.breed_model_path}")
            
            if os.path.exists(self.disease_model_path):
                self.disease_model = YOLO(self.disease_model_path)
                logger.info(f"✅ Disease model loaded: {self.disease_model_path}")
            else:
                logger.warning(f"⚠️ Disease model not found: {self.disease_model_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading YOLO models: {e}")

    def _decode_image(self, image_data: str) -> np.ndarray:
        """Decode base64 image"""
        try:
            if "data:image" in image_data:
                image_data = image_data.split(",")[1]
            
            image_bytes = base64.b64decode(image_data)
            pil_image = Image.open(BytesIO(image_bytes))
            
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            return np.array(pil_image)
        except Exception as e:
            logger.error(f"❌ Error decoding image: {e}")
            raise ValueError("Invalid image data")

    def detect_breed(self, image_data: str, session_id: str, user_id: str) -> YOLODetectionResult:
        """Detect dog breed using YOLO"""
        start_time = time.time()
        
        try:
            if not self.breed_model:
                raise ValueError("Breed detection model not available")
            
            image = self._decode_image(image_data)
            results = self.breed_model(image)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    best_confidence = float(confidences[best_idx])
                    best_class_idx = int(classes[best_idx])
                    
                    breed_name = result.names[best_class_idx] if hasattr(result, 'names') else f"breed_{best_class_idx}"
                    text_result = f"Detected breed: {breed_name} (confidence: {best_confidence:.1%})"
                else:
                    breed_name = "Unknown"
                    best_confidence = 0.0
                    text_result = "No clear breed detected. Please try a clearer photo."
            else:
                breed_name = "Unknown"
                best_confidence = 0.0
                text_result = "Could not analyze the image. Please try a different photo."
            
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="breed",
                detected_class=breed_name,
                confidence=best_confidence,
                text_result=text_result,
                additional_info={"session_id": session_id, "user_id": user_id},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Breed detection error: {e}")
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="breed",
                detected_class="Error",
                confidence=0.0,
                text_result=f"Error during breed detection: {str(e)}",
                additional_info={"error": str(e)},
                processing_time=processing_time
            )

    def detect_disease(self, image_data: str, session_id: str, user_id: str) -> YOLODetectionResult:
        """Detect skin condition using YOLO"""
        start_time = time.time()
        
        try:
            if not self.disease_model:
                raise ValueError("Disease detection model not available")
            
            image = self._decode_image(image_data)
            results = self.disease_model(image)
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    confidences = result.boxes.conf.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    
                    best_idx = np.argmax(confidences)
                    best_confidence = float(confidences[best_idx])
                    best_class_idx = int(classes[best_idx])
                    
                    condition_name = result.names[best_class_idx] if hasattr(result, 'names') else f"condition_{best_class_idx}"
                    text_result = f"Detected condition: {condition_name} (confidence: {best_confidence:.1%})"
                else:
                    condition_name = "Normal"
                    best_confidence = 0.8
                    text_result = "No concerning skin conditions detected. The area appears normal."
            else:
                condition_name = "Unclear"
                best_confidence = 0.0
                text_result = "Could not analyze the skin condition clearly. Please try a clearer photo."
            
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="disease",
                detected_class=condition_name,
                confidence=best_confidence,
                text_result=text_result,
                additional_info={"session_id": session_id, "user_id": user_id},
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"❌ Disease detection error: {e}")
            processing_time = time.time() - start_time
            
            return YOLODetectionResult(
                model_type="disease",
                detected_class="Error",
                confidence=0.0,
                text_result=f"Error during health analysis: {str(e)}",
                additional_info={"error": str(e)},
                processing_time=processing_time
            )

    def search_knowledge(self, query: str, namespace: str = "dog-health-knowledge") -> str:
        """Search knowledge base using RAG"""
        try:
            query_embedding = self.embedding_service.create_single_embedding(query)
            
            if not query_embedding:
                return ""
            
            search_results = self.vector_db.similarity_search(
                query_embedding=query_embedding,
                top_k=3,
                namespace=namespace
            )
            
            if not search_results:
                return ""
            
            # Combine relevant knowledge
            knowledge_text = ""
            for result in search_results:
                content = result.get("metadata", {}).get("text", "")
                if content:
                    knowledge_text += content + " "
            
            return knowledge_text.strip()
            
        except Exception as e:
            logger.error(f"❌ RAG search error: {e}")
            return ""

    def generate_response(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate response using Groq LLM"""
        try:
            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."

    # Specific response generators
    def generate_welcome_message(self) -> str:
        """Generate welcome message"""
        prompt = """You are Marshee, an AI dog health assistant. Generate a warm welcome message asking the user to upload a photo of their dog for breed identification and personalized care guidance. Keep it under 100 words."""
        return self.generate_response(prompt, 150)

    def generate_breed_response(self, breed: str, confidence: float) -> str:
        """Generate breed detection response"""
        prompt = f"""You are Marshee, an AI dog health assistant. A user uploaded their dog's photo and you detected it as a {breed} with {confidence:.1%} confidence.

Generate a friendly response that:
1. Confirms the breed detection
2. Shares 2-3 key characteristics about {breed} dogs
3. Shows enthusiasm
4. Keep it under 120 words

Be warm and personal."""
        
        return self.generate_response(prompt, 200)

    def generate_options_message(self, breed: str) -> str:
        """Generate Stage 2 options message"""
        prompt = f"""You are Marshee. The user has a {breed}. Present two options:
1. Disease Detection - for health analysis 
2. General Chat - for care questions

Make it friendly and brief, under 80 words."""
        
        return self.generate_response(prompt, 120)

    def generate_disease_request(self, breed: str) -> str:
        """Generate disease detection request message"""
        prompt = f"""You are Marshee. The user has a {breed} and wants disease detection. Ask them to upload a clear photo of the area of concern. Give brief photo tips. Keep it under 60 words."""
        
        return self.generate_response(prompt, 100)

    def generate_disease_response(self, condition: str, confidence: float, breed: str, knowledge: str) -> str:
        """Generate disease detection response with RAG"""
        prompt = f"""You are Marshee, an AI veterinary assistant. Based on image analysis, you detected: {condition} with {confidence:.1%} confidence.

Dog breed: {breed}

Relevant medical information:
{knowledge}

Generate a caring response that:
1. Acknowledges the condition
2. Explains what it means
3. Provides care recommendations for a {breed}
4. Mentions when to see a vet
5. Offers reassurance
6. Keep it under 250 words

Be professional but empathetic."""
        
        return self.generate_response(prompt, 400)

    def generate_chat_response(self, user_message: str, session: ChatSession, conversation_history: List[str], knowledge: str = "") -> str:
        """Generate chat response with context"""
        
        context = f"User's dog breed: {session.dog_breed}" if session.dog_breed else "General inquiry"
        if session.health_condition:
            context += f" | Detected health condition: {session.health_condition}"
        
        history = "\n".join(conversation_history[-4:]) if conversation_history else "First conversation"
        
        prompt = f"""You are Marshee, an AI dog health assistant.

USER CONTEXT: {context}
CONVERSATION HISTORY: {history}
USER MESSAGE: "{user_message}"
RELEVANT KNOWLEDGE: {knowledge}

Generate a helpful, personalized response that:
1. Addresses their question directly
2. Uses their dog's breed info when relevant
3. Provides actionable advice
4. Maintains a caring tone
5. Keep it under 200 words
6. Ask a follow-up question when appropriate

If you don't have specific info, be honest but still helpful."""
        
        return self.generate_response(prompt, 300)