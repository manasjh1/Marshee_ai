import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from services.yolo_service import YOLOService
from services.rag_service import RAGService
from repositories.chat_repository import ChatRepository
from modals.chat import (
    ChatRequest, ChatResponse, ChatSession, ChatMessage, ChatStage, 
    MessageType, ChatOption, YOLORequest, RAGRequest, YOLODetectionResult
)

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.yolo_service = YOLOService()
        self.rag_service = RAGService()
        self.chat_repo = ChatRepository()
        
        # Stage-based system responses
        self.stage_config = self._initialize_stage_config()

    def _initialize_stage_config(self) -> Dict[ChatStage, Dict[str, Any]]:
        """Initialize stage-based configuration"""
        return {
            ChatStage.STAGE_1_WELCOME: {
                "message": "🐕 **Welcome to Marshee - Your AI Dog Health Assistant!**\n\nI'm here to help you take the best care of your furry friend through a simple 2-stage process:\n\n**Stage 1:** First, I'll identify your dog's breed\n**Stage 2:** Then choose between health check or personalized care chat\n\n📸 **Let's start! Please upload a clear photo of your dog** so I can identify the breed and provide personalized recommendations.",
                "next_input": "image",
                "instructions": "Upload a clear, full-body photo of your dog"
            },
            
            ChatStage.STAGE_2_OPTIONS: {
                "message": "🎉 **Stage 1 Complete!** I've identified your dog's breed.\n\n**Stage 2:** What would you like to do next?",
                "options": [
                    ChatOption(
                        id="disease_detection",
                        text="🩺 Health Check",
                        description="Upload a photo to check for skin conditions or health issues",
                        icon="🏥"
                    ),
                    ChatOption(
                        id="general_chat", 
                        text="💬 General Care Chat",
                        description="Ask questions about nutrition, training, grooming, and care",
                        icon="💭"
                    )
                ]
            },
            
            ChatStage.STAGE_2A_DISEASE_REQUEST: {
                "message": "🩺 **Health Check Mode**\n\nPlease upload a clear, close-up photo of any area of concern on your dog's skin or coat. Make sure the image is:\n\n✅ Well-lit\n✅ Clear and focused\n✅ Shows the affected area clearly\n\n📸 If there's no specific concern, you can upload a general photo of your dog's skin/coat condition for a wellness check.",
                "next_input": "image",
                "instructions": "Upload a clear photo of the skin area to examine"
            }
        }

    async def process_chat_message(self, request: ChatRequest, user_id: str) -> ChatResponse:
        """🎯 Main chat processing - handles all stages"""
        try:
            # Get or create session
            session = await self._get_or_create_session(request.session_id, user_id)
            
            # Save user message if provided
            if request.message or request.image_data:
                await self._save_user_message(session, request)
            
            # 🎭 Process based on current stage
            response = await self._route_by_stage(session, request)
            
            # Save system response
            await self._save_system_message(session, response)
            
            # Update session in database
            await self._update_session(session)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            return self._create_error_response(request.session_id or "error", str(e))

    async def _route_by_stage(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🎭 Route request to appropriate stage handler"""
        
        stage_handlers = {
            ChatStage.STAGE_1_WELCOME: self._handle_stage_1_welcome,
            ChatStage.STAGE_1_BREED_DETECTION: self._handle_stage_1_breed_detection,
            ChatStage.STAGE_1_COMPLETE: self._handle_stage_1_complete,
            ChatStage.STAGE_2_OPTIONS: self._handle_stage_2_options,
            ChatStage.STAGE_2A_DISEASE_REQUEST: self._handle_stage_2a_disease_request,
            ChatStage.STAGE_2A_DISEASE_PROCESSING: self._handle_stage_2a_processing,
            ChatStage.STAGE_2A_DISEASE_RESULT: self._handle_stage_2a_result,
            ChatStage.STAGE_2A_FOLLOWUP: self._handle_stage_2a_followup,
            ChatStage.STAGE_2B_GENERAL_CHAT: self._handle_stage_2b_general_chat
        }
        
        handler = stage_handlers.get(session.current_stage, self._handle_stage_1_welcome)
        return await handler(session, request)

    async def _handle_stage_1_welcome(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🐕 Stage 1: Welcome and request dog image"""
        
        if request.image_data:
            # User uploaded image - process breed detection
            session.current_stage = ChatStage.STAGE_1_BREED_DETECTION
            return await self._process_breed_detection(session, request)
        else:
            # Send welcome message
            config = self.stage_config[ChatStage.STAGE_1_WELCOME]
            return ChatResponse(
                session_id=session.session_id,
                message_id="welcome",
                current_stage=session.current_stage,
                response_type=MessageType.SYSTEM,
                content=config["message"],
                next_input_expected="image"
            )

    async def _handle_stage_1_breed_detection(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🔍 Stage 1: Processing breed detection"""
        return await self._process_breed_detection(session, request)

    async def _process_breed_detection(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🤖 Process breed detection with YOLO"""
        try:
            # Create YOLO request
            yolo_request = YOLORequest(
                image_data=request.image_data,
                model_type="breed",
                session_id=session.session_id,
                user_id=session.user_id
            )
            
            # Call your YOLO model
            detection_result = self.yolo_service.detect_breed(yolo_request)
            
            # Save results to session
            session.breed_detection = detection_result
            session.dog_breed = detection_result.detected_class
            session.breed_confidence = detection_result.confidence
            session.current_stage = ChatStage.STAGE_1_COMPLETE
            
            # Create response with breed info
            breed_name = detection_result.detected_class
            confidence = detection_result.confidence
            
            if confidence > 0.6:
                content = f"🎉 **Breed Detected!**\n\n**{breed_name}** (Confidence: {confidence:.1%})\n\n"
                content += detection_result.text_result + "\n\n"
                content += "**Stage 1 Complete!** ✅\n\n"
                content += self.stage_config[ChatStage.STAGE_2_OPTIONS]["message"]
            else:
                content = f"🤔 I detected **{breed_name}** but with lower confidence ({confidence:.1%}).\n\n"
                content += detection_result.text_result + "\n\n"
                content += "**Stage 1 Complete!** ✅\n\n" 
                content += self.stage_config[ChatStage.STAGE_2_OPTIONS]["message"]
            
            session.current_stage = ChatStage.STAGE_2_OPTIONS
            
            return ChatResponse(
                session_id=session.session_id,
                message_id="breed_result",
                current_stage=session.current_stage,
                response_type=MessageType.OPTIONS,
                content=content,
                options=self.stage_config[ChatStage.STAGE_2_OPTIONS]["options"],
                detection_result=detection_result,
                dog_breed=breed_name,
                next_input_expected="option"
            )
            
        except Exception as e:
            logger.error(f"❌ Breed detection error: {e}")
            return self._create_error_response(session.session_id, "Breed detection failed. Please try uploading a clearer image.")

    async def _handle_stage_1_complete(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """✅ Stage 1 Complete - show options"""
        return await self._handle_stage_2_options(session, request)

    async def _handle_stage_2_options(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🎯 Stage 2: Handle option selection"""
        
        selected_option = request.selected_option or request.message
        
        if not selected_option:
            # Show options again
            config = self.stage_config[ChatStage.STAGE_2_OPTIONS]
            return ChatResponse(
                session_id=session.session_id,
                message_id="show_options",
                current_stage=session.current_stage,
                response_type=MessageType.OPTIONS,
                content=config["message"],
                options=config["options"],
                dog_breed=session.dog_breed,
                next_input_expected="option"
            )
        
        if "disease" in selected_option.lower() or "health" in selected_option.lower():
            # 🩺 User chose disease detection path
            session.current_stage = ChatStage.STAGE_2A_DISEASE_REQUEST
            
            config = self.stage_config[ChatStage.STAGE_2A_DISEASE_REQUEST]
            return ChatResponse(
                session_id=session.session_id,
                message_id="disease_request",
                current_stage=session.current_stage,
                response_type=MessageType.SYSTEM,
                content=config["message"],
                dog_breed=session.dog_breed,
                next_input_expected="image"
            )
        
        elif "chat" in selected_option.lower() or "general" in selected_option.lower():
            # 💬 User chose general chat path
            session.current_stage = ChatStage.STAGE_2B_GENERAL_CHAT
            
            # Get personalized welcome message using RAG
            rag_request = RAGRequest(
                query=f"general care tips and advice for {session.dog_breed}",
                session_id=session.session_id,
                user_id=session.user_id,
                dog_breed=session.dog_breed,
                context_type="breed_info"
            )
            
            rag_response = await self.rag_service.get_personalized_response(rag_request)
            
            content = f"💬 **General Care Chat Mode** - {session.dog_breed}\n\n"
            content += f"Great! I'm here to help you with personalized care advice for your {session.dog_breed}. "
            content += "You can ask me about:\n\n"
            content += "🥘 **Nutrition & Feeding**\n"
            content += "🏃 **Exercise & Training**\n" 
            content += "🛁 **Grooming & Care**\n"
            content += "🏥 **Health & Wellness**\n"
            content += "🎾 **Behavior & Training**\n\n"
            content += f"**Breed-Specific Advice:**\n{rag_response.response}\n\n"
            content += "💭 **What would you like to know about your dog's care?**"
            
            return ChatResponse(
                session_id=session.session_id,
                message_id="general_chat_start",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content=content,
                dog_breed=session.dog_breed,
                next_input_expected="text"
            )
        
        else:
            # Invalid option
            return ChatResponse(
                session_id=session.session_id,
                message_id="invalid_option",
                current_stage=session.current_stage,
                response_type=MessageType.OPTIONS,
                content="I didn't understand that choice. Please select one of the options below:",
                options=self.stage_config[ChatStage.STAGE_2_OPTIONS]["options"],
                dog_breed=session.dog_breed,
                next_input_expected="option"
            )

    async def _handle_stage_2a_disease_request(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🩺 Stage 2A: Request health/disease detection image"""
        
        if request.image_data:
            # User uploaded health image - process disease detection
            session.current_stage = ChatStage.STAGE_2A_DISEASE_PROCESSING
            return await self._process_disease_detection(session, request)
        else:
            # Show request for image again
            config = self.stage_config[ChatStage.STAGE_2A_DISEASE_REQUEST]
            return ChatResponse(
                session_id=session.session_id,
                message_id="disease_image_request",
                current_stage=session.current_stage,
                response_type=MessageType.SYSTEM,
                content=config["message"],
                dog_breed=session.dog_breed,
                next_input_expected="image"
            )

    async def _handle_stage_2a_processing(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🔬 Stage 2A: Processing disease detection"""
        return await self._process_disease_detection(session, request)

    async def _process_disease_detection(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """🤖 Process disease detection with YOLO Model 2"""
        try:
            # Create YOLO request for disease detection
            yolo_request = YOLORequest(
                image_data=request.image_data,
                model_type="disease",
                session_id=session.session_id,
                user_id=session.user_id
            )
            
            # Call your second YOLO model
            detection_result = self.yolo_service.detect_disease(yolo_request)
            
            # Save results to session
            session.disease_detection = detection_result
            session.health_condition = detection_result.detected_class
            session.condition_confidence = detection_result.confidence
            session.current_stage = ChatStage.STAGE_2A_DISEASE_RESULT
            
            # Get LLM guidance based on detection
            rag_request = RAGRequest(
                query=f"health guidance for {detection_result.detected_class} in {session.dog_breed}",
                session_id=session.session_id,
                user_id=session.user_id,
                dog_breed=session.dog_breed,
                health_condition=detection_result.detected_class,
                context_type="health_guidance"
            )
            
            rag_response = await self.rag_service.get_personalized_response(rag_request)
            
            # Create comprehensive response
            condition = detection_result.detected_class
            confidence = detection_result.confidence
            
            content = f"🩺 **Health Assessment Complete**\n\n"
            content += f"**Detected:** {condition} (Confidence: {confidence:.1%})\n\n"
            content += f"**YOLO Analysis:**\n{detection_result.text_result}\n\n"
            content += f"**AI Health Guidance:**\n{rag_response.response}\n\n"
            
            # Add follow-up options
            if condition.lower() not in ["healthy", "normal", "good"]:
                content += "❓ **Would you like more information about:**\n"
                content += "• Treatment options\n"
                content += "• When to see a vet\n" 
                content += "• Home care tips\n"
                content += "• Prevention advice\n\n"
                content += "Just ask me any questions!"
            
            session.current_stage = ChatStage.STAGE_2A_FOLLOWUP
            
            return ChatResponse(
                session_id=session.session_id,
                message_id="disease_result",
                current_stage=session.current_stage,
                response_type=MessageType.DETECTION_RESULT,
                content=content,
                detection_result=detection_result,
                dog_breed=session.dog_breed,
                health_condition=condition,
                next_input_expected="text"
            )
            
        except Exception as e:
            logger.error(f"❌ Disease detection error: {e}")
            return self._create_error_response(session.session_id, "Health assessment failed. Please try uploading a clearer image.")

    async def _handle_stage_2a_result(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """📋 Stage 2A: Show disease detection results"""
        return await self._handle_stage_2a_followup(session, request)

    async def _handle_stage_2a_followup(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """❓ Stage 2A: Handle follow-up questions after disease detection"""
        
        if not request.message:
            return ChatResponse(
                session_id=session.session_id,
                message_id="followup_prompt",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content="Feel free to ask me any questions about your dog's health condition or care!",
                dog_breed=session.dog_breed,
                health_condition=session.health_condition,
                next_input_expected="text"
            )
        
        # Use RAG to answer follow-up questions with health context
        rag_request = RAGRequest(
            query=request.message,
            session_id=session.session_id,
            user_id=session.user_id,
            dog_breed=session.dog_breed,
            health_condition=session.health_condition,
            conversation_history=session.conversation_history[-5:],  # Last 5 messages
            context_type="health_guidance"
        )
        
        rag_response = await self.rag_service.get_personalized_response(rag_request)
        
        # Add to conversation history
        session.conversation_history.append(f"User: {request.message}")
        session.conversation_history.append(f"Assistant: {rag_response.response}")
        
        return ChatResponse(
            session_id=session.session_id,
            message_id="health_followup",
            current_stage=session.current_stage,
            response_type=MessageType.TEXT,
            content=rag_response.response,
            dog_breed=session.dog_breed,
            health_condition=session.health_condition,
            next_input_expected="text"
        )

    async def _handle_stage_2b_general_chat(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """💬 Stage 2B: General personalized chat"""
        
        if not request.message:
            return ChatResponse(
                session_id=session.session_id,
                message_id="chat_prompt",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content=f"I'm here to help with any questions about caring for your {session.dog_breed}! What would you like to know?",
                dog_breed=session.dog_breed,
                next_input_expected="text"
            )
        
        # Use RAG for personalized responses
        rag_request = RAGRequest(
            query=request.message,
            session_id=session.session_id,
            user_id=session.user_id,
            dog_breed=session.dog_breed,
            conversation_history=session.conversation_history[-5:],  # Last 5 messages
            context_type="general"
        )
        
        rag_response = await self.rag_service.get_personalized_response(rag_request)
        
        # Add to conversation history
        session.conversation_history.append(f"User: {request.message}")
        session.conversation_history.append(f"Assistant: {rag_response.response}")
        
        return ChatResponse(
            session_id=session.session_id,
            message_id="general_response",
            current_stage=session.current_stage,
            response_type=MessageType.TEXT,
            content=rag_response.response,
            dog_breed=session.dog_breed,
            next_input_expected="text"
        )

    # Helper methods
    async def _get_or_create_session(self, session_id: Optional[str], user_id: str) -> ChatSession:
        """Get existing session or create new one"""
        if session_id:
            session = await self.chat_repo.get_session(session_id)
            if session and session.user_id == user_id:
                return session
        
        # Create new session
        session = ChatSession(user_id=user_id)
        await self.chat_repo.create_session(session)
        logger.info(f"🆕 Created new session: {session.session_id} for user: {user_id}")
        return session

    async def _save_user_message(self, session: ChatSession, request: ChatRequest):
        """Save user message to database"""
        try:
            message = ChatMessage(
                session_id=session.session_id,
                user_id=session.user_id,
                message_type=MessageType.IMAGE if request.image_data else MessageType.TEXT,
                content=request.message or "Image uploaded",
                image_data=request.image_data,
                is_user_message=True
            )
            await self.chat_repo.save_message(message)
        except Exception as e:
            logger.error(f"❌ Error saving user message: {e}")

    async def _save_system_message(self, session: ChatSession, response: ChatResponse):
        """Save system response to database"""
        try:
            message = ChatMessage(
                session_id=session.session_id,
                user_id=session.user_id,
                message_type=response.response_type,
                content=response.content,
                options=response.options,
                detection_result=response.detection_result,
                is_user_message=False
            )
            await self.chat_repo.save_message(message)
        except Exception as e:
            logger.error(f"❌ Error saving system message: {e}")

    async def _update_session(self, session: ChatSession):
        """Update session in database"""
        try:
            session.updated_at = datetime.utcnow()
            await self.chat_repo.update_session(session)
        except Exception as e:
            logger.error(f"❌ Error updating session: {e}")

    def _create_error_response(self, session_id: str, error_message: str) -> ChatResponse:
        """Create error response"""
        return ChatResponse(
            session_id=session_id,
            message_id="error",
            current_stage=ChatStage.STAGE_1_WELCOME,
            response_type=MessageType.ERROR,
            content=f"❌ {error_message}\n\nLet's start over. Please try again.",
            next_input_expected="image"
        )

    # Public methods for router
    async def get_session_info(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        session = await self.chat_repo.get_session(session_id)
        if not session or session.user_id != user_id:
            return None
        
        messages = await self.chat_repo.get_session_messages(session_id)
        
        return {
            "session_id": session.session_id,
            "current_stage": session.current_stage,
            "dog_breed": session.dog_breed,
            "health_condition": session.health_condition,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
            "message_count": len(messages),
            "is_active": session.is_active
        }

    async def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all user sessions"""
        sessions = await self.chat_repo.get_user_sessions(user_id)
        
        return [
            {
                "session_id": session.session_id,
                "current_stage": session.current_stage,
                "dog_breed": session.dog_breed,
                "health_condition": session.health_condition,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "is_active": session.is_active
            }
            for session in sessions
        ]

    async def end_session(self, session_id: str, user_id: str) -> bool:
        """End a chat session"""
        session = await self.chat_repo.get_session(session_id)
        if not session or session.user_id != user_id:
            return False
        
        return await self.chat_repo.end_session(session_id)