import logging
from typing import Dict, List, Optional
from datetime import datetime
from repositories.chat_repository import ChatRepository
from services.llm_service import LLMService
from modals.chat import (
    ChatRequest, ChatResponse, ChatSession, ChatMessage, ChatStage, 
    MessageType, ChatOption, YOLODetectionResult
)

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.chat_repo = ChatRepository()
        self.llm_service = LLMService()
        
        logger.info("✅ Chat Service initialized")

    async def process_chat_message(self, request: ChatRequest, user_id: str) -> ChatResponse:
        """Main method to process all chat interactions"""
        try:
            # Get or create session
            if request.session_id:
                session = await self.chat_repo.get_session(request.session_id)
                if not session or session.user_id != user_id:
                    raise ValueError("Invalid session")
            else:
                # Create new session - Stage 1 Welcome
                session = ChatSession(user_id=user_id)
                await self.chat_repo.create_session(session)
            
            # Process based on current stage
            if session.current_stage == ChatStage.STAGE_1_WELCOME:
                return await self._handle_welcome(session, request)
            elif session.current_stage == ChatStage.STAGE_1_BREED_DETECTION:
                return await self._handle_breed_detection(session, request)
            elif session.current_stage == ChatStage.STAGE_2_OPTIONS:
                return await self._handle_options(session, request)
            elif session.current_stage == ChatStage.STAGE_2A_DISEASE_REQUEST:
                return await self._handle_disease_request(session, request)
            elif session.current_stage in [ChatStage.STAGE_2A_DISEASE_RESULT, ChatStage.STAGE_2B_GENERAL_CHAT]:
                return await self._handle_chat(session, request)
            else:
                return await self._handle_fallback(session)
                
        except Exception as e:
            logger.error(f"❌ Chat processing error: {e}")
            return ChatResponse(
                session_id=request.session_id or "error",
                message_id="error",
                current_stage=ChatStage.STAGE_1_WELCOME,
                response_type=MessageType.ERROR,
                content="Something went wrong. Let's start fresh!",
                next_input_expected="image"
            )

    async def _handle_welcome(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """Stage 1: Welcome and request dog image"""
        if request.image_data:
            # Process breed detection
            session.current_stage = ChatStage.STAGE_1_BREED_DETECTION
            await self.chat_repo.update_session(session)
            return await self._handle_breed_detection(session, request)
        
        # Generate welcome message
        welcome_message = self.llm_service.generate_welcome_message()
        
        # Save message
        message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.TEXT,
            content=welcome_message,
            is_user_message=False
        )
        await self.chat_repo.save_message(message)
        
        return ChatResponse(
            session_id=session.session_id,
            message_id=message.message_id,
            current_stage=session.current_stage,
            response_type=MessageType.TEXT,
            content=welcome_message,
            next_input_expected="image"
        )

    async def _handle_breed_detection(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """Stage 1: Process breed detection"""
        if not request.image_data:
            return ChatResponse(
                session_id=session.session_id,
                message_id="error",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content="Please upload a clear photo of your dog!",
                next_input_expected="image"
            )
        
        # Save user image
        user_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.IMAGE,
            content="Uploaded dog photo",
            image_data=request.image_data,
            is_user_message=True
        )
        await self.chat_repo.save_message(user_message)
        
        # Run breed detection
        detection_result = self.llm_service.detect_breed(
            request.image_data, session.session_id, session.user_id
        )
        
        # Update session
        session.breed_detection = detection_result
        session.dog_breed = detection_result.detected_class
        session.breed_confidence = detection_result.confidence
        session.current_stage = ChatStage.STAGE_2_OPTIONS
        await self.chat_repo.update_session(session)
        
        # Generate responses
        breed_response = self.llm_service.generate_breed_response(
            detection_result.detected_class, detection_result.confidence
        )
        options_message = self.llm_service.generate_options_message(detection_result.detected_class)
        full_response = f"{breed_response}\n\n{options_message}"
        
        # Create options
        options = [
            ChatOption(
                id="disease_detection",
                text="🩺 Disease Detection",
                description="Upload a photo to check for skin conditions",
                icon="🩺"
            ),
            ChatOption(
                id="general_chat",
                text="💬 General Chat",
                description="Ask questions about your dog's care",
                icon="💬"
            )
        ]
        
        # Save response
        response_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.OPTIONS,
            content=full_response,
            options=options,
            detection_result=detection_result,
            is_user_message=False
        )
        await self.chat_repo.save_message(response_message)
        
        return ChatResponse(
            session_id=session.session_id,
            message_id=response_message.message_id,
            current_stage=session.current_stage,
            response_type=MessageType.OPTIONS,
            content=full_response,
            options=options,
            detection_result=detection_result,
            dog_breed=session.dog_breed,
            next_input_expected="option"
        )

    async def _handle_options(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """Stage 2: Handle option selection"""
        if not request.selected_option:
            return ChatResponse(
                session_id=session.session_id,
                message_id="error",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content="Please select Disease Detection or General Chat.",
                next_input_expected="option"
            )
        
        if request.selected_option == "disease_detection":
            session.current_stage = ChatStage.STAGE_2A_DISEASE_REQUEST
            await self.chat_repo.update_session(session)
            
            disease_request = self.llm_service.generate_disease_request(session.dog_breed)
            
            message = ChatMessage(
                session_id=session.session_id,
                user_id=session.user_id,
                message_type=MessageType.TEXT,
                content=disease_request,
                is_user_message=False
            )
            await self.chat_repo.save_message(message)
            
            return ChatResponse(
                session_id=session.session_id,
                message_id=message.message_id,
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content=disease_request,
                dog_breed=session.dog_breed,
                next_input_expected="image"
            )
        
        elif request.selected_option == "general_chat":
            session.current_stage = ChatStage.STAGE_2B_GENERAL_CHAT
            await self.chat_repo.update_session(session)
            
            chat_welcome = f"Perfect! I'm ready to help with any questions about caring for your {session.dog_breed}. What would you like to know?"
            
            message = ChatMessage(
                session_id=session.session_id,
                user_id=session.user_id,
                message_type=MessageType.TEXT,
                content=chat_welcome,
                is_user_message=False
            )
            await self.chat_repo.save_message(message)
            
            return ChatResponse(
                session_id=session.session_id,
                message_id=message.message_id,
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content=chat_welcome,
                dog_breed=session.dog_breed,
                next_input_expected="text"
            )

    async def _handle_disease_request(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """Stage 2A: Process disease detection"""
        if not request.image_data:
            return ChatResponse(
                session_id=session.session_id,
                message_id="error",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content="Please upload a photo of the area of concern on your dog.",
                next_input_expected="image"
            )
        
        # Save user image
        user_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.IMAGE,
            content="Uploaded health image",
            image_data=request.image_data,
            is_user_message=True
        )
        await self.chat_repo.save_message(user_message)
        
        # Run disease detection
        detection_result = self.llm_service.detect_disease(
            request.image_data, session.session_id, session.user_id
        )
        
        # Update session
        session.disease_detection = detection_result
        session.health_condition = detection_result.detected_class
        session.condition_confidence = detection_result.confidence
        session.current_stage = ChatStage.STAGE_2A_DISEASE_RESULT
        await self.chat_repo.update_session(session)
        
        # Get relevant knowledge
        knowledge = ""
        if detection_result.detected_class not in ["Error", "Normal", "Unclear"]:
            search_query = f"{detection_result.detected_class} {session.dog_breed} treatment"
            knowledge = self.llm_service.search_knowledge(search_query)
        
        # Generate response
        disease_response = self.llm_service.generate_disease_response(
            detection_result.detected_class,
            detection_result.confidence,
            session.dog_breed,
            knowledge
        )
        
        # Save response
        response_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.DETECTION_RESULT,
            content=disease_response,
            detection_result=detection_result,
            is_user_message=False
        )
        await self.chat_repo.save_message(response_message)
        
        return ChatResponse(
            session_id=session.session_id,
            message_id=response_message.message_id,
            current_stage=session.current_stage,
            response_type=MessageType.DETECTION_RESULT,
            content=disease_response,
            detection_result=detection_result,
            dog_breed=session.dog_breed,
            health_condition=session.health_condition,
            next_input_expected="text"
        )

    async def _handle_chat(self, session: ChatSession, request: ChatRequest) -> ChatResponse:
        """Handle general chat and follow-up questions"""
        if not request.message:
            return ChatResponse(
                session_id=session.session_id,
                message_id="error",
                current_stage=session.current_stage,
                response_type=MessageType.TEXT,
                content=f"Please ask me anything about caring for your {session.dog_breed}!",
                next_input_expected="text"
            )
        
        # Save user message
        user_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.TEXT,
            content=request.message,
            is_user_message=True
        )
        await self.chat_repo.save_message(user_message)
        
        # Get conversation history
        messages = await self.chat_repo.get_session_messages(session.session_id)
        conversation_history = [msg.content for msg in messages[-8:]]
        
        # Search for relevant knowledge
        search_query = f"{request.message} {session.dog_breed}"
        if session.health_condition:
            search_query += f" {session.health_condition}"
        
        knowledge = self.llm_service.search_knowledge(search_query)
        
        # Generate response
        response_content = self.llm_service.generate_chat_response(
            request.message, session, conversation_history, knowledge
        )
        
        # Save response
        response_message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.TEXT,
            content=response_content,
            is_user_message=False
        )
        await self.chat_repo.save_message(response_message)
        
        # Update history
        session.conversation_history.append(f"User: {request.message}")
        session.conversation_history.append(f"Assistant: {response_content}")
        await self.chat_repo.update_session(session)
        
        return ChatResponse(
            session_id=session.session_id,
            message_id=response_message.message_id,
            current_stage=session.current_stage,
            response_type=MessageType.TEXT,
            content=response_content,
            dog_breed=session.dog_breed,
            health_condition=session.health_condition,
            next_input_expected="text"
        )

    async def _handle_fallback(self, session: ChatSession) -> ChatResponse:
        """Fallback handler"""
        session.current_stage = ChatStage.STAGE_1_WELCOME
        await self.chat_repo.update_session(session)
        
        welcome_message = "Let's start fresh! Please upload a photo of your dog."
        
        message = ChatMessage(
            session_id=session.session_id,
            user_id=session.user_id,
            message_type=MessageType.TEXT,
            content=welcome_message,
            is_user_message=False
        )
        await self.chat_repo.save_message(message)
        
        return ChatResponse(
            session_id=session.session_id,
            message_id=message.message_id,
            current_stage=session.current_stage,
            response_type=MessageType.TEXT,
            content=welcome_message,
            next_input_expected="image"
        )

    # Helper methods
    async def get_session_info(self, session_id: str, user_id: str) -> Optional[Dict]:
        """Get session info"""
        try:
            session = await self.chat_repo.get_session(session_id)
            if not session or session.user_id != user_id:
                return None
            
            messages = await self.chat_repo.get_session_messages(session_id)
            
            return {
                "session": {
                    "session_id": session.session_id,
                    "current_stage": session.current_stage.value,
                    "dog_breed": session.dog_breed,
                    "health_condition": session.health_condition,
                    "is_active": session.is_active
                },
                "messages": [
                    {
                        "content": msg.content,
                        "is_user_message": msg.is_user_message,
                        "timestamp": msg.timestamp
                    }
                    for msg in messages
                ]
            }
        except Exception as e:
            logger.error(f"❌ Error getting session info: {e}")
            return None

    async def get_user_sessions(self, user_id: str) -> List[Dict]:
        """Get user sessions"""
        try:
            sessions = await self.chat_repo.get_user_sessions(user_id)
            return [
                {
                    "session_id": session.session_id,
                    "current_stage": session.current_stage.value,
                    "dog_breed": session.dog_breed,
                    "health_condition": session.health_condition,
                    "created_at": session.created_at,
                    "is_active": session.is_active
                }
                for session in sessions
            ]
        except Exception as e:
            logger.error(f"❌ Error getting user sessions: {e}")
            return []

    async def end_session(self, session_id: str, user_id: str) -> bool:
        """End a chat session"""
        try:
            session = await self.chat_repo.get_session(session_id)
            if not session or session.user_id != user_id:
                return False
            
            return await self.chat_repo.end_session(session_id)
        except Exception as e:
            logger.error(f"❌ Error ending session: {e}")
            return False