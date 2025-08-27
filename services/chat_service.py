from modals.chat import ApiRequest, ApiResponse, ChatSession, ChatStage
from repositories.chat_repository import ChatRepository
from services.yolo_service import yolo_service
from services.rag_service import rag_service
from routers.auth import auth_service
from modals.user import UserResponse
from database.connection import db_connection
import base64

class ChatService:
    def __init__(self, chat_repo: ChatRepository, yolo_service, rag_service, auth_service):
        self.chat_repo = chat_repo
        self.yolo_service = yolo_service
        self.rag_service = rag_service
        self.auth_service = auth_service

    async def _get_or_create_session(self, user_id: str) -> ChatSession:
        """Finds an active session for a user or creates a new one."""
        session = await self.chat_repo.get_session_by_user_id(user_id)
        if not session:
            session = ChatSession(user_id=user_id)
            await self.chat_repo.create_session(session)
        return session

    async def _handle_welcome(self, session: ChatSession, user: UserResponse) -> ApiResponse:
        """Handles the initial welcome stage."""
        is_new_user = user.created_at == user.last_active
        
        if is_new_user:
            welcome_message = f"Welcome, {user.name}, to Marshee Pet Tech! To get started, please upload a photo of your dog."
        else:
            welcome_message = f"Welcome back, {user.name}! Please upload a photo of your dog to continue."
        
        return ApiResponse(
            user_id=session.user_id,
            bot_response=welcome_message,
            next_input_expected="image",
            current_stage=session.current_stage
        )

    async def _handle_breed_detection(self, session: ChatSession, request: ApiRequest) -> ApiResponse:
        """Handles the breed detection stage."""
        if not (request.data and request.data.image_base64):
            return ApiResponse(
                user_id=session.user_id,
                bot_response="Please upload an image to detect the breed.",
                next_input_expected="image",
                current_stage=session.current_stage
            )

        image_data = base64.b64decode(request.data.image_base64)
        detected_breed = self.yolo_service.detect_breed(image_data)
        
        session.current_stage = ChatStage.STAGE_2_HEALTH_CHECK
        await self.chat_repo.update_session(session)
        
        response_text = f"Breed detected: {detected_breed}. Now, let's check your dog's health. You can ask me anything about it."
        
        return ApiResponse(
            user_id=session.user_id,
            bot_response=response_text,
            next_input_expected="text",
            current_stage=session.current_stage
        )

    async def process_chat_message(self, request: ApiRequest) -> ApiResponse:
        """Main entry point to process a chat message."""
        user = self.auth_service.get_user_by_id(request.user_id)
        if not user:
            raise ValueError("User not found")

        session = await self._get_or_create_session(request.user_id)

        # If the user sends a message or image, move past the welcome stage.
        if session.current_stage == ChatStage.STAGE_1_WELCOME:
            if request.user_message or (request.data and request.data.image_base64):
                 return await self._handle_breed_detection(session, request)
            else:
                # This is the very first interaction, just get the welcome message.
                return await self._handle_welcome(session, user)

        # Logic for subsequent stages
        if session.current_stage == ChatStage.STAGE_2_HEALTH_CHECK:
            if not request.user_message:
                return ApiResponse(
                    user_id=session.user_id,
                    bot_response="Please ask a question about your dog's health.",
                    next_input_expected="text",
                    current_stage=session.current_stage
                )
            
            rag_response = self.rag_service.query_knowledge_base(request.user_message)
            return ApiResponse(
                user_id=session.user_id,
                bot_response=rag_response,
                next_input_expected="text",
                current_stage=session.current_stage
            )

        # Fallback for unimplemented stages
        return ApiResponse(
            user_id=session.user_id,
            bot_response="This chat stage is not yet implemented.",
            next_input_expected="text",
            current_stage=session.current_stage
        )

# --- Create Singleton Instance ---
chat_repo = ChatRepository(
    sessions_collection=db_connection.get_collection("chat_sessions"),
    messages_collection=db_connection.get_collection("chat_messages")
)

chat_service = ChatService(
    chat_repo=chat_repo,
    yolo_service=yolo_service,
    rag_service=rag_service,
    auth_service=auth_service
)
