import logging
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from services.auth_service import AuthService
from services.chat_service import ChatService
from modals.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chat System"])
security = HTTPBearer()
auth_service = AuthService()
chat_service = ChatService()

logger = logging.getLogger(__name__)

@router.post("/message", response_model=ChatResponse)
async def chat_message(
    request: ChatRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    🤖 Single Chat Endpoint - Handles All Conversation Stages
    
    **Stage Flow:**
    1. **Stage 1**: Welcome → Breed Detection (upload dog image)
    2. **Stage 2**: Choose between:
       - 🩺 Disease Detection (upload skin condition image)  
       - 💬 General Chat (personalized Q&A)
    
    **Request Examples:**
    
    **Start New Conversation:**
    ```json
    {
        "session_id": null
    }
    ```
    
    **Upload Dog Image (Stage 1):**
    ```json
    {
        "session_id": "session_123",
        "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    }
    ```
    
    **Select Option (Stage 2):**
    ```json
    {
        "session_id": "session_123", 
        "selected_option": "disease_detection"
    }
    ```
    
    **Send Text Message:**
    ```json
    {
        "session_id": "session_123",
        "message": "What should I feed my dog?"
    }
    ```
    
    **Upload Health Image:**
    ```json
    {
        "session_id": "session_123",
        "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
    }
    ```
    """
    try:
        # Authenticate user
        current_user = auth_service.get_current_user(credentials.credentials)
        
        # Process chat message through stage-based system
        response = await chat_service.process_chat_message(request, current_user.user_id)
        
        logger.info(f"💬 User {current_user.user_id} - Stage: {response.current_stage} - Session: {response.session_id}")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat system error. Please try again."
        )

@router.get("/session/{session_id}")
async def get_session_info(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current session information and conversation history"""
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        session_info = await chat_service.get_session_info(session_id, current_user.user_id)
        
        if not session_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        return session_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Session info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve session information"
        )

@router.get("/sessions")
async def get_user_sessions(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get all user's chat sessions"""
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        sessions = await chat_service.get_user_sessions(current_user.user_id)
        
        return {
            "user_id": current_user.user_id,
            "sessions": sessions,
            "total_sessions": len(sessions)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Get sessions error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not retrieve sessions"
        )

@router.delete("/session/{session_id}")
async def end_session(
    session_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """End a chat session"""
    try:
        current_user = auth_service.get_current_user(credentials.credentials)
        success = await chat_service.end_session(session_id, current_user.user_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or already ended"
            )
        
        return {
            "message": "Session ended successfully",
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ End session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not end session"
        )

@router.get("/health")
async def chat_health_check():
    """Chat system health check"""
    return {
        "service": "Chat System",
        "status": "healthy",
        "components": {
            "stage_management": "ready",
            "yolo_models": "ready", 
            "rag_system": "ready",
            "vector_database": "ready",
            "mongodb": "ready"
        }
    }