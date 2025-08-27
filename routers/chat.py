from fastapi import APIRouter, Depends, HTTPException, status
# Use the new centralized models from the canvas
from modals.chat import ApiRequest, ApiResponse
from modals.user import UserResponse
from routers.auth import get_current_active_user
from services.chat_service import chat_service

router = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)

@router.post("/", response_model=ApiResponse)
async def process_chat(
    # The request body now expects the new ApiRequest format
    request: ApiRequest,
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Main endpoint for all chat interactions using the centralized user_id format.

    - You **must** provide the `user_id` in every request.
    - To start a chat, send only the `user_id`.
    - To send text, use `user_message`.
    - To send an image, use `data.image_base64`.
    """
    # Security check: Ensure the user_id in the token matches the one in the request body
    if request.user_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user_id in the request does not match the authenticated user."
        )

    try:
        # The service layer now handles the entire request object
        return await chat_service.process_chat_message(request=request)
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat message."
        )
