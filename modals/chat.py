import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

# --- Internal Data Structures ---

class ChatStage(str, Enum):
    STAGE_1_WELCOME = "welcome"
    STAGE_1_BREED_DETECTION = "breed_detection"
    STAGE_2_HEALTH_CHECK = "health_check"

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    current_stage: ChatStage = ChatStage.STAGE_1_WELCOME
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    message_type: MessageType
    content: str
    is_user_message: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# --- Centralized API Format (This was missing) ---

class ApiRequestData(BaseModel):
    """ The nested 'data' part of the request. """
    image_base64: Optional[str] = Field(None, description="Base64-encoded string of the uploaded image.")

class ApiRequest(BaseModel):
    """ The main request body for all chat interactions. """
    user_id: str = Field(description="The unique identifier for the user.")
    user_message: Optional[str] = Field(None, description="The text message from the user.")
    data: Optional[ApiRequestData] = None

class ApiResponse(BaseModel):
    """ The main response body for all chat interactions. """
    user_id: str = Field(description="The unique identifier for the user.")
    bot_response: str = Field(description="The text response from Marshee.")
    next_input_expected: Literal["image", "text", "choice"] = Field(description="Indicates what kind of input the bot expects next.")
    current_stage: ChatStage
