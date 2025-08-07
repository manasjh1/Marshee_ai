from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class ChatStage(str, Enum):
    """Conversation stages following your flow"""
    STAGE_1_WELCOME = "stage_1_welcome"           # Welcome, ask for dog image
    STAGE_1_BREED_DETECTION = "stage_1_breed"     # Processing breed detection
    STAGE_1_COMPLETE = "stage_1_complete"         # Breed detected, show results
    
    STAGE_2_OPTIONS = "stage_2_options"           # Show two options: Disease Detection or General Chat
    
    # Disease Detection Path
    STAGE_2A_DISEASE_REQUEST = "stage_2a_disease_request"   # Ask for skin condition image
    STAGE_2A_DISEASE_PROCESSING = "stage_2a_processing"     # Processing disease detection
    STAGE_2A_DISEASE_RESULT = "stage_2a_result"             # Show disease results with guidance
    STAGE_2A_FOLLOWUP = "stage_2a_followup"                 # Follow-up questions about condition
    
    # General Chat Path
    STAGE_2B_GENERAL_CHAT = "stage_2b_general_chat"        # Ongoing personalized conversation
    
    SESSION_COMPLETE = "session_complete"                   # Session ended

class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    SYSTEM = "system"
    OPTIONS = "options"
    DETECTION_RESULT = "detection_result"
    ERROR = "error"

class ChatOption(BaseModel):
    id: str
    text: str
    description: str
    icon: Optional[str] = None

class YOLODetectionResult(BaseModel):
    model_type: str  # "breed" or "disease"
    detected_class: str
    confidence: float
    text_result: str  # Your text response from YOLO
    additional_info: Optional[Dict[str, Any]] = {}
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    current_stage: ChatStage = ChatStage.STAGE_1_WELCOME
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Stage 1 Results
    breed_detection: Optional[YOLODetectionResult] = None
    dog_breed: Optional[str] = None
    breed_confidence: Optional[float] = None
    
    # Stage 2A Results (Disease Detection)
    disease_detection: Optional[YOLODetectionResult] = None
    health_condition: Optional[str] = None
    condition_confidence: Optional[float] = None
    
    # Conversation context for RAG
    conversation_history: List[str] = []
    user_preferences: Dict[str, Any] = {}
    
    # Session status
    is_active: bool = True
    completed_at: Optional[datetime] = None

class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    message_type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # For images
    image_data: Optional[str] = None  # Base64 encoded
    
    # For system messages with options
    options: Optional[List[ChatOption]] = None
    
    # For detection results
    detection_result: Optional[YOLODetectionResult] = None
    
    # Message direction
    is_user_message: bool = True
    
    # Additional metadata
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  # None = new session
    message: Optional[str] = None
    image_data: Optional[str] = None  # Base64 encoded image
    selected_option: Optional[str] = None  # For option selection

class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    current_stage: ChatStage
    response_type: MessageType
    content: str
    
    # For options
    options: Optional[List[ChatOption]] = None
    
    # For detection results
    detection_result: Optional[YOLODetectionResult] = None
    
    # Session info
    dog_breed: Optional[str] = None
    health_condition: Optional[str] = None
    
    # Next expected input
    next_input_expected: Optional[str] = None  # "text", "image", "option"
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# YOLO Service Models
class YOLORequest(BaseModel):
    image_data: str  # Base64 encoded
    model_type: str  # "breed" or "disease"
    session_id: str
    user_id: str

# RAG Service Models  
class RAGRequest(BaseModel):
    query: str
    session_id: str
    user_id: str
    dog_breed: Optional[str] = None
    health_condition: Optional[str] = None
    conversation_history: List[str] = []
    context_type: str = "general"  # "breed_info", "health_guidance", "general"

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: float
    response_type: str