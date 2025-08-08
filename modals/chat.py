from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class ChatStage(str, Enum):
    """Conversation stages following your flow"""
    STAGE_1_WELCOME = "stage_1_welcome"           
    STAGE_1_BREED_DETECTION = "stage_1_breed"     
    STAGE_1_COMPLETE = "stage_1_complete"         
    STAGE_2_OPTIONS = "stage_2_options"           
    STAGE_2A_DISEASE_REQUEST = "stage_2a_disease_request"   
    STAGE_2A_DISEASE_PROCESSING = "stage_2a_processing"     
    STAGE_2A_DISEASE_RESULT = "stage_2a_result"             
    STAGE_2A_FOLLOWUP = "stage_2a_followup"                 
    STAGE_2B_GENERAL_CHAT = "stage_2b_general_chat"    
    SESSION_COMPLETE = "session_complete"                   

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
    model_type: str  
    detected_class: str
    confidence: float
    text_result: str  
    additional_info: Optional[Dict[str, Any]] = {}
    processing_time: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    current_stage: ChatStage = ChatStage.STAGE_1_WELCOME
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    breed_detection: Optional[YOLODetectionResult] = None
    dog_breed: Optional[str] = None
    breed_confidence: Optional[float] = None
    
    disease_detection: Optional[YOLODetectionResult] = None
    health_condition: Optional[str] = None
    condition_confidence: Optional[float] = None
    
    conversation_history: List[str] = []
    user_preferences: Dict[str, Any] = {}
    
    is_active: bool = True
    completed_at: Optional[datetime] = None

class ChatMessage(BaseModel):
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    message_type: MessageType
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    image_data: Optional[str] = None  
    options: Optional[List[ChatOption]] = None
    detection_result: Optional[YOLODetectionResult] = None
    is_user_message: bool = True
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    session_id: Optional[str] = None  
    message: Optional[str] = None
    image_data: Optional[str] = None  
    selected_option: Optional[str] = None  

class ChatResponse(BaseModel):
    session_id: str
    message_id: str
    current_stage: ChatStage
    response_type: MessageType
    content: str
    
    options: Optional[List[ChatOption]] = None
    
    detection_result: Optional[YOLODetectionResult] = None
    
    dog_breed: Optional[str] = None
    health_condition: Optional[str] = None
    
    next_input_expected: Optional[str] = None  
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class YOLORequest(BaseModel):
    image_data: str  
    model_type: str  
    session_id: str
    user_id: str
  
class RAGRequest(BaseModel):
    query: str
    session_id: str
    user_id: str
    dog_breed: Optional[str] = None
    health_condition: Optional[str] = None
    conversation_history: List[str] = []
    context_type: str = "general"

class RAGResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]] = []
    confidence: float
    response_type: str