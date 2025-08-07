import uuid
from datetime import datetime
from typing import List, Dict, Optional
from pymongo.errors import DuplicateKeyError
from database.connection import db_connection
from modals.chat import ChatSession, ChatMessage, ChatStage, MessageType

class ChatRepository:
    def __init__(self):
        self.chat_sessions_collection = db_connection.chat_sessions_collection
        self.chat_messages_collection = db_connection.chat_messages_collection
        self.yolo_detections_collection = db_connection.yolo_detections_collection

    async def create_session(self, session: ChatSession) -> str:
        """Create a new chat session"""
        try:
            session_doc = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "current_stage": session.current_stage.value,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "breed_detection": session.breed_detection.dict() if session.breed_detection else None,
                "dog_breed": session.dog_breed,
                "breed_confidence": session.breed_confidence,
                "disease_detection": session.disease_detection.dict() if session.disease_detection else None,
                "health_condition": session.health_condition,
                "condition_confidence": session.condition_confidence,
                "conversation_history": session.conversation_history,
                "user_preferences": session.user_preferences,
                "is_active": session.is_active,
                "completed_at": session.completed_at
            }
            
            result = self.chat_sessions_collection.insert_one(session_doc)
            if result.inserted_id:
                print(f"✅ Created chat session: {session.session_id}")
                return session.session_id
            else:
                raise Exception("Failed to create session")
                
        except Exception as e:
            print(f"❌ Error creating session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get chat session by ID"""
        try:
            session_doc = self.chat_sessions_collection.find_one({"session_id": session_id})
            
            if session_doc:
                # Convert detection results back to objects if they exist
                breed_detection = None
                if session_doc.get("breed_detection"):
                    from modals.chat import YOLODetectionResult
                    breed_detection = YOLODetectionResult(**session_doc["breed_detection"])
                
                disease_detection = None
                if session_doc.get("disease_detection"):
                    from modals.chat import YOLODetectionResult
                    disease_detection = YOLODetectionResult(**session_doc["disease_detection"])
                
                return ChatSession(
                    session_id=session_doc["session_id"],
                    user_id=session_doc["user_id"],
                    current_stage=ChatStage(session_doc["current_stage"]),
                    created_at=session_doc["created_at"],
                    updated_at=session_doc["updated_at"],
                    breed_detection=breed_detection,
                    dog_breed=session_doc.get("dog_breed"),
                    breed_confidence=session_doc.get("breed_confidence"),
                    disease_detection=disease_detection,
                    health_condition=session_doc.get("health_condition"),
                    condition_confidence=session_doc.get("condition_confidence"),
                    conversation_history=session_doc.get("conversation_history", []),
                    user_preferences=session_doc.get("user_preferences", {}),
                    is_active=session_doc.get("is_active", True),
                    completed_at=session_doc.get("completed_at")
                )
            return None
            
        except Exception as e:
            print(f"❌ Error getting session: {e}")
            return None

    async def update_session(self, session: ChatSession) -> bool:
        """Update chat session"""
        try:
            update_doc = {
                "current_stage": session.current_stage.value,
                "updated_at": datetime.utcnow(),
                "breed_detection": session.breed_detection.dict() if session.breed_detection else None,
                "dog_breed": session.dog_breed,
                "breed_confidence": session.breed_confidence,
                "disease_detection": session.disease_detection.dict() if session.disease_detection else None,
                "health_condition": session.health_condition,
                "condition_confidence": session.condition_confidence,
                "conversation_history": session.conversation_history,
                "user_preferences": session.user_preferences,
                "is_active": session.is_active,
                "completed_at": session.completed_at
            }
            
            result = self.chat_sessions_collection.update_one(
                {"session_id": session.session_id},
                {"$set": update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error updating session: {e}")
            return False

    async def save_message(self, message: ChatMessage) -> str:
        """Save chat message"""
        try:
            message_doc = {
                "message_id": message.message_id,
                "session_id": message.session_id,
                "user_id": message.user_id,
                "message_type": message.message_type.value,
                "content": message.content,
                "timestamp": message.timestamp,
                "image_data": message.image_data,
                "options": [opt.dict() for opt in message.options] if message.options else None,
                "detection_result": message.detection_result.dict() if message.detection_result else None,
                "is_user_message": message.is_user_message,
                "metadata": message.metadata
            }
            
            result = self.chat_messages_collection.insert_one(message_doc)
            if result.inserted_id:
                return message.message_id
            else:
                raise Exception("Failed to save message")
                
        except Exception as e:
            print(f"❌ Error saving message: {e}")
            raise

    async def get_session_messages(self, session_id: str, limit: int = 50) -> List[ChatMessage]:
        """Get messages for a session"""
        try:
            messages_cursor = self.chat_messages_collection.find(
                {"session_id": session_id}
            ).sort("timestamp", 1).limit(limit)
            
            messages = []
            for msg_doc in messages_cursor:
                # Convert options back to objects if they exist
                options = None
                if msg_doc.get("options"):
                    from modals.chat import ChatOption
                    options = [ChatOption(**opt) for opt in msg_doc["options"]]
                
                # Convert detection result back to object if it exists
                detection_result = None
                if msg_doc.get("detection_result"):
                    from modals.chat import YOLODetectionResult
                    detection_result = YOLODetectionResult(**msg_doc["detection_result"])
                
                message = ChatMessage(
                    message_id=msg_doc["message_id"],
                    session_id=msg_doc["session_id"],
                    user_id=msg_doc["user_id"],
                    message_type=MessageType(msg_doc["message_type"]),
                    content=msg_doc["content"],
                    timestamp=msg_doc["timestamp"],
                    image_data=msg_doc.get("image_data"),
                    options=options,
                    detection_result=detection_result,
                    is_user_message=msg_doc.get("is_user_message", True),
                    metadata=msg_doc.get("metadata", {})
                )
                messages.append(message)
            
            return messages
            
        except Exception as e:
            print(f"❌ Error getting messages: {e}")
            return []

    async def save_detection_result(self, detection_data: Dict) -> str:
        """Save YOLO detection result"""
        try:
            detection_id = str(uuid.uuid4())
            detection_doc = {
                "detection_id": detection_id,
                "session_id": detection_data.get("session_id"),
                "user_id": detection_data.get("user_id"),
                "model_type": detection_data.get("model_type"),
                "detected_class": detection_data.get("detected_class"),
                "confidence": detection_data.get("confidence"),
                "additional_info": detection_data.get("additional_info"),
                "processing_time": detection_data.get("processing_time"),
                "timestamp": datetime.utcnow()
            }
            
            result = self.yolo_detections_collection.insert_one(detection_doc)
            if result.inserted_id:
                return detection_id
            else:
                raise Exception("Failed to save detection")
                
        except Exception as e:
            print(f"❌ Error saving detection: {e}")
            raise

    async def get_user_sessions(self, user_id: str, active_only: bool = True) -> List[ChatSession]:
        """Get all sessions for a user"""
        try:
            query = {"user_id": user_id}
            if active_only:
                query["is_active"] = True
            
            sessions_cursor = self.chat_sessions_collection.find(query).sort("updated_at", -1)
            
            sessions = []
            for session_doc in sessions_cursor:
                # Convert detection results back to objects if they exist
                breed_detection = None
                if session_doc.get("breed_detection"):
                    from modals.chat import YOLODetectionResult
                    breed_detection = YOLODetectionResult(**session_doc["breed_detection"])
                
                disease_detection = None
                if session_doc.get("disease_detection"):
                    from modals.chat import YOLODetectionResult
                    disease_detection = YOLODetectionResult(**session_doc["disease_detection"])
                
                session = ChatSession(
                    session_id=session_doc["session_id"],
                    user_id=session_doc["user_id"],
                    current_stage=ChatStage(session_doc["current_stage"]),
                    created_at=session_doc["created_at"],
                    updated_at=session_doc["updated_at"],
                    breed_detection=breed_detection,
                    dog_breed=session_doc.get("dog_breed"),
                    breed_confidence=session_doc.get("breed_confidence"),
                    disease_detection=disease_detection,
                    health_condition=session_doc.get("health_condition"),
                    condition_confidence=session_doc.get("condition_confidence"),
                    conversation_history=session_doc.get("conversation_history", []),
                    user_preferences=session_doc.get("user_preferences", {}),
                    is_active=session_doc.get("is_active", True),
                    completed_at=session_doc.get("completed_at")
                )
                sessions.append(session)
            
            return sessions
            
        except Exception as e:
            print(f"❌ Error getting user sessions: {e}")
            return []

    async def end_session(self, session_id: str) -> bool:
        """Mark session as completed"""
        try:
            result = self.chat_sessions_collection.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "is_active": False,
                        "completed_at": datetime.utcnow(),
                        "current_stage": ChatStage.SESSION_COMPLETE.value
                    }
                }
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            print(f"❌ Error ending session: {e}")
            return False

    async def get_session_stats(self) -> Dict:
        """Get session statistics"""
        try:
            total_sessions = self.chat_sessions_collection.count_documents({})
            active_sessions = self.chat_sessions_collection.count_documents({"is_active": True})
            completed_sessions = self.chat_sessions_collection.count_documents({"is_active": False})
            
            # Get stage distribution
            stage_pipeline = [
                {"$group": {"_id": "$current_stage", "count": {"$sum": 1}}}
            ]
            stage_stats = list(self.chat_sessions_collection.aggregate(stage_pipeline))
            
            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "completed_sessions": completed_sessions,
                "stage_distribution": {stat["_id"]: stat["count"] for stat in stage_stats}
            }
            
        except Exception as e:
            print(f"❌ Error getting session stats: {e}")
            return {"error": str(e)}