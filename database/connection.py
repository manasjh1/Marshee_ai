import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnection:
    _instance = None
    _client = None
    _database = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._client is None:
            self.connect()

    def connect(self):
        try:
            mongo_uri = os.getenv("MONGO_URI")
            db_name = os.getenv("MONGO_DB_NAME")
            
            if not mongo_uri or not db_name:
                raise ValueError("Missing MongoDB configuration in environment variables")
            
            self._client = MongoClient(mongo_uri)
            self._database = self._client[db_name]
            
            self._client.admin.command('ping')
            print(f"Successfully connected to MongoDB Atlas: {db_name}")
            
            self._create_indexes()
            
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB Atlas: {e}")
            raise
        except Exception as e:
            print(f"Database connection error: {e}")
            raise

    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            users_collection = os.getenv("USERS_COLLECTION", "users")
            chat_sessions_collection = os.getenv("CHAT_SESSIONS_COLLECTION", "chat_sessions")
            chat_messages_collection = os.getenv("CHAT_MESSAGES_COLLECTION", "chat_messages")
            yolo_detections_collection = os.getenv("YOLO_DETECTIONS_COLLECTION", "yolo_detections")

            self._database[users_collection].create_index("email", unique=True)
            self._database[users_collection].create_index("user_id", unique=True)
            self._database[users_collection].create_index("phone_number")
            
            self._database[chat_sessions_collection].create_index("user_id")
            self._database[chat_sessions_collection].create_index("session_id", unique=True)
            self._database[chat_sessions_collection].create_index("created_at")
            
            self._database[chat_messages_collection].create_index("session_id")
            self._database[chat_messages_collection].create_index("user_id")
            self._database[chat_messages_collection].create_index("timestamp")
            
            self._database[yolo_detections_collection].create_index("session_id")
            self._database[yolo_detections_collection].create_index("user_id")
            self._database[yolo_detections_collection].create_index("model_type")

            print("Database indexes created successfully for all collections")
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")

    @property
    def database(self):
        if self._database is None:
            self.connect()
        return self._database

    @property
    def users_collection(self):
        collection_name = os.getenv("USERS_COLLECTION", "users")
        return self.database[collection_name]

    @property
    def chat_sessions_collection(self):
        collection_name = os.getenv("CHAT_SESSIONS_COLLECTION", "chat_sessions")
        return self.database[collection_name]

    @property
    def chat_messages_collection(self):
        collection_name = os.getenv("CHAT_MESSAGES_COLLECTION", "chat_messages")
        return self.database[collection_name]

    @property
    def yolo_detections_collection(self):
        collection_name = os.getenv("YOLO_DETECTIONS_COLLECTION", "yolo_detections")
        return self.database[collection_name]

    @property
    def client(self):
        if self._client is None:
            self.connect()
        return self._client

    def close_connection(self):
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            print("Database connection closed")

db_connection = DatabaseConnection()