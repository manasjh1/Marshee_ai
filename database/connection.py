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
            
            # Test connection
            self._client.admin.command('ping')
            print(f"✅ Successfully connected to MongoDB Atlas: {db_name}")
            
            # Create indexes
            self._create_indexes()
            
        except ConnectionFailure as e:
            print(f"❌ Failed to connect to MongoDB Atlas: {e}")
            raise
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise

    def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # Get collection names from environment
            users_collection = os.getenv("USERS_COLLECTION", "users")
            chat_sessions_collection = os.getenv("CHAT_SESSIONS_COLLECTION", "chat_sessions")
            breed_detections_collection = os.getenv("BREED_DETECTIONS_COLLECTION", "breed_detections")
            health_detections_collection = os.getenv("HEALTH_DETECTIONS_COLLECTION", "health_detections")
            user_contexts_collection = os.getenv("USER_CONTEXTS_COLLECTION", "user_contexts")
            dog_profiles_collection = os.getenv("DOG_PROFILES_COLLECTION", "dog_profiles")

            # Users collection indexes
            self._database[users_collection].create_index("email", unique=True)
            self._database[users_collection].create_index("user_id", unique=True)
            self._database[users_collection].create_index("phone_number")
            
            # Chat sessions indexes
            self._database[chat_sessions_collection].create_index("user_id")
            self._database[chat_sessions_collection].create_index("session_id", unique=True)
            self._database[chat_sessions_collection].create_index("created_at")
            
            # Breed detections indexes
            self._database[breed_detections_collection].create_index("user_id")
            self._database[breed_detections_collection].create_index("detection_id", unique=True)
            self._database[breed_detections_collection].create_index("created_at")
            
            # Health detections indexes
            self._database[health_detections_collection].create_index("user_id")
            self._database[health_detections_collection].create_index("detection_id", unique=True)
            self._database[health_detections_collection].create_index("created_at")
            
            # User contexts indexes
            self._database[user_contexts_collection].create_index("user_id", unique=True)
            self._database[user_contexts_collection].create_index("last_updated")
            
            # Dog profiles indexes
            self._database[dog_profiles_collection].create_index("user_id")
            self._database[dog_profiles_collection].create_index("dog_id", unique=True)
            
            print("✅ Database indexes created successfully for all collections")
        except Exception as e:
            print(f"⚠️ Warning: Could not create indexes: {e}")

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
    def breed_detections_collection(self):
        collection_name = os.getenv("BREED_DETECTIONS_COLLECTION", "breed_detections")
        return self.database[collection_name]

    @property
    def health_detections_collection(self):
        collection_name = os.getenv("HEALTH_DETECTIONS_COLLECTION", "health_detections")
        return self.database[collection_name]

    @property
    def user_contexts_collection(self):
        collection_name = os.getenv("USER_CONTEXTS_COLLECTION", "user_contexts")
        return self.database[collection_name]

    @property
    def dog_profiles_collection(self):
        collection_name = os.getenv("DOG_PROFILES_COLLECTION", "dog_profiles")
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
            print("📊 Database connection closed")

# Global database instance
db_connection = DatabaseConnection()
