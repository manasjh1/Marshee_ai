import os
import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class DatabaseConnection:
    """
    Singleton class for managing the connection to the MongoDB database.
    Ensures that only one connection is established and provides
    methods to access database collections.
    """
    _instance = None
    _client = None
    _db = None

    def __new__(cls):
        """
        Ensures that only one instance of the DatabaseConnection is created.
        """
        if cls._instance is None:
            cls._instance = super(DatabaseConnection, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initializes the database connection if it doesn't already exist.
        """
        if self._client is None:
            self.connect()

    def connect(self):
        """
        Establishes a connection to the MongoDB Atlas cluster using
        credentials from environment variables.
        """
        try:
            mongo_uri = os.getenv("MONGO_URI")
            db_name = os.getenv("DB_NAME")
            
            if not mongo_uri or not db_name:
                raise ValueError("MONGO_URI and DB_NAME must be set in the environment variables")
            
            # Connect to the MongoDB client using certifi for SSL validation
            self._client = MongoClient(mongo_uri, tlsCAFile=certifi.where())
            self._db = self._client[db_name]
            
            # Ping the server to confirm a successful connection
            self._client.admin.command('ping')
            print(f"Successfully connected to MongoDB Atlas: {db_name}")
            
            self._create_indexes()
            
        except ConnectionFailure as e:
            print(f"Failed to connect to MongoDB Atlas: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during database connection: {e}")
            raise

    def _create_indexes(self):
        """
        Creates indexes on various collections to improve query performance.
        This is a good practice for any production database.
        """
        try:
            # User collection indexes
            self.get_collection("users").create_index("email", unique=True)
            self.get_collection("users").create_index("user_id", unique=True)
            
            # Chat sessions collection indexes
            self.get_collection("chat_sessions").create_index("user_id")
            self.get_collection("chat_sessions").create_index("session_id", unique=True)
            
            # Chat messages collection indexes
            self.get_collection("chat_messages").create_index("session_id")
            self.get_collection("chat_messages").create_index("user_id")

            print("Database indexes checked/created successfully.")
        except Exception as e:
            # It's okay if index creation fails (e.g., due to permissions),
            # but we should log it as a warning.
            print(f"Warning: Could not create or verify indexes: {e}")

    def get_collection(self, collection_name: str):
        """
        Provides access to a specific collection within the database.
        
        Args:
            collection_name: The name of the collection to access.
        
        Returns:
            A PyMongo Collection object.
        """
        if self._db is None:
            self.connect()
        return self._db[collection_name]

    def close_connection(self):
        """
        Closes the active MongoDB connection.
        """
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            print("MongoDB connection closed.")

# Create a single, globally accessible instance of the database connection
db_connection = DatabaseConnection()
