from datetime import datetime
from pymongo.collection import Collection
from bson import ObjectId

class UserRepository:
    """
    Handles all database operations related to users.
    """
    def __init__(self, collection: Collection):
        """
        Initializes the repository with a specific MongoDB collection.

        Args:
            collection: The PyMongo collection object for users.
        """
        self.collection = collection

    def create_user(self, name: str, email: str, password_hash: str, phone_number: str = None) -> str:
        """Creates a new user in the database."""
        user_doc = {
            "user_id": str(ObjectId()),
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "phone_number": phone_number,
            "created_at": datetime.utcnow(),
            "last_active": datetime.utcnow(),
            "is_active": True,
            "login_attempts": 0
        }
        result = self.collection.insert_one(user_doc)
        return user_doc["user_id"]

    def get_user_by_email(self, email: str):
        """Finds a user by their email address."""
        return self.collection.find_one({"email": email})

    def get_user_by_id(self, user_id: str):
        """Finds a user by their unique user ID."""
        return self.collection.find_one({"user_id": user_id})

    def update_last_active(self, user_id: str):
        """Updates the last_active timestamp for a user."""
        self.collection.update_one(
            {"user_id": user_id},
            {"$set": {"last_active": datetime.utcnow()}}
        )

    def increment_login_attempts(self, email: str):
        """Increments the failed login attempt counter for a user."""
        self.collection.update_one(
            {"email": email},
            {"$inc": {"login_attempts": 1}}
        )

    def reset_login_attempts(self, email: str):
        """Resets the failed login attempt counter for a user."""
        self.collection.update_one(
            {"email": email},
            {"$set": {"login_attempts": 0}}
        )