import uuid
from datetime import datetime
from typing import Optional, Dict
from pymongo.errors import DuplicateKeyError
from database.connection import db_connection
from modals.user import UserCreate, UserResponse

class UserRepository:
    def __init__(self):
        self.users_collection = db_connection.users_collection

    def create_user(self, user_data: UserCreate, password_hash: str) -> str:
        """Create a new user in the database"""
        try:
            user_id = str(uuid.uuid4())
            user_doc = {
                "user_id": user_id,
                "email": user_data.email.lower(),
                "name": user_data.name,
                "phone_number": user_data.phone_number,
                "password_hash": password_hash,
                "created_at": datetime.utcnow(),
                "last_active": datetime.utcnow(),
                "is_active": True,
                "login_attempts": 0,
                "account_locked_until": None
            }
            
            result = self.users_collection.insert_one(user_doc)
            if result.inserted_id:
                print(f"User created successfully in collection '{self.users_collection.name}': {user_data.email}")
                return user_id
            else:
                raise Exception("Failed to insert user")
                
        except DuplicateKeyError:
            raise ValueError("Email already registered")
        except Exception as e:
            print(f"Error creating user: {e}")
            raise

    def get_user_by_email(self, email: str) -> Optional[Dict]:
        """Retrieve user by email"""
        try:
            return self.users_collection.find_one({"email": email.lower()})
        except Exception as e:
            print(f"Error retrieving user by email: {e}")
            return None

    def get_user_by_id(self, user_id: str) -> Optional[Dict]:
        """Retrieve user by user_id"""
        try:
            return self.users_collection.find_one({"user_id": user_id})
        except Exception as e:
            print(f"Error retrieving user by ID: {e}")
            return None

    def update_last_active(self, user_id: str):
        """Update user's last active timestamp"""
        try:
            self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"last_active": datetime.utcnow()}}
            )
        except Exception as e:
            print(f"Error updating last active: {e}")

    def check_email_exists(self, email: str) -> bool:
        """Check if email already exists"""
        try:
            result = self.users_collection.find_one({"email": email.lower()})
            return result is not None
        except Exception as e:
            print(f"Error checking email existence: {e}")
            return False

    def check_phone_exists(self, phone_number: str) -> bool:
        """Check if phone number already exists"""
        try:
            result = self.users_collection.find_one({"phone_number": phone_number})
            return result is not None
        except Exception as e:
            print(f"Error checking phone existence: {e}")
            return False

    def increment_login_attempts(self, email: str):
        """Increment failed login attempts"""
        try:
            self.users_collection.update_one(
                {"email": email.lower()},
                {"$inc": {"login_attempts": 1}}
            )
        except Exception as e:
            print(f"Error incrementing login attempts: {e}")

    def reset_login_attempts(self, email: str):
        """Reset failed login attempts"""
        try:
            self.users_collection.update_one(
                {"email": email.lower()},
                {"$set": {"login_attempts": 0, "account_locked_until": None}}
            )
        except Exception as e:
            print(f"Error resetting login attempts: {e}")

    def get_user_stats(self) -> Dict:
        """Get user statistics"""
        try:
            total_users = self.users_collection.count_documents({})
            active_users = self.users_collection.count_documents({"is_active": True})
            inactive_users = total_users - active_users
            
            return {
                "total_users": total_users,
                "active_users": active_users,
                "inactive_users": inactive_users
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {"total_users": 0, "active_users": 0, "inactive_users": 0}

    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user account"""
        try:
            result = self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"is_active": False, "deactivated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error deactivating user: {e}")
            return False

    def activate_user(self, user_id: str) -> bool:
        """Activate a user account"""
        try:
            result = self.users_collection.update_one(
                {"user_id": user_id},
                {"$set": {"is_active": True}, "$unset": {"deactivated_at": ""}}
            )
            return result.modified_count > 0
        except Exception as e:
            print(f"Error activating user: {e}")
            return False