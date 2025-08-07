import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from modals.user import UserCreate, UserLogin, Token, UserResponse  # Changed to 'modals'
    print("✅ Modals imported successfully")
except ImportError as e:
    print(f"❌ Modals import failed: {e}")

try:
    from database.connection import db_connection
    print("✅ Database connection imported successfully")
except ImportError as e:
    print(f"❌ Database import failed: {e}")

try:
    from services.auth_service import AuthService
    print("✅ Auth service imported successfully")
except ImportError as e:
    print(f"❌ Auth service import failed: {e}")

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    test_hash = pwd_context.hash("test123")
    print("✅ Password hashing works")
except Exception as e:
    print(f"❌ Password hashing failed: {e}")
