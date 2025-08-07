import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from modals.user import UserCreate, UserResponse, Token, TokenData
from repositories.user_repository import UserRepository
import warnings
warnings.filterwarnings("ignore",
                        category=DeprecationWarning)

from passlib.context import CryptContext
class AuthService:
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.user_repo = UserRepository()
        self.secret_key = os.getenv("JWT_SECRET_KEY")
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        self.access_token_expire_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    def hash_password(self, password: str) -> str:
        """Hash a password using bcrypt"""
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return self.pwd_context.verify(plain_password, hashed_password)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        try:
            to_encode = data.copy()
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({"exp": expire, "iat": datetime.utcnow()})
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
            return encoded_jwt
        except Exception as e:
            print(f"❌ Error creating access token: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not create access token"
            )

    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            email: str = payload.get("sub")
            if email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            token_data = TokenData(email=email)
            return token_data
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def register_user(self, user_data: UserCreate) -> Token:
        """Register a new user"""
        try:
            # Check if email already exists
            if self.user_repo.check_email_exists(user_data.email):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )

            # Check if phone number already exists
            if self.user_repo.check_phone_exists(user_data.phone_number):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Phone number already registered"
                )

            # Hash password and create user
            password_hash = self.hash_password(user_data.password)
            user_id = self.user_repo.create_user(user_data, password_hash)

            # Get created user
            user_doc = self.user_repo.get_user_by_id(user_id)
            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="User creation failed"
                )

            # Create user response
            user_response = UserResponse(
                user_id=user_doc["user_id"],
                email=user_doc["email"],
                name=user_doc["name"],
                phone_number=user_doc["phone_number"],
                created_at=user_doc["created_at"],
                last_active=user_doc["last_active"],
                is_active=user_doc["is_active"]
            )

            # Create access token
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token = self.create_access_token(
                data={"sub": user_data.email}, expires_delta=access_token_expires
            )

            return Token(
                access_token=access_token,
                token_type="bearer",
                user_info=user_response
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Registration error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Registration failed"
            )

    def authenticate_user(self, email: str, password: str) -> Token:
        """Authenticate user and return token"""
        try:
            user_doc = self.user_repo.get_user_by_email(email)
            
            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )

            if not user_doc["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )

            # Check password
            if not self.verify_password(password, user_doc["password_hash"]):
                self.user_repo.increment_login_attempts(email)
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid email or password"
                )

            # Reset login attempts on successful login
            self.user_repo.reset_login_attempts(email)
            self.user_repo.update_last_active(user_doc["user_id"])

            # Create user response
            user_response = UserResponse(
                user_id=user_doc["user_id"],
                email=user_doc["email"],
                name=user_doc["name"],
                phone_number=user_doc["phone_number"],
                created_at=user_doc["created_at"],
                last_active=datetime.utcnow(),
                is_active=user_doc["is_active"]
            )

            # Create access token
            access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
            access_token = self.create_access_token(
                data={"sub": email}, expires_delta=access_token_expires
            )

            return Token(
                access_token=access_token,
                token_type="bearer",
                user_info=user_response
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )

    def get_current_user(self, token: str) -> UserResponse:
        """Get current user from token"""
        try:
            token_data = self.verify_token(token)
            user_doc = self.user_repo.get_user_by_email(token_data.email)
            
            if not user_doc:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )

            if not user_doc["is_active"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account is deactivated"
                )

            # Update last active
            self.user_repo.update_last_active(user_doc["user_id"])

            return UserResponse(
                user_id=user_doc["user_id"],
                email=user_doc["email"],
                name=user_doc["name"],
                phone_number=user_doc["phone_number"],
                created_at=user_doc["created_at"],
                last_active=datetime.utcnow(),
                is_active=user_doc["is_active"]
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Get current user error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
