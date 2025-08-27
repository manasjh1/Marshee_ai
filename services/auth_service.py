# Marshee_model/services/auth_service.py
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from modals.user import UserCreate, UserResponse, Token, TokenData
from repositories.user_repository import UserRepository

class AuthService:
    def __init__(self, user_repo: UserRepository):
        self.user_repo = user_repo
        self.secret_key = os.getenv("SECRET_KEY")
        self.algorithm = os.getenv("ALGORITHM")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

    def verify_password(self, plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password):
        return self.pwd_context.hash(password)

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def create_user(self, user: UserCreate) -> UserResponse:
        """Create a new user"""
        # Check if user already exists
        if self.user_repo.get_user_by_email(user.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # Hash password
        hashed_password = self.get_password_hash(user.password)

        # Create user in database
        user_id = self.user_repo.create_user(
            name=user.name,
            email=user.email,
            password_hash=hashed_password,
            phone_number=user.phone_number
        )

        # Return user response
        return UserResponse(
            user_id=user_id,
            email=user.email,
            name=user.name,
            phone_number=user.phone_number,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            is_active=True
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

            # Determine if it's a new user
            is_new = user_doc["created_at"] == user_doc["last_active"]

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
                user_info=user_response,
                is_new_user=is_new
            )

        except HTTPException:
            raise
        except Exception as e:
            print(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed"
            )

    def get_current_user(self, token: str) -> UserResponse:
        """Get current user from token"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            email: str = payload.get("sub")
            if email is None:
                raise credentials_exception
            token_data = TokenData(email=email)
        except JWTError:
            raise credentials_exception

        user_doc = self.user_repo.get_user_by_email(token_data.email)
        if user_doc is None:
            raise credentials_exception

        return UserResponse(
            user_id=user_doc["user_id"],
            email=user_doc["email"],
            name=user_doc["name"],
            phone_number=user_doc["phone_number"],
            created_at=user_doc["created_at"],
            last_active=user_doc["last_active"],
            is_active=user_doc["is_active"]
        )