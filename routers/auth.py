from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from modals.user import UserCreate, UserLogin, Token, UserResponse
from services.auth_service import AuthService
from database.connection import db_connection
from repositories.user_repository import UserRepository

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)

# --- Define Multiple Security Schemes ---

# 1. Standard OAuth2 Password Flow (for the login form)
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    description="Standard login with email and password."
)

# 2. API Key Header (for pasting a token directly)
api_key_scheme = APIKeyHeader(
    name="Authorization",
    description="Paste a Bearer token here for direct authorization.",
    auto_error=False # Set to False to allow either method
)

# --- Updated Dependency ---

async def get_current_active_user(
    password_token: str = Depends(oauth2_scheme),
    api_key_token: str = Depends(api_key_scheme)
) -> UserResponse:
    """
    Dependency that validates a user from either a password flow token
    or a directly provided API key (Bearer token).
    """
    # Prioritize the token from the direct paste (API Key) method
    token = api_key_token or password_token

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # The API Key header includes "Bearer ", so we remove it
    if token.startswith("Bearer "):
        token = token.split(" ")[1]

    user = auth_service.get_current_user(token)
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

# --- Initialize Services ---
user_repo = UserRepository(db_connection.get_collection("users"))
auth_service = AuthService(user_repo)


# --- Endpoints ---

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """Register a new user."""
    return auth_service.create_user(user)

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """Authenticate and get an access token."""
    return auth_service.authenticate_user(user.email, user.password)

