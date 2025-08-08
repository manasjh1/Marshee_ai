from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from modals.user import UserCreate, UserLogin, Token, UserResponse
from services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()
auth_service = AuthService()

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate):
    """
    Register a new user
    
    - **email**: Valid email address
    - **name**: Full name (2-100 characters)
    - **phone_number**: Phone number (10-15 digits)
    - **password**: Password (minimum 6 characters)
    """
    try:
        return auth_service.register_user(user)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=Token)
async def login(user: UserLogin):
    """
    Authenticate user and return access token
    
    - **email**: Registered email address
    - **password**: User password
    """
    try:
        return auth_service.authenticate_user(user.email, user.password)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Get current authenticated user information
    
    Requires: Bearer token in Authorization header
    """
    try:
        token = credentials.credentials
        return auth_service.get_current_user(token)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get user endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user (client should delete the token)
    
    Note: JWT tokens are stateless, so logout is handled client-side
    """
    try:
        # Verify token is valid
        token = credentials.credentials
        auth_service.verify_token(token)
        
        return {"message": "Successfully logged out"}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Logout endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/health")
async def auth_health_check():
    """Health check for authentication service"""
    return {
        "service": "Authentication",
        "status": "healthy",
        "timestamp": str(datetime.utcnow())
    }