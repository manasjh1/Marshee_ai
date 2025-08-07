import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime
from routers import auth
from database.connection import db_connection
from routers import documents

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Dog Health System API...")
    print("📊 Database connection initialized")
    print("🔐 Authentication service ready")
    print("✅ API is ready to serve requests")
    yield
    # Shutdown
    print("📊 Closing database connections...")
    db_connection.close_connection()
    print("👋 Dog Health System API shutdown complete")

# Create FastAPI app with lifespan
app = FastAPI(
    title="Dog Health System API",
    description="Backend API for AI-powered dog health detection and consultation system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(documents.router)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dog Health System API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    """Complete system health check"""
    try:
        # Test database connection
        db_connection.client.admin.command('ping')
        db_status = "healthy"
    except Exception as e:
        print(f"❌ Database health check failed: {e}")
        db_status = "unhealthy"
    
    return {
        "api": "healthy",
        "database": db_status,
        "timestamp": str(datetime.utcnow())
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🌐 Starting server on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
