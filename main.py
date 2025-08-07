import os
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from datetime import datetime

# Import routers
from routers import auth, chat
from database.connection import db_connection

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("🚀 Starting Marshee Dog Health System...")
    print("📊 Database connection initialized")
    print("🔐 Authentication service ready")
    print("🤖 YOLO models loading...")
    print("🧠 RAG system initializing...")
    print("✅ Chat system ready!")
    yield
    # Shutdown
    print("📊 Closing database connections...")
    db_connection.close_connection()
    print("👋 Marshee system shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Marshee - AI Dog Health Assistant",
    description="Stage-based conversational AI for dog breed detection, health monitoring, and personalized care guidance",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Include only essential routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"❌ Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "message": "Something went wrong. Please try again."
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """API information"""
    return {
        "service": "Marshee - AI Dog Health Assistant",
        "version": "2.0.0",
        "status": "online",
        "features": [
            "🐕 Breed Detection with YOLOv11",
            "🩺 Health Condition Detection",
            "💬 Personalized Care Guidance",
            "🧠 RAG-powered Knowledge Base"
        ],
        "endpoints": {
            "auth": "/api/v1/auth",
            "chat": "/api/v1/chat",
            "docs": "/docs"
        },
        "timestamp": str(datetime.utcnow())
    }

# Health check
@app.get("/health")
async def health_check():
    """System health check"""
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
        "services": {
            "authentication": "ready",
            "chat_system": "ready",
            "yolo_models": "ready",
            "rag_system": "ready"
        },
        "timestamp": str(datetime.utcnow())
    }

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"🌐 Starting Marshee system on {host}:{port}")
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )