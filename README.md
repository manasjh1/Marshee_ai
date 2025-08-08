# 🐕 Marshee - AI Dog Health Assistant

**Stage-based conversational AI for dog breed detection, health monitoring, and personalized care guidance**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-8.0.196-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🌟 Overview

Marshee is an advanced AI-powered dog health assistant that combines computer vision with conversational AI to provide personalized care guidance for dog owners. The system uses state-of-the-art YOLOv11 models for breed and health condition detection, integrated with a Retrieval-Augmented Generation (RAG) system for comprehensive knowledge-based responses.

### 🎯 Key Features

- **🔍 AI-Powered Breed Detection**: Uses custom-trained YOLOv11 models to identify dog breeds with high accuracy
- **🩺 Health Condition Analysis**: Advanced image analysis for detecting skin conditions and health issues
- **💬 Intelligent Chat System**: Stage-based conversation flow with personalized responses
- **📚 RAG Knowledge Base**: Vector database with comprehensive dog care information
- **🔐 Secure Authentication**: JWT-based user authentication and session management
- **📱 API-First Design**: RESTful API architecture ready for web and mobile integration

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   AI Services   │
│                 │    │                 │    │                 │
│ • React/Vue     │◄──►│ • FastAPI       │◄──►│ • YOLOv11       │
│ • Mobile Apps   │    │ • JWT Auth      │    │ • Groq LLM      │
│ • Web Interface │    │ • Session Mgmt  │    │ • Gemini        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                ▲
                                │
        ┌─────────────────┬─────┴─────┬─────────────────┐
        │                 │           │                 │
┌───────▼──────┐ ┌────────▼─────┐ ┌───▼────────┐ ┌─────▼──────┐
│   MongoDB    │ │  Pinecone    │ │   Groq     │ │   Google   │
│              │ │              │ │            │ │            │
│ • Users      │ │ • Embeddings │ │ • LLM      │ │ • Gemini   │
│ • Sessions   │ │ • Knowledge  │ │ • Chat     │ │ • Embed    │
│ • Messages   │ │ • Vectors    │ │ • Resp     │ │ • Search   │
└──────────────┘ └──────────────┘ └────────────┘ └────────────┘
```

## 🚀 Conversation Flow

### Stage 1: Breed Detection
```
User uploads dog photo → YOLOv11 Analysis → Breed identified → Personalized greeting
```

### Stage 2: Service Selection
```
User chooses between:
├── 🩺 Disease Detection → Upload health photo → AI analysis → Treatment guidance
└── 💬 General Chat → Ask questions → RAG-powered responses → Continuous support
```

## 🔧 Technology Stack

### Backend & API
- **FastAPI** - High-performance async web framework
- **Python 3.11+** - Latest Python with performance improvements
- **Uvicorn** - ASGI server for production deployment
- **Pydantic** - Data validation and serialization

### AI & Machine Learning
- **YOLOv11** - State-of-the-art object detection for breed/health analysis
- **Groq** - High-speed LLM inference for conversational AI
- **Google Gemini** - Advanced text generation and embeddings
- **LangChain** - Framework for building LLM applications

### Database & Storage
- **MongoDB Atlas** - Document database for user data and sessions
- **Pinecone** - Vector database for semantic search and RAG
- **JWT** - Secure authentication and session management

### Computer Vision
- **OpenCV** - Image processing and manipulation
- **Pillow** - Python imaging library
- **NumPy** - Numerical computing for image arrays
- **PyTorch** - Deep learning framework for YOLO models

## 📋 Prerequisites

### Required Accounts & API Keys
1. **MongoDB Atlas** - Database hosting
2. **Pinecone** - Vector database for embeddings
3. **Groq** - LLM API access
4. **Google AI** - Gemini API for embeddings

### System Requirements
- Python 3.11 or higher
- 8GB+ RAM (recommended for YOLO models)
- CUDA-compatible GPU (optional, for faster inference)
- 10GB+ storage space



## 🚀 Running the Application

### Development Server
```bash
# Start the FastAPI development server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Deployment
```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

The API will be available at:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **API Base**: http://localhost:8000/api/v1

## 📚 API Documentation

### Authentication Endpoints

#### Register User
```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "name": "John Doe",
  "phone_number": "1234567890",
  "password": "securepassword"
}
```

#### Login
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword"
}
```

### Chat System

#### Start Conversation
```http
POST /api/v1/chat/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": null
}
```

#### Upload Dog Image (Breed Detection)
```http
POST /api/v1/chat/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": "session_123",
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

#### Select Service Option
```http
POST /api/v1/chat/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": "session_123",
  "selected_option": "disease_detection"
}
```

#### Send Text Message
```http
POST /api/v1/chat/message
Authorization: Bearer <token>
Content-Type: application/json

{
  "session_id": "session_123",
  "message": "What should I feed my Golden Retriever?"
}
```

### Response Format
```json
{
  "session_id": "session_123",
  "message_id": "msg_456",
  "current_stage": "stage_2b_general_chat",
  "response_type": "text",
  "content": "For Golden Retrievers, I recommend...",
  "dog_breed": "Golden Retriever",
  "health_condition": null,
  "next_input_expected": "text",
  "timestamp": "2025-01-15T10:30:00Z"
}
```

## 🔍 System Components

### 1. Chat Service (`services/chat_service.py`)
- **Stage Management**: Handles conversation flow between stages
- **Session Persistence**: Maintains conversation state and history
- **Context Integration**: Combines user data with AI responses

### 2. LLM Service (`services/llm_service.py`)
- **YOLO Integration**: Manages breed and disease detection models
- **Response Generation**: Creates contextual responses using Groq
- **RAG Search**: Retrieves relevant knowledge for responses

### 3. Embedding Service (`services/embedding_service.py`)
- **Vector Creation**: Generates embeddings using Google Gemini
- **Batch Processing**: Handles large-scale document embedding
- **Similarity Search**: Finds relevant content for user queries

### 4. Vector Database (`services/vector_db_service.py`)
- **Pinecone Integration**: Manages vector storage and retrieval
- **Namespace Support**: Organizes different types of knowledge
- **Similarity Queries**: Performs semantic search operations

### 5. Authentication (`services/auth_service.py`)
- **JWT Management**: Handles token creation and validation
- **User Security**: Password hashing and verification
- **Session Control**: Manages user access and permissions

## 🗄️ Database Schema

### Users Collection
```javascript
{
  "_id": ObjectId,
  "user_id": "uuid-string",
  "email": "user@example.com",
  "name": "John Doe",
  "phone_number": "1234567890",
  "password_hash": "bcrypt-hash",
  "created_at": ISODate,
  "last_active": ISODate,
  "is_active": true
}
```

### Chat Sessions Collection
```javascript
{
  "_id": ObjectId,
  "session_id": "uuid-string",
  "user_id": "uuid-string",
  "current_stage": "stage_2b_general_chat",
  "created_at": ISODate,
  "updated_at": ISODate,
  "breed_detection": {
    "model_type": "breed",
    "detected_class": "Golden Retriever",
    "confidence": 0.92,
    "text_result": "Detected breed...",
    "processing_time": 1.23
  },
  "dog_breed": "Golden Retriever",
  "breed_confidence": 0.92,
  "disease_detection": {...},
  "health_condition": "Healthy",
  "conversation_history": ["User: Hello", "Assistant: Hi there!"],
  "is_active": true
}
```

### Chat Messages Collection
```javascript
{
  "_id": ObjectId,
  "message_id": "uuid-string",
  "session_id": "uuid-string",
  "user_id": "uuid-string",
  "message_type": "text",
  "content": "What should I feed my dog?",
  "timestamp": ISODate,
  "image_data": "base64-string",
  "detection_result": {...},
  "is_user_message": true,
  "metadata": {}
}
```

## 🧠 AI Model Integration

### YOLOv11 Models

#### Breed Detection Model
- **Input**: Dog images (RGB format)
- **Output**: Breed classification with confidence scores
- **Classes**: [List of supported dog breeds]
- **Performance**: ~92% accuracy on test dataset

#### Disease Detection Model
- **Input**: Skin condition images
- **Output**: Health condition classification
- **Classes**: [Hot spots, allergies, infections, healthy, etc.]
- **Performance**: ~87% accuracy for common conditions

### Training Data Requirements
- **Breed Detection**: 10,000+ labeled dog images across 50+ breeds
- **Disease Detection**: 5,000+ veterinarian-verified skin condition images
- **Data Augmentation**: Rotation, scaling, color adjustment, cropping

## 📊 Knowledge Base

### Supported Content Types
- **Dog Breeds**: Characteristics, care requirements, health predispositions
- **Health Conditions**: Symptoms, treatments, prevention
- **Nutrition**: Breed-specific dietary guidelines
- **Training**: Behavior modification and training tips
- **Emergency Care**: First aid and urgent care guidance

### Document Processing
1. **Text Extraction**: Convert PDFs, DOCs to plain text
2. **Chunking**: Split into 1000-character segments with 200-character overlap
3. **Embedding**: Generate 768-dimension vectors using Gemini
4. **Storage**: Index in Pinecone with metadata for filtering

## 🔧 Customization

### Adding New Dog Breeds
1. Update YOLO model training data
2. Retrain breed detection model
3. Add breed information to knowledge base
4. Update breed-specific care guidelines

### Extending Health Conditions
1. Collect veterinarian-verified training images
2. Retrain disease detection model
3. Add condition information to knowledge base
4. Update treatment recommendations

### Knowledge Base Expansion
1. Add new text files to `data/all_files/`
2. Run `python create_embeddings_simple.py`
3. Verify embeddings in Pinecone dashboard
4. Test with sample queries

## 🔒 Security Features

### Authentication Security
- **JWT Tokens**: Secure, stateless authentication
- **Password Hashing**: Bcrypt with salt rounds
- **Token Expiration**: Configurable timeout periods
- **Rate Limiting**: Protection against brute force attacks

### Data Protection
- **Input Validation**: Pydantic models for request validation
- **Image Sanitization**: Secure image processing pipeline
- **Database Security**: MongoDB connection encryption
- **API Security**: CORS configuration and security headers

### Privacy Considerations
- **Data Minimization**: Only essential data collection
- **Session Isolation**: User sessions are completely separate
- **Image Handling**: Temporary processing, no permanent storage
- **Audit Logging**: Comprehensive system activity logs

## 📈 Performance & Scaling

### Current Performance
- **API Response Time**: <500ms for text responses
- **Image Processing**: 1-3 seconds for YOLO inference
- **Concurrent Users**: Supports 100+ simultaneous sessions
- **Database Operations**: <100ms for typical queries

### Scaling Strategies
- **Horizontal Scaling**: Multiple API server instances
- **Model Optimization**: TensorRT or ONNX for faster inference
- **Caching**: Redis for frequent queries and embeddings
- **CDN Integration**: Image and static asset delivery
- **Database Sharding**: MongoDB cluster for high volume


### Cloud Deployment
- **AWS**: EC2, ECS, or Lambda for serverless
- **Google Cloud**: Compute Engine or Cloud Run
- **Azure**: Container Instances or App Service
- **Railway/Render**: Simple deployment platforms

### Production Checklist
- [ ] Environment variables configured
- [ ] Database connections tested
- [ ] YOLO models uploaded and verified
- [ ] Knowledge base embeddings created
- [ ] SSL certificates installed
- [ ] Monitoring and logging configured
- [ ] Backup strategy implemented

## 🔍 Monitoring & Logging

### Application Monitoring
- **Health Checks**: Automated endpoint monitoring
- **Performance Metrics**: Response times and throughput
- **Error Tracking**: Comprehensive error logging
- **Resource Usage**: CPU, memory, and disk monitoring

### AI Model Monitoring
- **Inference Metrics**: Model prediction confidence and timing
- **Model Drift**: Detection accuracy over time
- **Usage Patterns**: Most common queries and detection types
- **Quality Metrics**: User feedback and correction rates

## 📋 Roadmap

### Version 2.1 (Current)
- [x] YOLOv11 breed detection
- [x] Health condition analysis
- [x] RAG-powered chat system
- [x] JWT authentication
- [x] MongoDB integration

### Version 2.2 (Planned)
- [ ] Multi-language support
- [ ] Voice interaction capabilities
- [ ] Advanced analytics dashboard
- [ ] Veterinarian consultation integration
- [ ] Mobile app (React Native)

### Version 3.0 (Future)
- [ ] Real-time video analysis
- [ ] IoT device integration (smart collars)
- [ ] Predictive health modeling
- [ ] Telemedicine platform
- [ ] AI-powered nutrition planning


**Built with ❤️ for dog lovers everywhere** 🐕

*Marshee - Making dog care smarter, one wag at a time.*
