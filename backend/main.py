"""
HappyCoach2 Backend API

FastAPI application for golf swing analysis with real-time pose detection.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    
API docs available at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router as api_router
from api.websocket import websocket_endpoint

# =============================================================================
# Logging Configuration
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Lifespan (startup/shutdown)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    
    Runs startup code before app starts accepting requests,
    and cleanup code when app shuts down.
    """
    # Startup
    logger.info(" HappyCoach2 API starting up...")
    logger.info(" API docs: http://localhost:8000/docs")
    logger.info(" WebSocket: ws://localhost:8000/ws/pose")
    
    # Test MediaPipe availability
    try:
        from core.services import PoseDetector
        with PoseDetector() as detector:
            logger.info(" MediaPipe initialized successfully")
    except Exception as e:
        logger.warning(f" MediaPipe initialization warning: {e}")
    
    yield  # App runs here
    
    # Shutdown
    logger.info(" HappyCoach2 API shutting down...")


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="HappyCoach2 API",
    description="""
    **AI-Powered Golf Swing Analyzer**
    
    Real-time pose detection and biomechanics analysis for golf swings.
    
    ## Features
    
    - **Real-time Pose Detection** via WebSocket
    - **Video Analysis** with phase detection
    - **Biomechanical Scoring** (posture, tempo, rotation, balance)
    - **Coaching Tips** based on analysis
    
    ## Endpoints
    
    - `GET /api/health` - Health check
    - `POST /api/pose/detect` - Single image pose detection
    - `POST /api/analysis/video` - Full video analysis
    - `WS /ws/pose` - Real-time pose detection stream
    
    ## WebSocket Protocol
    
    Connect to `/ws/pose` and send frames as JSON:
```json
    {
        "type": "frame",
        "data": {"image_base64": "...", "frame_number": 0},
        "timestamp": 1704067200000
    }
```
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React dev server
        "http://localhost:5173",      # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",                          # Allow all for development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Routes
# =============================================================================

# Include REST API routes
app.include_router(api_router, prefix="/api")

# WebSocket endpoint
app.websocket("/ws/pose")(websocket_endpoint)


# =============================================================================
# Root endpoint
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint - API information.
    """
    return {
        "name": "HappyCoach2 API",
        "version": "1.0.0",
        "description": "AI-Powered Golf Swing Analyzer",
        "docs": "/docs",
        "health": "/api/health",
        "websocket": "ws://localhost:8000/ws/pose"
    }


# =============================================================================
# Run directly (for development)
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )