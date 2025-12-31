"""
WebSocket Handler

Real-time pose detection via WebSocket connection.
Allows frontend to stream video frames and receive pose data instantly.
"""

import json
import time
import logging
import asyncio
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect

from .schemas import (
    WebSocketMessageType,
    LandmarkSchema,
    PoseFrameSchema,
)
from core.services import PoseDetector

# Configure logging
logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections.
    
    Handles multiple concurrent connections and broadcasts.
    """
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.pose_detectors: dict[WebSocket, PoseDetector] = {}
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Create dedicated pose detector for this connection
        self.pose_detectors[websocket] = PoseDetector(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Handle WebSocket disconnection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up pose detector
        if websocket in self.pose_detectors:
            self.pose_detectors[websocket].close()
            del self.pose_detectors[websocket]
        
        logger.info(f"WebSocket disconnected. Remaining: {len(self.active_connections)}")
    
    def get_detector(self, websocket: WebSocket) -> Optional[PoseDetector]:
        """Get pose detector for a connection."""
        return self.pose_detectors.get(websocket)
    
    async def send_json(self, websocket: WebSocket, data: dict) -> None:
        """Send JSON data to a specific connection."""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for real-time pose detection.
    
    Protocol:
    1. Client connects
    2. Client sends frames as base64 images
    3. Server responds with pose landmarks
    4. Client disconnects when done
    
    Message format (client -> server):
    {
        "type": "frame",
        "data": {
            "image_base64": "...",
            "frame_number": 0
        },
        "timestamp": 1704067200000
    }
    
    Message format (server -> client):
    {
        "type": "pose_result",
        "data": {
            "frame_number": 0,
            "pose": { ... },
            "processing_time_ms": 25.5
        },
        "timestamp": 1704067200025
    }
    """
    await manager.connect(websocket)
    
    try:
        # Send session started message
        await manager.send_json(websocket, {
            "type": WebSocketMessageType.SESSION_STARTED.value,
            "data": {"message": "Connected to HappyCoach pose detection"},
            "timestamp": int(time.time() * 1000)
        })
        
        # Main message loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_json()
                
                # Process based on message type
                msg_type = data.get("type")
                
                if msg_type == WebSocketMessageType.FRAME.value:
                    await handle_frame(websocket, data)
                    
                elif msg_type == WebSocketMessageType.END_SESSION.value:
                    await manager.send_json(websocket, {
                        "type": WebSocketMessageType.SESSION_ENDED.value,
                        "data": {"message": "Session ended"},
                        "timestamp": int(time.time() * 1000)
                    })
                    break
                    
                else:
                    await manager.send_json(websocket, {
                        "type": WebSocketMessageType.ERROR.value,
                        "data": {"error": f"Unknown message type: {msg_type}"},
                        "timestamp": int(time.time() * 1000)
                    })
                    
            except json.JSONDecodeError:
                await manager.send_json(websocket, {
                    "type": WebSocketMessageType.ERROR.value,
                    "data": {"error": "Invalid JSON"},
                    "timestamp": int(time.time() * 1000)
                })
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


async def handle_frame(websocket: WebSocket, message: dict) -> None:
    """
    Process a video frame and return pose detection result.
    """
    start_time = time.time()
    
    try:
        frame_data = message.get("data", {})
        image_base64 = frame_data.get("image_base64", "")
        frame_number = frame_data.get("frame_number", 0)
        
        if not image_base64:
            await manager.send_json(websocket, {
                "type": WebSocketMessageType.ERROR.value,
                "data": {"error": "No image data provided"},
                "timestamp": int(time.time() * 1000)
            })
            return
        
        # Get detector for this connection
        detector = manager.get_detector(websocket)
        if not detector:
            await manager.send_json(websocket, {
                "type": WebSocketMessageType.ERROR.value,
                "data": {"error": "Detector not initialized"},
                "timestamp": int(time.time() * 1000)
            })
            return
        
        # Detect pose
        pose_frame = detector.detect_from_base64(
            image_base64,
            timestamp_ms=message.get("timestamp", 0),
            frame_number=frame_number
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        if pose_frame:
            landmarks = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility,
                    "body_part": lm.body_part.name if lm.body_part else None
                }
                for lm in pose_frame.landmarks
            ]
            
            pose_data = {
                "landmarks": landmarks,
                "timestamp_ms": pose_frame.timestamp_ms,
                "frame_number": pose_frame.frame_number,
                "confidence": pose_frame.confidence
            }
        else:
            pose_data = None
        
        await manager.send_json(websocket, {
            "type": WebSocketMessageType.POSE_RESULT.value,
            "data": {
                "frame_number": frame_number,
                "pose": pose_data,
                "processing_time_ms": processing_time
            },
            "timestamp": int(time.time() * 1000)
        })
        
    except Exception as e:
        logger.error(f"Frame processing error: {e}")
        await manager.send_json(websocket, {
            "type": WebSocketMessageType.ERROR.value,
            "data": {"error": str(e)},
            "timestamp": int(time.time() * 1000)
        })