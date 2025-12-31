"""
HappyCoach2 API Module

FastAPI routes and WebSocket handlers for golf swing analysis.
"""

from .routes import router
from .websocket import websocket_endpoint, manager

__all__ = [
    "router",
    "websocket_endpoint",
    "manager",
]