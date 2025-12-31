/**
 * WebSocket Hook
 * 
 * Real-time pose detection via WebSocket connection to backend.
 */

import { useState, useEffect, useCallback, useRef } from 'react';
import type { PoseFrame, WebSocketMessage, PoseResultData } from '@/types';

interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface UseWebSocketReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  lastPose: PoseFrame | null;
  processingTime: number;
  frameCount: number;
  connect: () => void;
  disconnect: () => void;
  sendFrame: (imageBase64: string, frameNumber: number) => void;
}

export function useWebSocket(
  url: string = 'ws://localhost:8000/ws/pose',
  options: UseWebSocketOptions = {}
): UseWebSocketReturn {
  const {
    autoConnect = false,
    reconnectAttempts = 3,
    reconnectInterval = 2000,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastPose, setLastPose] = useState<PoseFrame | null>(null);
  const [processingTime, setProcessingTime] = useState(0);
  const [frameCount, setFrameCount] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectCountRef = useRef(0);

  // Handle incoming messages
  const handleMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'session_started':
        console.log('📍 Session started');
        break;

      case 'pose_result': {
        const data = message.data as unknown as PoseResultData;
        if (data.pose) {
          setLastPose(data.pose);
        }
        setProcessingTime(data.processing_time_ms);
        setFrameCount((prev) => prev + 1);
        break;
      }

      case 'error':
        console.error('Server error:', message.data);
        setError(message.data.error as string);
        break;

      case 'session_ended':
        console.log('📍 Session ended');
        break;

      default:
        console.log('Unknown message type:', message.type);
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setIsConnecting(false);
        reconnectCountRef.current = 0;
      };

      ws.onclose = (event) => {
        console.log('❌ WebSocket closed:', event.code, event.reason);
        setIsConnected(false);
        setIsConnecting(false);

        // Auto-reconnect if not intentional close
        if (event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          reconnectCountRef.current++;
          console.log(`🔄 Reconnecting... (${reconnectCountRef.current}/${reconnectAttempts})`);
          setTimeout(connect, reconnectInterval);
        }
      };

      ws.onerror = () => {
        console.error('⚠️ WebSocket error');
        setError('WebSocket connection error');
        setIsConnecting(false);
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleMessage(message);
        } catch (e) {
          console.error('Failed to parse WebSocket message:', e);
        }
      };

      wsRef.current = ws;
    } catch (e) {
      setError(`Failed to connect: ${e}`);
      setIsConnecting(false);
    }
  }, [url, reconnectAttempts, reconnectInterval, handleMessage]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (wsRef.current) {
      // Send end session message
      if (wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          type: 'end_session',
          data: {},
          timestamp: Date.now(),
        }));
      }
      wsRef.current.close(1000, 'Client disconnect');
      wsRef.current = null;
    }
    setIsConnected(false);
    setLastPose(null);
    setFrameCount(0);
  }, []);

  // Send frame for analysis
  const sendFrame = useCallback((imageBase64: string, frameNumber: number) => {
    if (wsRef.current?.readyState !== WebSocket.OPEN) {
      console.warn('WebSocket not connected');
      return;
    }

    const message: WebSocketMessage = {
      type: 'frame',
      data: {
        image_base64: imageBase64,
        frame_number: frameNumber,
      },
      timestamp: Date.now(),
    };

    wsRef.current.send(JSON.stringify(message));
  }, []);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    error,
    lastPose,
    processingTime,
    frameCount,
    connect,
    disconnect,
    sendFrame,
  };
}