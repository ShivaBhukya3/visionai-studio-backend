import base64
import json
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def _make_frame_b64():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def test_websocket_connect():
    with client.websocket_connect("/ws/stream/test-client") as ws:
        data = ws.receive_json()
        assert data["type"] == "connected"
        assert data["client_id"] == "test-client"


def test_websocket_ping():
    with client.websocket_connect("/ws/stream/test-ping") as ws:
        ws.receive_json()  # connected message
        ws.send_json({"type": "ping"})
        pong = ws.receive_json()
        assert pong["type"] == "pong"


def test_websocket_frame():
    with client.websocket_connect("/ws/stream/test-frame") as ws:
        ws.receive_json()  # connected
        frame = _make_frame_b64()
        ws.send_json({
            "type": "frame",
            "data": frame,
            "settings": {"confidence": 0.5},
        })
        result = ws.receive_json()
        assert result["type"] == "detection_result"
        assert "detections" in result
