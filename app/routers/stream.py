import asyncio
import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from app.services.websocket_manager import ws_manager
from app.services.frame_processor import frame_processor, FrameProcessor
from app.services.yolo_service import yolo_service
from app.config import settings

router = APIRouter(tags=["stream"])

_FRAME_INTERVAL = 1.0 / settings.WS_FRAME_RATE


@router.websocket("/ws/stream/{client_id}")
async def websocket_stream(websocket: WebSocket, client_id: str):
    if len(ws_manager.active_connections) >= settings.WS_MAX_CONNECTIONS:
        await websocket.close(code=1008, reason="Max connections reached")
        return

    await ws_manager.connect(websocket, client_id)

    # Initialize frame processor for this client
    processor = FrameProcessor()
    detector = yolo_service.get_detector()
    processor.set_detector(detector)

    last_frame_time = 0.0

    try:
        while True:
            data = await websocket.receive_json()
            msg_type = data.get("type", "frame")

            if msg_type == "ping":
                await ws_manager.send_json(client_id, {"type": "pong"})
                continue

            if msg_type == "settings":
                # Hot-swap model/confidence
                new_model = data.get("model")
                new_conf = data.get("confidence")
                if new_model or new_conf:
                    processor.set_detector(
                        yolo_service.get_detector(new_model, new_conf))
                await ws_manager.send_json(client_id, {
                    "type": "settings_applied",
                    "model": new_model or "unchanged",
                })
                continue

            if msg_type == "frame":
                # Rate limiting
                now = time.monotonic()
                if now - last_frame_time < _FRAME_INTERVAL:
                    continue
                last_frame_time = now

                frame_data = data.get("data", "")
                stream_settings = data.get("settings", {})

                if not frame_data:
                    continue

                result = await processor.process_webcam_frame(
                    frame_data, client_id, stream_settings)

                if result:
                    await ws_manager.send_json(client_id, {
                        "type": "detection_result",
                        **result,
                    })

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        ws_manager.disconnect(client_id)
