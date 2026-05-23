import asyncio
import base64
import json
import time
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self._stats: dict[str, dict] = {}
        self._global_frames_processed = 0
        self._start_time = time.time()

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self._stats[client_id] = {
            "connected_at": time.time(),
            "frames_sent": 0,
            "frames_received": 0,
            "latency_samples": [],
        }
        logger.info(f"WS client connected: {client_id} "
                    f"(total: {len(self.active_connections)})")
        await self.send_json(client_id, {
            "type": "connected",
            "client_id": client_id,
            "message": "VisionAI Studio WebSocket connected",
        })

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)
        self._stats.pop(client_id, None)
        logger.info(f"WS client disconnected: {client_id}")

    async def send_json(self, client_id: str, data: dict) -> bool:
        ws = self.active_connections.get(client_id)
        if ws is None:
            return False
        try:
            await ws.send_json(data)
            if client_id in self._stats:
                self._stats[client_id]["frames_sent"] += 1
            return True
        except Exception as e:
            logger.warning(f"Failed to send to {client_id}: {e}")
            self.disconnect(client_id)
            return False

    async def send_detection_result(self, client_id: str,
                                    result: dict) -> bool:
        return await self.send_json(client_id, {
            "type": "detection_result",
            **result,
        })

    async def send_frame(self, client_id: str,
                         frame_bytes: bytes,
                         result: dict) -> bool:
        frame_b64 = base64.b64encode(frame_bytes).decode("utf-8")
        payload = {
            "type": "detection_result",
            "frame": frame_b64,
            **result,
        }
        self._global_frames_processed += 1
        return await self.send_json(client_id, payload)

    async def broadcast(self, message: dict) -> None:
        disconnected = []
        for cid in list(self.active_connections.keys()):
            if not await self.send_json(cid, message):
                disconnected.append(cid)
        for cid in disconnected:
            self.disconnect(cid)

    def record_latency(self, client_id: str, latency_ms: float) -> None:
        if client_id in self._stats:
            samples = self._stats[client_id]["latency_samples"]
            samples.append(latency_ms)
            if len(samples) > 60:
                samples.pop(0)

    def get_connection_stats(self) -> dict:
        total_latency = []
        total_frames = 0
        for s in self._stats.values():
            total_latency.extend(s["latency_samples"])
            total_frames += s["frames_sent"]

        avg_latency = (sum(total_latency) / len(total_latency)
                       if total_latency else 0)
        return {
            "total_connections": len(self.active_connections),
            "active_streams": len(self.active_connections),
            "frames_processed": self._global_frames_processed,
            "avg_latency_ms": round(avg_latency, 2),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }

    def is_connected(self, client_id: str) -> bool:
        return client_id in self.active_connections


ws_manager = ConnectionManager()
