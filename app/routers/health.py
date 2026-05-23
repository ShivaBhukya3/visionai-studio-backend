import time
from fastapi import APIRouter
from app.config import settings
from app.services.websocket_manager import ws_manager
from ml.model_manager import model_manager

router = APIRouter(prefix="/health", tags=["health"])

_start_time = time.time()


@router.get("")
async def health_check():
    active_model = model_manager.get_active_model()
    ws_stats = ws_manager.get_connection_stats()
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "uptime_seconds": round(time.time() - _start_time, 1),
        "model": {
            "active": active_model,
            "status": "loaded",
        },
        "websocket": ws_stats,
        "services": {
            "backend": "online",
            "model": "loaded",
            "websocket": "ready",
        },
    }


@router.get("/ping")
async def ping():
    return {"pong": True, "timestamp": time.time()}
