from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    APP_NAME: str = "VisionAI Studio"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"

    DATABASE_URL: str = "sqlite+aiosqlite:///./visionai.db"
    REDIS_URL: str = "redis://localhost:6379"

    YOLO_MODEL: str = "yolov8n.pt"
    YOLO_CONFIDENCE: float = 0.5
    YOLO_IOU_THRESHOLD: float = 0.45
    MAX_DETECTIONS: int = 100

    MAX_IMAGE_SIZE_MB: int = 50
    MAX_VIDEO_SIZE_MB: int = 500
    ALLOWED_IMAGE_TYPES: List[str] = [
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    ]
    ALLOWED_VIDEO_TYPES: List[str] = [
        "video/mp4", "video/avi", "video/mov", "video/mkv"
    ]

    WS_MAX_CONNECTIONS: int = 100
    WS_FRAME_RATE: int = 30

    AWS_BUCKET: str = ""
    AWS_REGION: str = "us-east-1"

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
        "https://frontend-gold-xi-99.vercel.app",
    ]

    PRODUCTION_CORS_ORIGIN: str = ""  # set to your Vercel URL in HF env vars

    MODEL_CACHE_DIR: str = os.path.join(os.path.expanduser("~"), ".cache", "ultralytics")

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
