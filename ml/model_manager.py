import os
from pathlib import Path
from typing import Optional
from loguru import logger

from ml.yolo_detector import YOLODetector


MODEL_CATALOG = [
    {
        "name": "yolov8n.pt",
        "display": "YOLOv8 Nano",
        "size_mb": 6.2,
        "speed_ms": 22.1,
        "fps": 45.2,
        "map50": 37.3,
        "description": "Fastest model, ideal for real-time edge deployment",
        "badge": "fast",
        "params": "3.2M",
    },
    {
        "name": "yolov8s.pt",
        "display": "YOLOv8 Small",
        "size_mb": 21.5,
        "speed_ms": 33.4,
        "fps": 29.9,
        "map50": 44.9,
        "description": "Balanced speed/accuracy for most applications",
        "badge": "balanced",
        "params": "11.2M",
    },
    {
        "name": "yolov8m.pt",
        "display": "YOLOv8 Medium",
        "size_mb": 52.0,
        "speed_ms": 55.7,
        "fps": 17.9,
        "map50": 50.2,
        "description": "Higher accuracy for production workloads",
        "badge": "balanced",
        "params": "25.9M",
    },
    {
        "name": "yolov8l.pt",
        "display": "YOLOv8 Large",
        "size_mb": 87.7,
        "speed_ms": 83.2,
        "fps": 12.0,
        "map50": 52.9,
        "description": "High accuracy with moderate GPU requirements",
        "badge": "accurate",
        "params": "43.7M",
    },
    {
        "name": "yolov8x.pt",
        "display": "YOLOv8 XLarge",
        "size_mb": 136.7,
        "speed_ms": 125.6,
        "fps": 7.96,
        "map50": 53.9,
        "description": "Maximum accuracy, GPU strongly recommended",
        "badge": "accurate",
        "params": "68.2M",
    },
]


class ModelManager:
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or os.path.join(
            os.path.expanduser("~"), ".cache", "ultralytics"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._loaded_models: dict[str, YOLODetector] = {}
        self._active_model: str = "yolov8n.pt"

    def list_available_models(self) -> list:
        models = []
        for m in MODEL_CATALOG:
            status = "downloaded" if self._is_downloaded(m["name"]) else "not_downloaded"
            models.append({**m, "status": status})
        return models

    def _is_downloaded(self, model_name: str) -> bool:
        return (self.cache_dir / model_name).exists()

    def download_model(self, model_name: str) -> bool:
        try:
            from ultralytics import YOLO
            logger.info(f"Downloading {model_name}...")
            YOLO(model_name)
            logger.info(f"Downloaded {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    def load_model(self, model_name: str,
                   confidence: float = 0.5,
                   iou: float = 0.45) -> YOLODetector:
        cache_key = f"{model_name}_{confidence}_{iou}"
        if cache_key not in self._loaded_models:
            logger.info(f"Loading model {model_name}")
            self._loaded_models[cache_key] = YOLODetector(
                model_path=model_name,
                confidence=confidence,
                iou=iou,
            )
        return self._loaded_models[cache_key]

    def switch_model(self, model_name: str) -> None:
        self._active_model = model_name
        logger.info(f"Active model switched to {model_name}")

    def get_active_model(self) -> str:
        return self._active_model

    def get_model_metrics(self) -> dict:
        metrics = {}
        for key, detector in self._loaded_models.items():
            model_name = key.split("_")[0]
            metrics[model_name] = detector.get_model_info()
        return metrics

    def get_model_info(self, model_name: str) -> dict:
        for m in MODEL_CATALOG:
            if m["name"] == model_name:
                return m
        return {}


model_manager = ModelManager()
