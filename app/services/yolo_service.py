from typing import Optional
from loguru import logger

from ml.yolo_detector import YOLODetector
from ml.model_manager import model_manager
from app.config import settings


class YOLOService:
    def __init__(self):
        self._detector: Optional[YOLODetector] = None
        self._current_model: str = settings.YOLO_MODEL

    def get_detector(self, model_name: Optional[str] = None,
                     confidence: Optional[float] = None,
                     iou: Optional[float] = None) -> YOLODetector:
        model = model_name or self._current_model
        conf = confidence if confidence is not None else settings.YOLO_CONFIDENCE
        iou_val = iou if iou is not None else settings.YOLO_IOU_THRESHOLD

        return model_manager.load_model(model, conf, iou_val)

    def switch_model(self, model_name: str) -> None:
        self._current_model = model_name
        model_manager.switch_model(model_name)
        logger.info(f"Switched to model: {model_name}")

    def get_current_model(self) -> str:
        return self._current_model


yolo_service = YOLOService()
