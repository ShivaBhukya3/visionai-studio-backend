import time
import base64
import io
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from loguru import logger


# 80 COCO class colors — consistent palette
COCO_COLORS = {
    "person": "#FF6B6B", "bicycle": "#4ECDC4", "car": "#45B7D1",
    "motorcycle": "#96CEB4", "airplane": "#FFEAA7", "bus": "#DDA0DD",
    "train": "#98D8C8", "truck": "#F7DC6F", "boat": "#BB8FCE",
    "traffic light": "#85C1E9", "fire hydrant": "#F8C471", "stop sign": "#EC7063",
    "parking meter": "#A9CCE3", "bench": "#A3E4D7", "bird": "#FAD7A0",
    "cat": "#D7BDE2", "dog": "#A9DFBF", "horse": "#F9E79F",
    "sheep": "#ABEBC6", "cow": "#F0B27A", "elephant": "#AED6F1",
    "bear": "#A8D5A2", "zebra": "#F8BBD9", "giraffe": "#FFD54F",
    "backpack": "#80CBC4", "umbrella": "#CE93D8", "handbag": "#80DEEA",
    "tie": "#FFCC02", "suitcase": "#FF8A65", "frisbee": "#A5D6A7",
    "skis": "#90CAF9", "snowboard": "#F48FB1", "sports ball": "#FFAB40",
    "kite": "#BCAAA4", "baseball bat": "#EF9A9A", "baseball glove": "#80CBC4",
    "skateboard": "#B39DDB", "surfboard": "#FFE082", "tennis racket": "#80DEEA",
    "bottle": "#C5E1A5", "wine glass": "#FFCCBC", "cup": "#B3E5FC",
    "fork": "#DCEDC8", "knife": "#FFF9C4", "spoon": "#FCE4EC",
    "bowl": "#E8F5E9", "banana": "#FFFDE7", "apple": "#FBE9E7",
    "sandwich": "#E3F2FD", "orange": "#FFF8E1", "broccoli": "#F3E5F5",
    "carrot": "#E0F2F1", "hot dog": "#FBE9E7", "pizza": "#F9FBE7",
    "donut": "#E8EAF6", "cake": "#FCE4EC", "chair": "#E0F7FA",
    "couch": "#F1F8E9", "potted plant": "#FFF3E0", "bed": "#EDE7F6",
    "dining table": "#E8F5E9", "toilet": "#E3F2FD", "tv": "#FFF8E1",
    "laptop": "#F3E5F5", "mouse": "#E8EAF6", "remote": "#FCE4EC",
    "keyboard": "#E0F2F1", "cell phone": "#FBE9E7", "microwave": "#E1F5FE",
    "oven": "#F9FBE7", "toaster": "#FFFDE7", "sink": "#E8F5E9",
    "refrigerator": "#E3F2FD", "book": "#FFF8E1", "clock": "#F3E5F5",
    "vase": "#E8EAF6", "scissors": "#FCE4EC", "teddy bear": "#FFF9C4",
    "hair drier": "#DCEDC8", "toothbrush": "#B3E5FC",
}

DEFAULT_COLOR = "#6366F1"


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: BBox
    area: float
    aspect_ratio: float
    color: str = DEFAULT_COLOR
    track_id: Optional[int] = None


@dataclass
class DetectionResult:
    detections: list = field(default_factory=list)
    image_shape: dict = field(default_factory=dict)
    inference_time_ms: float = 0.0
    device_used: str = "cpu"
    model_name: str = ""
    total_objects: int = 0
    class_counts: dict = field(default_factory=dict)


class YOLODetector:
    SUPPORTED_MODELS = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
        "yolov8l.pt", "yolov8x.pt",
    ]

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.5, iou: float = 0.45):
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.model = None
        self.device = "cpu"
        self.model_name = Path(model_path).name
        self._load_model()

    def _load_model(self):
        try:
            import torch
            from ultralytics import YOLO

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading {self.model_path} on {self.device}")
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded: {self.model_name} | Device: {self.device}")
            self._warmup()
        except ImportError:
            logger.warning("ultralytics not installed — running in mock mode")
            self.model = None

    def _warmup(self):
        if self.model is None:
            return
        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(blank, verbose=False)
        logger.info("Model warmup complete")

    def detect_image(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            return self._mock_result(image)

        t0 = time.perf_counter()
        results = self.model(
            image,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
            device=self.device,
        )
        inference_ms = (time.perf_counter() - t0) * 1000

        h, w = image.shape[:2]
        detections = self._parse_results(results[0], w, h)
        class_counts: dict = {}
        for d in detections:
            class_counts[d.class_name] = class_counts.get(d.class_name, 0) + 1

        return DetectionResult(
            detections=detections,
            image_shape={"width": w, "height": h},
            inference_time_ms=round(inference_ms, 2),
            device_used=self.device,
            model_name=self.model_name,
            total_objects=len(detections),
            class_counts=class_counts,
        )

    def _parse_results(self, result, img_w: int, img_h: int) -> list:
        detections = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        names = result.names
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            aspect = bw / bh if bh > 0 else 1.0

            cls_name = names.get(cls_id, f"class_{cls_id}")
            color = COCO_COLORS.get(cls_name.lower(), DEFAULT_COLOR)

            det = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf, 4),
                bbox=BBox(
                    x1=round(x1, 1), y1=round(y1, 1),
                    x2=round(x2, 1), y2=round(y2, 1),
                    cx=round(cx, 1), cy=round(cy, 1),
                    w=round(bw, 1), h=round(bh, 1),
                ),
                area=round(area, 1),
                aspect_ratio=round(aspect, 3),
                color=color,
            )
            detections.append(det)
        return detections

    def detect_batch(self, images: list) -> list:
        return [self.detect_image(img) for img in images]

    def detect_video_frame(self, frame_bytes: bytes) -> DetectionResult:
        arr = np.frombuffer(frame_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return DetectionResult()
        return self.detect_image(frame)

    def get_annotated_image(self, image: np.ndarray,
                            result: DetectionResult,
                            style: str = "modern") -> np.ndarray:
        annotated = image.copy()

        for det in result.detections:
            color_hex = det.color
            color_bgr = self._hex_to_bgr(color_hex)
            b = det.bbox

            if style == "modern":
                annotated = self._draw_modern_box(
                    annotated, det, b, color_bgr)
            elif style == "minimal":
                annotated = self._draw_minimal_box(
                    annotated, det, b, color_bgr)
            elif style == "heatmap":
                annotated = self._draw_heatmap_box(
                    annotated, det, b)
            else:
                annotated = self._draw_modern_box(
                    annotated, det, b, color_bgr)

        return annotated

    def _draw_modern_box(self, img, det, b, color):
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)

        # Semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

        # Main border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        cl = 14
        cv2.line(img, (x1, y1), (x1 + cl, y1), color, 3)
        cv2.line(img, (x1, y1), (x1, y1 + cl), color, 3)
        cv2.line(img, (x2, y1), (x2 - cl, y1), color, 3)
        cv2.line(img, (x2, y1), (x2, y1 + cl), color, 3)
        cv2.line(img, (x1, y2), (x1 + cl, y2), color, 3)
        cv2.line(img, (x1, y2), (x1, y2 - cl), color, 3)
        cv2.line(img, (x2, y2), (x2 - cl, y2), color, 3)
        cv2.line(img, (x2, y2), (x2, y2 - cl), color, 3)

        # Label background
        label = f"{det.class_name}  {det.confidence:.0%}"
        font = cv2.FONT_HERSHEY_DUPLEX
        fs = 0.45
        (tw, th), _ = cv2.getTextSize(label, font, fs, 1)
        lx1, ly1 = x1, max(y1 - th - 10, 0)
        lx2, ly2 = x1 + tw + 10, y1
        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
        cv2.putText(img, label, (lx1 + 5, ly2 - 4),
                    font, fs, (10, 10, 10), 1, cv2.LINE_AA)
        return img

    def _draw_minimal_box(self, img, det, b, color):
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(img, label, (x1 + 3, y1 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
        return img

    def _draw_heatmap_box(self, img, det, b):
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        # Map confidence to color: low=blue, mid=green, high=red
        conf = det.confidence
        r = int(conf * 255)
        g = int((1 - abs(conf - 0.5) * 2) * 255)
        bl = int((1 - conf) * 255)
        color = (bl, g, r)
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, conf * 0.4, img, 1 - conf * 0.4, 0, img)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        return img

    @staticmethod
    def _hex_to_bgr(hex_color: str):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)

    def get_class_colors(self) -> dict:
        return dict(COCO_COLORS)

    def benchmark(self, n_iterations: int = 100) -> dict:
        if self.model is None:
            return {"error": "Model not loaded"}

        blank = np.zeros((640, 640, 3), dtype=np.uint8)
        times = []
        for _ in range(n_iterations):
            t0 = time.perf_counter()
            self.model(blank, verbose=False)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        return {
            "avg_ms": round(avg, 2),
            "min_ms": round(min(times), 2),
            "max_ms": round(max(times), 2),
            "fps": round(1000 / avg, 1),
            "device": self.device,
            "model": self.model_name,
            "iterations": n_iterations,
        }

    def get_model_info(self) -> dict:
        info: dict = {
            "model_name": self.model_name,
            "device": self.device,
            "confidence_threshold": self.confidence,
            "iou_threshold": self.iou,
        }
        if self.model is not None:
            try:
                info["classes"] = list(self.model.names.values())
                info["num_classes"] = len(self.model.names)
            except Exception:
                pass
        return info

    def _mock_result(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        return DetectionResult(
            detections=[],
            image_shape={"width": w, "height": h},
            inference_time_ms=0.0,
            device_used="mock",
            model_name=self.model_name,
            total_objects=0,
            class_counts={},
        )
