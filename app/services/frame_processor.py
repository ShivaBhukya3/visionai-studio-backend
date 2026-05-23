import asyncio
import base64
import time
from typing import AsyncGenerator, Optional
from dataclasses import asdict

import cv2
import numpy as np
from loguru import logger

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from ml.yolo_detector import YOLODetector, DetectionResult
from ml.post_processor import PostProcessor
from app.services.websocket_manager import ws_manager


def _detection_to_dict(det) -> dict:
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": det.confidence,
        "color": det.color,
        "bbox": {
            "x1": det.bbox.x1, "y1": det.bbox.y1,
            "x2": det.bbox.x2, "y2": det.bbox.y2,
            "cx": det.bbox.cx, "cy": det.bbox.cy,
            "w": det.bbox.w, "h": det.bbox.h,
        },
        "area": det.area,
        "track_id": det.track_id,
    }


class FrameProcessor:
    def __init__(self, detector: Optional[YOLODetector] = None):
        self.detector = detector
        self.post = PostProcessor()
        self._fps_tracker: dict[str, list] = {}
        self._prev_detections: dict[str, list] = {}

    def set_detector(self, detector: YOLODetector) -> None:
        self.detector = detector

    def decode_frame(self, frame_data: bytes) -> Optional[np.ndarray]:
        try:
            if isinstance(frame_data, str):
                # base64 string
                frame_data = base64.b64decode(frame_data)
            arr = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return frame
        except Exception as e:
            logger.warning(f"Frame decode failed: {e}")
            return None

    def encode_frame(self, frame: np.ndarray, quality: int = 85) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def resize_for_detection(self, frame: np.ndarray, target: int = 640) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = target / max(h, w)
        if abs(scale - 1.0) < 0.01:
            return frame
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def _track_fps(self, client_id: str) -> float:
        now = time.time()
        if client_id not in self._fps_tracker:
            self._fps_tracker[client_id] = []
        samples = self._fps_tracker[client_id]
        samples.append(now)
        # Keep last 30 timestamps
        cutoff = now - 3.0
        self._fps_tracker[client_id] = [t for t in samples if t > cutoff]
        count = len(self._fps_tracker[client_id])
        return round(count / 3.0, 1) if count > 1 else 0.0

    async def process_webcam_frame(self, frame_data: str,
                                   client_id: str,
                                   settings: dict) -> Optional[dict]:
        if self.detector is None:
            return None

        frame = self.decode_frame(frame_data)
        if frame is None:
            return None

        conf = settings.get("confidence", 0.5)
        classes_filter = settings.get("classes", [])
        box_style = settings.get("box_style", "modern")
        enable_tracking = settings.get("tracking", True)

        original = frame.copy()
        detect_frame = self.resize_for_detection(frame)

        # Run detection
        t0 = time.perf_counter()
        result = self.detector.detect_image(detect_frame)
        inf_ms = round((time.perf_counter() - t0) * 1000, 1)

        # Post-processing
        detections = result.detections
        if classes_filter:
            detections = self.post.filter_by_class(detections, classes_filter)
        detections = self.post.filter_by_confidence(detections, conf)

        # Tracking
        if enable_tracking:
            prev = self._prev_detections.get(client_id, [])
            detections = self.post.track_objects_simple(prev, detections)
            self._prev_detections[client_id] = detections

        result.detections = detections
        result.total_objects = len(detections)

        # Annotate original resolution frame
        # Scale bboxes back to original size
        scale_x = original.shape[1] / detect_frame.shape[1]
        scale_y = original.shape[0] / detect_frame.shape[0]
        for d in detections:
            d.bbox.x1 *= scale_x; d.bbox.x2 *= scale_x
            d.bbox.y1 *= scale_y; d.bbox.y2 *= scale_y
            d.bbox.cx *= scale_x; d.bbox.cy *= scale_y
            d.bbox.w *= scale_x; d.bbox.h *= scale_y

        annotated = self.detector.get_annotated_image(original, result, box_style)
        encoded = self.encode_frame(annotated)
        fps = self._track_fps(client_id)

        return {
            "frame": encoded,
            "detections": [_detection_to_dict(d) for d in detections],
            "fps": fps,
            "inference_ms": inf_ms,
            "total_objects": result.total_objects,
            "class_counts": result.class_counts,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    async def process_video_file(self, video_path: str,
                                 client_id: str,
                                 settings: dict = None) -> AsyncGenerator:
        if settings is None:
            settings = {}
        if self.detector is None:
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        frame_num = 0
        prev_dets = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                result = self.detector.detect_image(frame)
                detections = self.post.track_objects_simple(
                    prev_dets, result.detections)
                prev_dets = detections
                result.detections = detections

                annotated = self.detector.get_annotated_image(
                    frame, result,
                    settings.get("box_style", "modern"))
                encoded = self.encode_frame(annotated)

                yield {
                    "frame": encoded,
                    "frame_number": frame_num,
                    "detections": [_detection_to_dict(d) for d in detections],
                    "inference_ms": result.inference_time_ms,
                    "total_objects": result.total_objects,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
                await asyncio.sleep(0)
        finally:
            cap.release()


frame_processor = FrameProcessor()
