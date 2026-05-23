import io
import uuid
import base64
import asyncio
import httpx
from datetime import datetime
from typing import Optional, List

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, Form
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image

from app.config import settings
from app.services.yolo_service import yolo_service
from app.services.analytics_service import analytics_service
from ml.post_processor import PostProcessor
from app.schemas.detection_schemas import URLDetectionRequest

router = APIRouter(prefix="/api/v1/detect", tags=["detection"])
post_processor = PostProcessor()

# In-memory detection store (production: use database)
_detection_store: dict = {}


def _build_response(detection_id: str, result, annotated_b64: str,
                    image_shape: dict, model_name: str) -> dict:
    dets = []
    total_conf = 0.0
    for d in result.detections:
        dets.append({
            "class_name": d.class_name,
            "class_id": d.class_id,
            "confidence": d.confidence,
            "color": d.color,
            "bbox": {
                "x1": d.bbox.x1, "y1": d.bbox.y1,
                "x2": d.bbox.x2, "y2": d.bbox.y2,
                "cx": d.bbox.cx, "cy": d.bbox.cy,
                "w": d.bbox.w, "h": d.bbox.h,
            },
        })
        total_conf += d.confidence

    avg_conf = total_conf / len(dets) if dets else 0.0
    response = {
        "detection_id": detection_id,
        "detections": dets,
        "summary": {
            "total_objects": result.total_objects,
            "class_counts": result.class_counts,
            "avg_confidence": round(avg_conf, 4),
            "processing_time_ms": result.inference_time_ms,
        },
        "annotated_image_base64": annotated_b64,
        "model_used": model_name,
        "image_dimensions": image_shape,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return response


@router.post("/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Query(default=0.5, ge=0.1, le=1.0),
    iou_threshold: float = Query(default=0.45, ge=0.1, le=1.0),
    classes: Optional[str] = Query(default=None),
    model: str = Query(default="yolov8n.pt"),
    return_annotated: bool = Query(default=True),
    box_style: str = Query(default="modern"),
):
    # Validate content type
    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported type: {file.content_type}")

    contents = await file.read()
    if len(contents) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, "Image too large")

    # Decode image
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Cannot decode image")

    detector = yolo_service.get_detector(model, confidence, iou_threshold)
    result = detector.detect_image(image)

    # Filter by classes
    classes_list = [c.strip() for c in classes.split(",")] if classes else []
    if classes_list:
        result.detections = post_processor.filter_by_class(
            result.detections, classes_list)
        result.total_objects = len(result.detections)

    annotated_b64 = None
    if return_annotated:
        annotated = detector.get_annotated_image(image, result, box_style)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    detection_id = str(uuid.uuid4())
    response = _build_response(
        detection_id, result, annotated_b64,
        result.image_shape, result.model_name)

    _detection_store[detection_id] = response
    return JSONResponse(content=response)


@router.post("/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Query(default=0.5, ge=0.1, le=1.0),
    model: str = Query(default="yolov8n.pt"),
    return_annotated: bool = Query(default=True),
):
    if len(files) > 10:
        raise HTTPException(400, "Max 10 images per batch")

    detector = yolo_service.get_detector(model, confidence)
    results = []

    for f in files:
        contents = await f.read()
        arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            continue

        result = detector.detect_image(image)
        annotated_b64 = None
        if return_annotated:
            annotated = detector.get_annotated_image(image, result, "modern")
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        did = str(uuid.uuid4())
        r = _build_response(did, result, annotated_b64,
                            result.image_shape, result.model_name)
        results.append(r)

    total_objects = sum(r["summary"]["total_objects"] for r in results)
    return {"results": results, "total_images": len(results), "total_objects": total_objects}


@router.post("/url")
async def detect_from_url(body: URLDetectionRequest):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(body.image_url)
            resp.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch image: {e}")

    contents = resp.content
    arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Cannot decode image from URL")

    detector = yolo_service.get_detector(body.model, body.confidence, body.iou_threshold)
    result = detector.detect_image(image)

    annotated_b64 = None
    if body.return_annotated:
        annotated = detector.get_annotated_image(image, result, "modern")
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    did = str(uuid.uuid4())
    return _build_response(did, result, annotated_b64,
                           result.image_shape, result.model_name)


@router.get("/history")
async def get_detection_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    all_items = list(_detection_store.values())
    total = len(all_items)
    items = all_items[offset: offset + limit]
    return {"items": items, "total": total, "limit": limit, "offset": offset}


@router.get("/{detection_id}")
async def get_detection(detection_id: str):
    result = _detection_store.get(detection_id)
    if result is None:
        raise HTTPException(404, "Detection not found")
    return result


@router.delete("/{detection_id}")
async def delete_detection(detection_id: str):
    if detection_id not in _detection_store:
        raise HTTPException(404, "Detection not found")
    del _detection_store[detection_id]
    return {"deleted": True, "detection_id": detection_id}
