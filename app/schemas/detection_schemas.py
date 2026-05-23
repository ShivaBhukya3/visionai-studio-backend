from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class BBoxSchema(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float
    w: float
    h: float


class DetectionSchema(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBoxSchema
    area: float
    color: str
    track_id: Optional[int] = None


class DetectionSummary(BaseModel):
    total_objects: int
    class_counts: dict
    avg_confidence: float
    processing_time_ms: float


class DetectionResponse(BaseModel):
    detection_id: str
    detections: List[DetectionSchema]
    summary: DetectionSummary
    annotated_image_base64: Optional[str] = None
    model_used: str
    image_dimensions: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total_images: int
    total_objects: int


class URLDetectionRequest(BaseModel):
    image_url: str
    confidence: float = Field(default=0.5, ge=0.1, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.1, le=1.0)
    classes: Optional[List[str]] = None
    model: str = "yolov8n.pt"
    return_annotated: bool = True
