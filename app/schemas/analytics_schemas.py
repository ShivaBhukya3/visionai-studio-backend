from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class ClassCount(BaseModel):
    name: str
    count: int


class DailyCount(BaseModel):
    date: str
    count: int


class AnalyticsSummary(BaseModel):
    total_detections: int
    total_sessions: int
    avg_confidence: float
    top_classes: List[ClassCount]
    period_days: int


class ModelBenchmarkResult(BaseModel):
    model_name: str
    avg_ms: float
    min_ms: float
    max_ms: float
    fps: float
    device: str
    iterations: int
