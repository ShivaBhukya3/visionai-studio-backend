import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Date
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.database import Base


class ModelBenchmark(Base):
    __tablename__ = "model_benchmarks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(100), nullable=False)
    avg_inference_ms = Column(Float)
    min_inference_ms = Column(Float)
    max_inference_ms = Column(Float)
    fps = Column(Float)
    map50 = Column(Float)
    device = Column(String(50))
    iterations = Column(Integer, default=100)
    benchmark_date = Column(DateTime, default=datetime.utcnow)


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    snapshot_date = Column(Date, default=datetime.utcnow)
    total_sessions = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    top_classes = Column(JSONB, default=[])
    model_usage = Column(JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
