import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from app.database import Base


class DetectionSession(Base):
    __tablename__ = "detection_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_name = Column(String(255))
    source_type = Column(String(50))
    model_used = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    total_frames = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    avg_confidence = Column(Float)
    processing_time_ms = Column(Float)
    status = Column(String(50), default="active")
    metadata_ = Column("metadata", JSONB, default={})

    detections = relationship("DetectionRecord", back_populates="session",
                               cascade="all, delete-orphan")


class DetectionRecord(Base):
    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("detection_sessions.id"))
    frame_number = Column(Integer, default=0)
    class_name = Column(String(100), nullable=False)
    class_id = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float)
    y1 = Column(Float)
    x2 = Column(Float)
    y2 = Column(Float)
    cx = Column(Float)
    cy = Column(Float)
    width = Column(Float)
    height = Column(Float)
    area = Column(Float)
    track_id = Column(Integer)
    image_width = Column(Integer)
    image_height = Column(Integer)
    inference_time_ms = Column(Float)
    annotated_image_path = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("DetectionSession", back_populates="detections")
