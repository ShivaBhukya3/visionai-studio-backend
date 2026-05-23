from datetime import datetime, timedelta
from typing import Optional
from loguru import logger


class AnalyticsService:
    """In-memory analytics service (production would use DB queries)."""

    def __init__(self):
        self._sessions: list = []
        self._detections: list = []

    def record_session(self, session: dict) -> None:
        self._sessions.append({**session, "created_at": datetime.utcnow()})

    def record_detection(self, detection: dict) -> None:
        self._detections.append({**detection, "created_at": datetime.utcnow()})

    def get_summary(self, days: int = 7) -> dict:
        cutoff = datetime.utcnow() - timedelta(days=days)
        recent = [d for d in self._detections if d.get("created_at", datetime.min) > cutoff]

        class_counts: dict = {}
        confidences = []
        for d in recent:
            cn = d.get("class_name", "unknown")
            class_counts[cn] = class_counts.get(cn, 0) + 1
            confidences.append(d.get("confidence", 0))

        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        return {
            "total_detections": len(recent),
            "total_sessions": len(self._sessions),
            "avg_confidence": round(avg_conf, 4),
            "top_classes": [{"name": k, "count": v} for k, v in top_classes],
            "period_days": days,
        }

    def get_detections_over_time(self, days: int = 7) -> list:
        from collections import defaultdict
        counts: dict = defaultdict(int)
        cutoff = datetime.utcnow() - timedelta(days=days)
        for d in self._detections:
            ts = d.get("created_at", datetime.min)
            if ts > cutoff:
                date_str = ts.strftime("%Y-%m-%d")
                counts[date_str] += 1

        result = []
        for i in range(days):
            date = (datetime.utcnow() - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
            result.append({"date": date, "count": counts.get(date, 0)})
        return result


analytics_service = AnalyticsService()
