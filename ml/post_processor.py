from typing import Optional
from ml.yolo_detector import Detection, BBox


class PostProcessor:

    def apply_nms(self, detections: list, iou_threshold: float = 0.45) -> list:
        if not detections:
            return []

        sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
        keep = []

        while sorted_dets:
            best = sorted_dets.pop(0)
            keep.append(best)
            sorted_dets = [
                d for d in sorted_dets
                if self._iou(best.bbox, d.bbox) < iou_threshold
            ]
        return keep

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def filter_by_class(self, detections: list, allowed_classes: list) -> list:
        if not allowed_classes:
            return detections
        allowed_lower = {c.lower() for c in allowed_classes}
        return [d for d in detections if d.class_name.lower() in allowed_lower]

    def filter_by_confidence(self, detections: list, min_confidence: float) -> list:
        return [d for d in detections if d.confidence >= min_confidence]

    def filter_by_region(self, detections: list, roi: dict) -> list:
        rx1 = roi.get("x1", 0)
        ry1 = roi.get("y1", 0)
        rx2 = roi.get("x2", float("inf"))
        ry2 = roi.get("y2", float("inf"))

        def inside(d: Detection) -> bool:
            b = d.bbox
            return b.cx >= rx1 and b.cx <= rx2 and b.cy >= ry1 and b.cy <= ry2

        return [d for d in detections if inside(d)]

    def count_by_class(self, detections: list) -> dict:
        counts: dict = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def compute_detection_density(self, detections: list,
                                  image_shape: dict) -> float:
        w = image_shape.get("width", 1)
        h = image_shape.get("height", 1)
        area_kpx = (w * h) / 1000.0
        return round(len(detections) / area_kpx if area_kpx > 0 else 0.0, 4)

    def track_objects_simple(self, prev_detections: list,
                             curr_detections: list,
                             iou_threshold: float = 0.3) -> list:
        if not prev_detections:
            for i, d in enumerate(curr_detections):
                d.track_id = i + 1
            return curr_detections

        max_id = max((d.track_id or 0) for d in prev_detections)

        used_prev = set()
        for curr in curr_detections:
            best_iou = 0.0
            best_prev = None
            for j, prev in enumerate(prev_detections):
                if j in used_prev:
                    continue
                iou = self._iou(curr.bbox, prev.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_prev = (j, prev)

            if best_prev and best_iou >= iou_threshold:
                idx, matched = best_prev
                curr.track_id = matched.track_id
                used_prev.add(idx)
            else:
                max_id += 1
                curr.track_id = max_id

        return curr_detections
