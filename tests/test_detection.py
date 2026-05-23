import pytest
import numpy as np
from ml.yolo_detector import YOLODetector, DetectionResult
from ml.post_processor import PostProcessor


@pytest.fixture
def detector():
    return YOLODetector("yolov8n.pt")


@pytest.fixture
def post():
    return PostProcessor()


@pytest.fixture
def blank_image():
    return np.zeros((640, 640, 3), dtype=np.uint8)


def test_detector_initializes(detector):
    assert detector is not None
    assert detector.model_name == "yolov8n.pt"


def test_detect_image_returns_result(detector, blank_image):
    result = detector.detect_image(blank_image)
    assert isinstance(result, DetectionResult)
    assert result.image_shape["width"] == 640
    assert result.image_shape["height"] == 640
    assert isinstance(result.detections, list)
    assert result.inference_time_ms >= 0


def test_detect_image_shape(detector):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect_image(img)
    assert result.image_shape["width"] == 640
    assert result.image_shape["height"] == 480


def test_get_annotated_image(detector, blank_image):
    result = detector.detect_image(blank_image)
    annotated = detector.get_annotated_image(blank_image, result, "modern")
    assert annotated.shape == blank_image.shape


def test_get_annotated_image_styles(detector, blank_image):
    result = detector.detect_image(blank_image)
    for style in ["modern", "minimal", "heatmap"]:
        ann = detector.get_annotated_image(blank_image, result, style)
        assert ann is not None


def test_get_model_info(detector):
    info = detector.get_model_info()
    assert "model_name" in info
    assert "device" in info


def test_get_class_colors(detector):
    colors = detector.get_class_colors()
    assert isinstance(colors, dict)
    assert len(colors) > 0
    assert "person" in colors


def test_post_processor_nms(post):
    from ml.yolo_detector import Detection, BBox
    dets = [
        Detection(0, "person", 0.9, BBox(10, 10, 100, 100, 55, 55, 90, 90), 8100, 1.0),
        Detection(0, "person", 0.7, BBox(15, 15, 105, 105, 60, 60, 90, 90), 8100, 1.0),
    ]
    result = post.apply_nms(dets, iou_threshold=0.5)
    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_post_processor_filter_class(post):
    from ml.yolo_detector import Detection, BBox
    dets = [
        Detection(0, "person", 0.9, BBox(0, 0, 100, 100, 50, 50, 100, 100), 10000, 1.0),
        Detection(2, "car", 0.8, BBox(0, 0, 100, 100, 50, 50, 100, 100), 10000, 1.0),
    ]
    result = post.filter_by_class(dets, ["person"])
    assert len(result) == 1
    assert result[0].class_name == "person"


def test_post_processor_confidence_filter(post):
    from ml.yolo_detector import Detection, BBox
    dets = [
        Detection(0, "person", 0.9, BBox(0, 0, 100, 100, 50, 50, 100, 100), 10000, 1.0),
        Detection(0, "cat", 0.3, BBox(0, 0, 100, 100, 50, 50, 100, 100), 10000, 1.0),
    ]
    result = post.filter_by_confidence(dets, 0.5)
    assert len(result) == 1
    assert result[0].confidence == 0.9
