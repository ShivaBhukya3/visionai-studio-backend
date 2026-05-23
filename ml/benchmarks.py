import time
from typing import Optional
import numpy as np
from loguru import logger


def run_benchmark(model_name: str = "yolov8n.pt",
                  n_iterations: int = 100,
                  image_size: int = 640) -> dict:
    try:
        from ml.yolo_detector import YOLODetector
        detector = YOLODetector(model_path=model_name)
        return detector.benchmark(n_iterations)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


def compare_models(models: Optional[list] = None,
                   n_iterations: int = 50) -> dict:
    if models is None:
        models = ["yolov8n.pt", "yolov8s.pt"]

    results = {}
    for model_name in models:
        logger.info(f"Benchmarking {model_name}...")
        results[model_name] = run_benchmark(model_name, n_iterations)
    return results
