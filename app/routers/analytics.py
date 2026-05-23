from fastapi import APIRouter, Query, HTTPException
from app.services.analytics_service import analytics_service
from ml.model_manager import model_manager, MODEL_CATALOG
from ml.benchmarks import run_benchmark

router = APIRouter(prefix="/api/v1/analytics", tags=["analytics"])


@router.get("/summary")
async def get_summary(days: int = Query(default=7, ge=1, le=90)):
    return analytics_service.get_summary(days)


@router.get("/detections-over-time")
async def detections_over_time(days: int = Query(default=7, ge=1, le=90)):
    return analytics_service.get_detections_over_time(days)


@router.get("/models")
async def list_models():
    return model_manager.list_available_models()


@router.post("/models/{model_name}/download")
async def download_model(model_name: str):
    valid = [m["name"] for m in MODEL_CATALOG]
    if model_name not in valid:
        raise HTTPException(400, f"Unknown model: {model_name}")
    success = model_manager.download_model(model_name)
    return {"success": success, "model": model_name}


@router.post("/models/{model_name}/benchmark")
async def benchmark_model(
    model_name: str,
    iterations: int = Query(default=50, ge=10, le=200),
):
    result = run_benchmark(model_name, iterations)
    return result


@router.get("/models/{model_name}/info")
async def model_info(model_name: str):
    info = model_manager.get_model_info(model_name)
    if not info:
        raise HTTPException(404, "Model not found")
    return info


@router.post("/models/{model_name}/activate")
async def activate_model(model_name: str):
    from app.services.yolo_service import yolo_service
    yolo_service.switch_model(model_name)
    return {"activated": model_name}
