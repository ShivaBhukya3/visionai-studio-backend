import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.config import settings
from app.routers import detection, stream, analytics, health
from app.services.yolo_service import yolo_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    # Pre-load default model
    try:
        detector = yolo_service.get_detector()
        logger.info(f"Default model loaded: {settings.YOLO_MODEL}")
    except Exception as e:
        logger.warning(f"Model pre-load skipped: {e}")

    yield

    logger.info("Shutting down VisionAI Studio")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Real-Time Object Detection Platform powered by YOLOv8",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Middleware
cors_origins = list(settings.CORS_ORIGINS)
if settings.PRODUCTION_CORS_ORIGIN:
    cors_origins.append(settings.PRODUCTION_CORS_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https://.*\.vercel\.app|https://.*\.hf\.space",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(
        round((time.perf_counter() - t0) * 1000, 2))
    return response


# Routers
app.include_router(health.router)
app.include_router(detection.router)
app.include_router(stream.router)
app.include_router(analytics.router)


@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "running",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    origin = request.headers.get("origin", "")
    headers = {}
    if origin:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
        headers=headers,
    )
