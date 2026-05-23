---
title: VisionAI Studio API
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# VisionAI Studio — Backend API

Real-Time Object Detection Platform powered by YOLOv8 + FastAPI + WebSockets.

## Endpoints
- `GET /health` — health check
- `POST /api/v1/detect/image` — detect objects in an image
- `WS /ws/stream/{client_id}` — real-time webcam/video frame stream
- `GET /docs` — Swagger UI
