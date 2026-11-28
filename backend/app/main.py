import asyncio
from contextlib import asynccontextmanager
from functools import partial
import hmac
import os
import uuid

from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text as sa_text
import torch

from .api.routes.agronomy import router as agronomy_router
from .api.routes.auth import router as auth_router
from .api.routes.environment import router as environment_router
from .api.routes.field import router as field_router
from .api.routes.history import router as history_router
from .api.routes.jobs import router as jobs_router
from .api.routes.leaf import router as leaf_router
from .api.routes.model_registry import router as model_registry_router
from .core.config import settings
from .core.limiter import limiter
from .core.logger import logger
from .db.database import engine, init_db
from .services.inference_service import InferenceService

# Rate limiter moved to .core.limiter to avoid circular imports


# ---------------------------------------------------------------------------
# API Key auth — timing-safe comparison (prevents timing attacks)
# ---------------------------------------------------------------------------
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    if settings.API_KEY:
        provided = api_key or ""
        if not hmac.compare_digest(provided, settings.API_KEY):
            raise HTTPException(
                status_code=401,
                detail="Invalid or missing API key. Provide it via the X-API-Key header.",  # noqa: E501
            )


# ---------------------------------------------------------------------------
# Lifespan — model loading + DB init
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")

    # Init SQLite tables with timeout guard
    try:
        async with asyncio.timeout(2.0):
            await init_db()
        logger.info("Database initialised.")
    except Exception as e:
        logger.warning(f"Database init warning / timeout: {e}")

    # Load ML models in a thread if not already set
    if not getattr(app.state, "inference_service", None):
        try:
            model_paths = settings.get_model_paths()
            loop = asyncio.get_running_loop()
            svc = await loop.run_in_executor(
                None,
                partial(
                    InferenceService, model_paths=model_paths if model_paths else None
                ),
            )
            app.state.inference_service = svc
        except Exception as e:
            logger.warning(f"Model loading warning: {e}")
            app.state.inference_service = InferenceService()

    svc = getattr(app.state, "inference_service", None)
    if svc:
        logger.info(f"Inference engine ready — device: {svc.device}")
    yield
    logger.info("Shutting down AgriVision AI.")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Agricultural disease intelligence — GraphSAGE + 1D-CNN + BiLSTM.",  # noqa: E501
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Mount static model weights for frontend access (Confusion Matrix,
# classes.json)
os.makedirs(settings.MODEL_BASE_PATH, exist_ok=True)
app.mount(
    f"{settings.API_V1_STR}/model_weights",
    StaticFiles(directory=settings.MODEL_BASE_PATH),
    name="model_weights_api",
)
app.mount(
    "/model_weights",
    StaticFiles(directory=settings.MODEL_BASE_PATH),
    name="model_weights_root",
)


# ---------------------------------------------------------------------------
# Request ID middleware — injects X-Request-ID into every request/response
# ---------------------------------------------------------------------------
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    logger.debug(f"[{request_id}] {request.method} {request.url.path}")
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Security headers middleware (CSP, Permissions-Policy, HSTS hint, etc.)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data: blob:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "
        "connect-src 'self' http://localhost:* ws://localhost:* https://*.vercel.app https://*.onrender.com"
    )
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=(), payment=()"
    )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    return response


# ---------------------------------------------------------------------------
# Upload size guard
# ---------------------------------------------------------------------------
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    if request.method == "POST":
        content_length = request.headers.get("content-length")
        if (
            content_length
            and int(content_length) > settings.MAX_FILE_SIZE_BYTES
        ):
            return JSONResponse(
                status_code=413,
                content={
                    "detail": f"File too large. Maximum {settings.MAX_FILE_SIZE_MB} MB."  # noqa: E501
                },
            )
    return await call_next(request)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app|http://localhost:\d+|http://127\.0\.0\.1:\d+|http://192\.168\.\d+\.\d+:\d+|http://10\.\d+\.\d+\.\d+:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Include routers
# ---------------------------------------------------------------------------

PREFIX = settings.API_V1_STR
app.include_router(field_router, prefix=PREFIX, tags=["Field Analysis"])
app.include_router(leaf_router, prefix=PREFIX, tags=["Leaf Analysis"])
app.include_router(history_router, prefix=PREFIX, tags=["History"])
app.include_router(auth_router, prefix=PREFIX, tags=["Auth"])
app.include_router(environment_router, prefix=PREFIX, tags=["Environment & External APIs"])
# Root alias for frontend widgets (e.g. WeatherWidget) that call VITE_API_URL/environment directly
app.include_router(environment_router, include_in_schema=False)
app.include_router(jobs_router, prefix=PREFIX, tags=["Async Job Queue"])
app.include_router(agronomy_router, prefix=PREFIX, tags=["Agronomy & Advisory"])
app.include_router(model_registry_router, prefix=PREFIX, tags=["MLOps & Model Registry"])


@app.get(f"{PREFIX}/samples", tags=["Benchmark Evaluation"])
async def get_benchmark_samples():
    """Returns metadata for the benchmark sample evaluation suite."""
    return {
        "benchmark_suite": "AgriPulse 2.0 Real-World Benchmark",
        "samples": [
            {
                "id": "healthy",
                "label": "Healthy Foliage",
                "crop": "Oryza sativa / Paddy",
                "condition": "Negative Control Baseline",
                "expected_prediction": "Healthy",
                "expected_action": "Routine monitoring only",
            },
            {
                "id": "blast",
                "label": "Rice Blast",
                "crop": "Oryza sativa / Paddy",
                "condition": "Magnaporthe oryzae (Foliar necrosis)",
                "expected_prediction": "Rice Blast",
                "icar_guidance": "Tricyclazole 75% WP @ 0.6 g/L every 14 days",
            },
            {
                "id": "brown_spot",
                "label": "Brown Spot",
                "crop": "Oryza sativa / Paddy",
                "condition": "Bipolaris oryzae (Fungal leaf spot)",
                "expected_prediction": "Brown Spot",
                "icar_guidance": "Propiconazole 25% EC @ 1 ml/L at boot-leaf stage",
            },
            {
                "id": "ood_trap",
                "label": "Non-Crop Mechanical Trap",
                "crop": "None (Out-of-Distribution)",
                "condition": "Non-foliar synthetic pattern",
                "expected_prediction": "Withheld / Low Certainty",
                "ood_behavior": "Triggers MSP & Energy-based entropy gate rejection",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Core & Kubernetes Health Probes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": f"{settings.PROJECT_NAME} Online",
        "status": "Ready",
        "docs": f"{PREFIX}/docs",
    }


@app.get("/healthz")
@app.get(f"{PREFIX}/healthz")
async def liveness_probe():
    """Kubernetes Liveness probe: returns 200 immediately if process is alive."""
    return {"status": "ok", "probe": "liveness"}


@app.get("/readyz")
@app.get(f"{PREFIX}/readyz")
async def readiness_probe():
    """Kubernetes Readiness probe: validates database and inference engine readiness."""
    db_ok = False
    try:
        from pathlib import Path
        import sqlite3
        candidate_paths = [
            Path("backend/data/agrivision.db"),
            Path("data/agrivision.db"),
        ]
        db_file = next((p for p in candidate_paths if p.exists()), None)
        if db_file:
            with sqlite3.connect(str(db_file), timeout=1.0) as s_conn:
                s_conn.execute("SELECT 1")
            db_ok = True
        else:
            db_ok = True  # Memory / freshly provisioned
    except Exception:
        pass

    svc = getattr(app.state, "inference_service", None)
    models_ok = svc is not None

    if db_ok and models_ok:
        return {"status": "ready", "database": "ok", "models": "ok"}
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": db_ok, "models": models_ok},
        )


@app.get("/metrics")
@app.get(f"{PREFIX}/metrics")
async def prometheus_metrics():
    """Prometheus-compatible plain text metrics endpoint."""
    from fastapi.responses import PlainTextResponse

    svc = getattr(app.state, "inference_service", None)
    loaded_models_count = (
        len(
            set(getattr(svc, "loaded_models", set()))
            | set(getattr(svc, "ort_sessions", {}).keys())
        )
        if svc
        else 0
    )

    lines = [
        "# HELP agripulse_system_info AgriPulse system metadata",
        "# TYPE agripulse_system_info gauge",
        f'agripulse_system_info{{version="{settings.VERSION}",device="{svc.device if svc else "none"}"}} 1',
        "# HELP agripulse_models_loaded Total number of active ML sub-models",
        "# TYPE agripulse_models_loaded gauge",
        f"agripulse_models_loaded {loaded_models_count}",
        "# HELP agripulse_service_ready Service readiness state (1=ready, 0=not ready)",
        "# TYPE agripulse_service_ready gauge",
        f"agripulse_service_ready {1 if svc else 0}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/health")
@app.get(f"{PREFIX}/health")
async def health_check(request: Request):
    svc = getattr(app.state, "inference_service", None)

    # DB connectivity probe
    db_ok = False
    try:
        async with engine.connect() as conn:
            await conn.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.warning("Health check: database connectivity failed.")

    # Hardware & Memory diagnostics
    device_name = str(svc.device) if svc else "initializing"
    loaded_weights_map = settings.get_model_paths() or {}
    pt_keys = set(getattr(svc, "loaded_models", set())) if svc else set()
    onnx_keys = set(getattr(svc, "ort_sessions", {}).keys()) if svc else set()
    loaded_keys = sorted(pt_keys | onnx_keys)

    memory_stats = {}
    if torch.cuda.is_available() and "cuda" in device_name:
        memory_stats["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        memory_stats["vram_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
        memory_stats["device_name"] = torch.cuda.get_device_name(0)
    else:
        memory_stats["execution_mode"] = "High-Performance CPU Fast-Path"

    return {
        "status": "healthy" if db_ok else "degraded",
        "version": settings.VERSION,
        "platform": "AgriPulse 2.0 Enterprise Agricultural Intelligence",
        "device": device_name,
        "models_loaded": svc is not None,
        "weights_loaded": (
            len(loaded_keys) > 0 or len(loaded_weights_map) > 0
        ),
        "active_models": loaded_keys,
        "available_weights": sorted(
            set(loaded_weights_map.keys()) | onnx_keys
        ),
        "database": "connected" if db_ok else "error",
        "telemetry": memory_stats,
    }


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
