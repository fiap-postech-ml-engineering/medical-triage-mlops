from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.inference import triage_service
from src.api.middleware import register_observability_middleware
from src.api.routes.classify import router as classify_router
from src.api.routes.health import router as health_router
from src.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_settings()
    triage_service.load()
    yield


settings = get_settings()

app = FastAPI(
    title="Medical Triage API",
    version=settings.MODEL_VERSION,
    lifespan=lifespan,
)

register_observability_middleware(app)
app.include_router(classify_router)
app.include_router(health_router)


@app.get("/")
def home():
    return {
        "service": "Medical Triage API",
        "version": settings.MODEL_VERSION,
        "env": settings.APP_ENV,
        "status": "ok",
    }


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
