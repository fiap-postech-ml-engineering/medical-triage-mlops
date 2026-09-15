"""Rotas da API de triagem."""

from fastapi import APIRouter, Request
from starlette.responses import Response

from src.api import inference
from src.api.metrics import metrics_response
from src.api.schemas import ClassifyRequest, ClassifyResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/metrics")
def metrics() -> Response:
    return metrics_response()


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, payload: ClassifyRequest) -> ClassifyResponse:
    model = request.app.state.model
    resultado = inference.classify(model, payload.texto)
    return ClassifyResponse(**resultado)
