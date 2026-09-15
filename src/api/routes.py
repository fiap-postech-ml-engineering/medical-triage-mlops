"""Rotas da API de triagem."""

from fastapi import APIRouter, Request

from src.api import inference
from src.api.schemas import ClassifyRequest, ClassifyResponse, HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/classify", response_model=ClassifyResponse)
def classify(request: Request, payload: ClassifyRequest) -> ClassifyResponse:
    model = request.app.state.model
    resultado = inference.classify(model, payload.texto)
    return ClassifyResponse(**resultado)
