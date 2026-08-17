from fastapi import APIRouter

from src.api.inference import triage_service

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {
        "api_status": "operacional",
        "modelo_carregado": triage_service.is_ready,
    }
