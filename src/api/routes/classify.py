from fastapi import APIRouter, HTTPException

from src.api.inference import triage_service
from src.api.schemas import TriageRequest, TriageResponse

router = APIRouter()


@router.post("/classify", response_model=TriageResponse)
async def classify(payload: TriageRequest) -> TriageResponse:
    if not triage_service.is_ready:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não disponível — rode `make train` (e `make onnx-export` se USE_ONNX=true).",
        )
    especialidade, confianca, probabilidades = triage_service.classify(payload.laudo_texto)
    return TriageResponse(
        especialidade_recomendada=especialidade,
        confianca=confianca,
        probabilidades=probabilidades,
    )
