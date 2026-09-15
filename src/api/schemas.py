"""Schemas Pydantic dos contratos de request/response da API."""

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    texto: str = Field(min_length=1, description="Texto do laudo médico a ser classificado.")


class ClassifyResponse(BaseModel):
    classificacao: str
    probabilidade_especialista: float
    probabilidade_clinico_geral: float
    especialidade_provavel: str | None
    probabilidades_especialidade: dict[str, float] | None
    threshold_usado: float


class HealthResponse(BaseModel):
    status: str
