from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    laudo_texto: str = Field(
        min_length=1, description="Texto do laudo/relato clínico do paciente."
    )


class TriageResponse(BaseModel):
    especialidade_recomendada: str
    confianca: float = Field(ge=0.0, le=1.0)
    probabilidades: dict[str, float]
