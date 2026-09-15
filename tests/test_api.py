"""Tests for the FastAPI application (`src.api.api`)."""

from contextlib import contextmanager

from fastapi.testclient import TestClient
import numpy as np
import pytest

from src.api import inference
from src.api.api import api
from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier
from src.config.settings import settings


class _FakePipeline:
    classes_ = np.array(["CLINICO_GERAL", "ESPECIALISTA"])

    def predict_proba(self, x):
        n = len(x)
        p_pos = np.full(n, 0.90)
        return np.column_stack([1.0 - p_pos, p_pos])


@contextmanager
def _client_with_fake_model(monkeypatch):
    """Sobe a app com o lifespan real, mas fazendo `load_model` devolver um fake
    em vez de ler o `.joblib` do disco.
    """
    fake_model = ThresholdedBinaryClassifier(pipeline=_FakePipeline(), threshold=0.40)
    monkeypatch.setattr(inference, "load_model", lambda _path: fake_model)
    with TestClient(api) as client:
        yield client


pytestmark = pytest.mark.api


def test_health_retorna_ok(monkeypatch):
    with _client_with_fake_model(monkeypatch) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_retorna_classificacao_e_probabilidades(monkeypatch):
    with _client_with_fake_model(monkeypatch) as client:
        response = client.post("/classify", json={"texto": "dor no peito e falta de ar"})

    assert response.status_code == 200
    body = response.json()
    assert body["classificacao"] == "ESPECIALISTA"
    assert body["probabilidade_especialista"] == pytest.approx(0.90)
    assert body["threshold_usado"] == pytest.approx(0.40)
    assert body["especialidade_provavel"] is None


def test_classify_rejeita_texto_vazio(monkeypatch):
    with _client_with_fake_model(monkeypatch) as client:
        response = client.post("/classify", json={"texto": ""})

    assert response.status_code == 422


@pytest.mark.slow
def test_classify_com_modelo_real_treinado():
    """Integração de ponta a ponta com o .joblib gerado por train.py.

    Requer que `models/modelo.joblib` já exista (uv run python -m
    src.binary_triage.train). Marcado slow porque depende do artefato
    treinado, não só de código.
    """
    if not settings.model_path.exists():
        pytest.skip("models/modelo.joblib não encontrado — rode o treino antes.")

    with TestClient(api) as client:
        response = client.post(
            "/classify",
            json={"texto": "Patient presents with chest pain and ST elevation on ECG."},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["classificacao"] in {"CLINICO_GERAL", "ESPECIALISTA"}
    assert 0.0 <= body["probabilidade_especialista"] <= 1.0
