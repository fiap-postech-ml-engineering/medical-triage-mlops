from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from src.api.app import app, home
from src.api.middleware import REQUEST_ID_HEADER
from src.config import Settings


def test_home_returns_service_status():
    assert home() == {
        "service": "Medical Triage API",
        "version": "0.1.0",
        "env": "development",
        "status": "ok",
    }


def test_app_metadata_uses_model_version():
    assert app.title == "Medical Triage API"
    assert app.version == "0.1.0"


def test_settings_reject_invalid_app_env():
    with pytest.raises(ValidationError):
        Settings(APP_ENV="local")


def test_response_includes_request_id_header():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER]


def test_response_reuses_client_provided_request_id():
    client = TestClient(app)

    response = client.get("/", headers={REQUEST_ID_HEADER: "my-custom-id"})

    assert response.headers[REQUEST_ID_HEADER] == "my-custom-id"


def test_metrics_endpoint_exposes_prometheus_format():
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "http_requests_total" in response.text


def test_classify_returns_503_when_model_not_ready(monkeypatch):
    from src.api.inference import triage_service

    monkeypatch.setattr(triage_service, "load", lambda: None)
    monkeypatch.setattr(triage_service, "_sklearn_model", None)
    monkeypatch.setattr(triage_service, "_use_onnx", False)

    client = TestClient(app)
    response = client.post("/classify", json={"laudo_texto": "dor no peito"})

    assert response.status_code == 503


def test_classify_returns_especialidade_when_model_ready(monkeypatch):
    from src.api.inference import triage_service

    monkeypatch.setattr(triage_service, "load", lambda: None)
    monkeypatch.setattr(triage_service, "_sklearn_model", object())
    monkeypatch.setattr(triage_service, "_use_onnx", False)
    monkeypatch.setattr(
        triage_service,
        "classify",
        lambda texto: ("cardiologia", 0.82, {"cardiologia": 0.82, "clinica_geral": 0.18}),
    )

    client = TestClient(app)
    response = client.post("/classify", json={"laudo_texto": "dor no peito e falta de ar"})

    assert response.status_code == 200
    body = response.json()
    assert body["especialidade_recomendada"] == "cardiologia"
    assert body["confianca"] == pytest.approx(0.82)
    assert body["probabilidades"]["cardiologia"] == pytest.approx(0.82)


def test_classify_rejects_empty_payload():
    client = TestClient(app)

    response = client.post("/classify", json={"laudo_texto": ""})

    assert response.status_code == 422


def test_health_reports_model_not_ready(monkeypatch):
    from src.api.inference import triage_service

    monkeypatch.setattr(triage_service, "_sklearn_model", None)
    monkeypatch.setattr(triage_service, "_use_onnx", False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"api_status": "operacional", "modelo_carregado": False}


def test_health_reports_model_ready(monkeypatch):
    from src.api.inference import triage_service

    monkeypatch.setattr(triage_service, "_sklearn_model", object())
    monkeypatch.setattr(triage_service, "_use_onnx", False)

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"api_status": "operacional", "modelo_carregado": True}
