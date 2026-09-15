"""Ponto de entrada da API FastAPI (`src.api.api:api`)."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI

from src.api import inference
from src.api.metrics import MetricsMiddleware
from src.api.middleware import RequestLoggingMiddleware
from src.api.routes import router
from src.config.settings import settings

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Carregar o modelo aqui faz a aplicação falhar no startup (fail-fast) se o
    # artefato não existir ou estiver corrompido, em vez de subir e falhar
    # silenciosamente na primeira requisição.
    app.state.model = inference.load_model(settings.model_path)
    yield


api = FastAPI(title="Medical Triage API", lifespan=lifespan)
api.add_middleware(RequestLoggingMiddleware)
api.add_middleware(MetricsMiddleware)
api.include_router(router)
