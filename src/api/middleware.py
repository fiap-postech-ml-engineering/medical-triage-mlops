"""Middlewares de observabilidade para a API."""

import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request

from src.config import get_settings
from src.logging_config import clear_request_id, set_request_id
from src.metrics import REQUEST_COUNT, REQUEST_LATENCY_SECONDS

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


def register_observability_middleware(app: FastAPI) -> None:
    """Registra middleware de latência e correlação de requisições."""

    @app.middleware("http")
    async def latency_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        token = set_request_id(request_id)
        started_at = perf_counter()
        status_code = 500
        response = None

        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            elapsed_s = perf_counter() - started_at
            latency_ms = round(elapsed_s * 1000, 2)
            path = request.url.path
            log_extra = {
                "event": "request.completed",
                "method": request.method,
                "path": path,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "client_ip": request.client.host if request.client else None,
            }

            latency_warn_ms = get_settings().LATENCY_WARN_MS
            log_fn = logger.warning if latency_ms >= latency_warn_ms else logger.info
            log_fn("request.completed", extra=log_extra)

            # /metrics fica de fora para não poluir a própria série de métricas.
            if path != "/metrics":
                REQUEST_COUNT.labels(
                    method=request.method, path=path, status_code=status_code
                ).inc()
                REQUEST_LATENCY_SECONDS.labels(method=request.method, path=path).observe(elapsed_s)

            if response is not None and REQUEST_ID_HEADER not in response.headers:
                response.headers[REQUEST_ID_HEADER] = request_id

            clear_request_id(token)
