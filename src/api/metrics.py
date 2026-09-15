"""Métricas Prometheus da API e middleware de instrumentação HTTP."""

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

http_requests_total = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas pela API.",
    ["method", "path", "status_code"],
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "Duração das requisições HTTP, incluindo overhead de rede/serialização.",
    ["method", "path"],
)

triage_classifications_total = Counter(
    "triage_classifications_total",
    "Total de laudos classificados, por classificação binária.",
    ["classificacao"],
)

triage_inference_duration_seconds = Histogram(
    "triage_inference_duration_seconds",
    "Duração isolada da inferência do modelo (sem overhead de rede/serialização).",
)


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        inicio = time.perf_counter()
        response = await call_next(request)
        duracao = time.perf_counter() - inicio

        path = request.url.path
        http_requests_total.labels(
            method=request.method, path=path, status_code=response.status_code
        ).inc()
        http_request_duration_seconds.labels(method=request.method, path=path).observe(duracao)
        return response


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
