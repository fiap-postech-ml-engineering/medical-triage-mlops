"""Métricas Prometheus expostas pela API (requisitos da Fase 3: total de
requisições, latência e taxa de erro)."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total de requisições HTTP recebidas.",
    ["method", "path", "status_code"],
)

REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_duration_seconds",
    "Latência das requisições HTTP, em segundos.",
    ["method", "path"],
)
