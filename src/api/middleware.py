"""Middleware de logging de requisições."""

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        inicio = time.perf_counter()
        response = await call_next(request)
        latencia_ms = (time.perf_counter() - inicio) * 1000
        logger.info(
            "%s %s -> %d (%.2fms)",
            request.method,
            request.url.path,
            response.status_code,
            latencia_ms,
        )
        return response
