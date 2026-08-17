"""Configuração centralizada de logging."""

from contextvars import ContextVar, Token
from datetime import UTC, datetime
import json
import logging
from typing import TYPE_CHECKING

from src.config import LOGS_DIR

if TYPE_CHECKING:
    from src.config import Settings

_REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="-")

_EXTRA_FIELDS = (
    "event",
    "method",
    "path",
    "status_code",
    "latency_ms",
    "client_ip",
)


def set_request_id(request_id: str) -> Token:
    """Salva o request_id no contexto atual para enriquecimento dos logs."""
    return _REQUEST_ID_CTX.set(request_id)


def clear_request_id(token: Token) -> None:
    """Restaura o valor anterior de request_id no contexto."""
    _REQUEST_ID_CTX.reset(token)


def get_request_id() -> str:
    """Retorna o request_id atual do contexto de execução."""
    return _REQUEST_ID_CTX.get()


def _truncate(value: str, width: int) -> str:
    """Encurta `value` para `width` chars, marcando corte com '...'."""
    if len(value) <= width:
        return value
    return value[: width - 3] + "..."


class TextFormatter(logging.Formatter):
    """Formata logs em texto com colunas de largura fixa."""

    LEVEL_WIDTH = 8
    LOGGER_WIDTH = 25
    LOCATION_WIDTH = 22
    REQUEST_ID_WIDTH = 36

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname.ljust(self.LEVEL_WIDTH)
        logger_name = _truncate(record.name, self.LOGGER_WIDTH).ljust(self.LOGGER_WIDTH)
        location = _truncate(f"{record.module}:{record.lineno}", self.LOCATION_WIDTH).ljust(
            self.LOCATION_WIDTH
        )
        request_id = _truncate(get_request_id(), self.REQUEST_ID_WIDTH).ljust(
            self.REQUEST_ID_WIDTH
        )

        line = (
            f"{self.formatTime(record)} - [{level}] - {logger_name} - "
            f"{location} - request_id={request_id} - {record.getMessage()}"
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


class JsonFormatter(logging.Formatter):
    """Formata logs em JSON para observabilidade."""

    def __init__(self, environment: str):
        super().__init__()
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "location": f"{record.module}:{record.lineno}",
            "message": record.getMessage(),
            "environment": self.environment,
            "request_id": get_request_id(),
        }

        for field in _EXTRA_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def setup_logging(settings: "Settings") -> None:
    """Configura o root logger a partir das Settings da aplicação."""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if settings.APP_ENV == "production":
        LOGS_DIR.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(LOGS_DIR / "app.log", encoding="utf-8"))

    if settings.LOG_FORMAT == "json":
        formatter: logging.Formatter = JsonFormatter(environment=settings.APP_ENV)
    else:
        formatter = TextFormatter()

    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(settings.LOG_LEVEL)
