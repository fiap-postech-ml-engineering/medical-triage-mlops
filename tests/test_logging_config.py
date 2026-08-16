import json
import logging

from src.config import Settings
from src.logging_config import setup_logging


def _settings(**overrides) -> Settings:
    defaults = {"_env_file": None}
    defaults.update(overrides)
    return Settings(**defaults)


def test_setup_logging_creates_handlers():
    setup_logging(_settings())

    logger = logging.getLogger()
    assert len(logger.handlers) > 0


def test_console_handler_always_present():
    setup_logging(_settings())

    logger = logging.getLogger()
    has_stream = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    assert has_stream


def test_file_handler_present_in_production():
    setup_logging(_settings(APP_ENV="production"))

    logger = logging.getLogger()
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert has_file


def test_no_file_handler_in_development():
    setup_logging(_settings(APP_ENV="development"))

    logger = logging.getLogger()
    has_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert not has_file


def test_json_formatter_produces_valid_payload():
    setup_logging(_settings(LOG_FORMAT="json"))

    logger = logging.getLogger()
    stream_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )

    record = logging.LogRecord(
        name="tests.logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="health check",
        args=(),
        exc_info=None,
    )

    payload = json.loads(stream_handler.format(record))

    assert payload["message"] == "health check"
    assert payload["level"] == "INFO"
    assert payload["request_id"] == "-"
    assert payload["environment"] == "development"
    assert payload["location"] == "test_logging_config:1"


def test_text_formatter_includes_request_id_placeholder():
    setup_logging(_settings(LOG_FORMAT="text"))

    logger = logging.getLogger()
    stream_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )

    record = logging.LogRecord(
        name="tests.logging",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="health check",
        args=(),
        exc_info=None,
    )

    formatted = stream_handler.format(record)

    assert "request_id=-" in formatted
    assert "health check" in formatted
    assert "test_logging_config:1" in formatted


def test_text_formatter_pads_columns_to_fixed_width():
    setup_logging(_settings(LOG_FORMAT="text"))

    logger = logging.getLogger()
    stream_handler = next(
        h
        for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )

    short_record = logging.LogRecord(
        name="a",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="short",
        args=(),
        exc_info=None,
    )
    long_record = logging.LogRecord(
        name="src.tracking.mlflow_utils.long.module.name",
        level=logging.WARNING,
        pathname=__file__,
        lineno=999,
        msg="long",
        args=(),
        exc_info=None,
    )

    short_line = stream_handler.format(short_record)
    long_line = stream_handler.format(long_record)

    short_prefix_len = len(short_line.split(" - request_id=")[0])
    long_prefix_len = len(long_line.split(" - request_id=")[0])
    assert short_prefix_len == long_prefix_len


def test_setup_logging_is_idempotent_without_handler_duplication():
    setup_logging(_settings())
    logger = logging.getLogger()
    first_handler_count = len(logger.handlers)

    setup_logging(_settings())
    second_handler_count = len(logger.handlers)

    assert first_handler_count == second_handler_count == 1


def test_setup_logging_switches_handlers_when_environment_changes():
    setup_logging(_settings(APP_ENV="production"))
    assert len(logging.getLogger().handlers) == 2

    setup_logging(_settings(APP_ENV="development"))
    assert len(logging.getLogger().handlers) == 1

    setup_logging(_settings(APP_ENV="production"))
    assert len(logging.getLogger().handlers) == 2


def test_setup_logging_applies_log_level():
    setup_logging(_settings(LOG_LEVEL="DEBUG"))
    assert logging.getLogger().level == logging.DEBUG
