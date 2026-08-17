"""Valida o ambiente antes de rodar a API ou o pipeline de treino.

Confere que `.env`/variáveis de ambiente carregam em `Settings` sem erro e que
o artefato do modelo configurado (`MODEL_PATH` ou `ONNX_MODEL_PATH`, conforme
`USE_ONNX`) existe em disco. Falha cedo com mensagem clara em vez de deixar o
erro estourar no startup da API.

Uso: `python -m scripts.validate_env` (ou `uv run python -m scripts.validate_env`).
"""

import logging
import sys

from pydantic import ValidationError

from src.config import BASE_DIR, Settings, get_settings

logger = logging.getLogger(__name__)


def _load_settings() -> Settings:
    """Instancia `Settings` via `get_settings()`, deixando a `ValidationError` propagar.

    Usa `get_settings()` (não `Settings()` diretamente) porque é ela quem chama
    `setup_logging(settings)` — sem isso, os `logger.info`/`logger.warning` abaixo
    não têm handler configurado e o script roda mudo.
    """
    return get_settings()


def main() -> None:
    """Ponto de entrada CLI: valida `Settings` e a presença do artefato do modelo."""
    try:
        settings = _load_settings()
    except ValidationError as exc:
        logger.error("Settings inválido — corrija o .env:\n%s", exc)
        sys.exit(1)

    logger.info("Settings carregado com sucesso (APP_ENV=%s).", settings.APP_ENV)

    model_path = BASE_DIR / (settings.ONNX_MODEL_PATH if settings.USE_ONNX else settings.MODEL_PATH)
    if not model_path.exists():
        logger.warning(
            "Artefato do modelo não encontrado em %s. Rode `make train`%s antes de subir a API.",
            model_path,
            " && make onnx-export" if settings.USE_ONNX else "",
        )
    else:
        logger.info("Artefato do modelo encontrado em %s.", model_path)


if __name__ == "__main__":
    main()
