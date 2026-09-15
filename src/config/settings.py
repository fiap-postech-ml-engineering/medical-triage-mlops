"""Configurações da aplicação, carregadas de variáveis de ambiente/.env."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    model_path: Path = BASE_DIR / "models" / "modelo.joblib"


settings = Settings()
