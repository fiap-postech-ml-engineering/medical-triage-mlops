from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

# As classes de especialidade não são fixas no código: vêm do dataset de treino
# (coluna LABEL_COLUMN) e ficam gravadas em `classifier.classes_` no artefato
# treinado. Com o dataset real usado no projeto (Medical Abstracts TC Corpus,
# ver notebooks/01_eda_medical_abstracts.ipynb), as classes são: cardiologia,
# clinica_geral, gastroenterologia, neurologia, oncologia.


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    # App
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="text")
    LATENCY_WARN_MS: int = Field(default=1000, gt=0)
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    MODEL_VERSION: str = Field(default="0.1.0")

    # Modelo / inferência
    # Quando True, a API carrega o runtime ONNX (models/classifier.onnx);
    # quando False, carrega o pipeline scikit-learn (models/classifier.joblib).
    USE_ONNX: bool = Field(default=False)
    # models/classifier.joblib guarda um sklearn Pipeline(TfidfVectorizer, RandomForestClassifier)
    # já com o vetorizador embutido — um único artefato para servir.
    MODEL_PATH: str = Field(default="models/classifier.joblib")
    ONNX_MODEL_PATH: str = Field(default="models/classifier.onnx")
    ONNX_LABELS_PATH: str = Field(default="models/classifier_labels.json")

    # Treino
    RANDOM_SEED: int = Field(default=42)
    TEST_SIZE: float = Field(default=0.2, gt=0.0, lt=1.0)
    DATASET_PATH: str = Field(default="data/raw/laudos.csv")
    TEXT_COLUMN: str = Field(default="texto")
    LABEL_COLUMN: str = Field(default="especialidade")
    TFIDF_MAX_FEATURES: int = Field(default=5000, gt=0)
    RF_N_ESTIMATORS: int = Field(default=200, gt=0)
    RF_MAX_DEPTH: int | None = Field(default=None)

    @field_validator("APP_ENV")
    @classmethod
    def _validate_app_env(cls, v: str) -> str:
        allowed = {"development", "staging", "production"}
        if v not in allowed:
            raise ValueError(f"APP_ENV must be one of {allowed}")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {allowed}")
        return v.upper()

    @field_validator("LOG_FORMAT")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        allowed = {"json", "text"}
        if v.lower() not in allowed:
            raise ValueError(f"LOG_FORMAT must be one of {allowed}")
        return v.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    from src.logging_config import setup_logging

    settings = Settings()
    setup_logging(settings)
    return settings
