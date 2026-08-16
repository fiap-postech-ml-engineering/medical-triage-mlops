"""Treina o classificador de especialidade médica (TF-IDF + RandomForest).

Lê `DATASET_PATH` (CSV com colunas `TEXT_COLUMN` e `LABEL_COLUMN`, ver
`src/config.py`), treina um `Pipeline(TfidfVectorizer, RandomForestClassifier)`
e salva o pipeline completo em `MODEL_PATH` via joblib — um único artefato,
sem precisar versionar vetorizador e classificador separadamente.

Uso: `make train` (ou `uv run python -m src.training.train`).
Consumido também pela DAG do Airflow (`dags/triage_training_dag.py`).
"""

import logging

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import BASE_DIR, get_settings

logger = logging.getLogger(__name__)


def load_dataset(path, text_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {text_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes em {path}: {missing}. "
            f"Esperado pelo menos '{text_column}' (texto do laudo) e "
            f"'{label_column}' (especialidade de destino)."
        )
    return df[[text_column, label_column]].dropna()


def build_pipeline(
    max_features: int, n_estimators: int, max_depth: int | None, seed: int
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)),
            ),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    settings = get_settings()

    dataset_path = BASE_DIR / settings.DATASET_PATH
    logger.info("Carregando dataset de %s", dataset_path)
    df = load_dataset(dataset_path, settings.TEXT_COLUMN, settings.LABEL_COLUMN)
    logger.info(
        "Dataset carregado: %d amostras, %d classes", len(df), df[settings.LABEL_COLUMN].nunique()
    )

    x_train, x_test, y_train, y_test = train_test_split(
        df[settings.TEXT_COLUMN],
        df[settings.LABEL_COLUMN],
        test_size=settings.TEST_SIZE,
        random_state=settings.RANDOM_SEED,
        stratify=df[settings.LABEL_COLUMN],
    )

    pipeline = build_pipeline(
        max_features=settings.TFIDF_MAX_FEATURES,
        n_estimators=settings.RF_N_ESTIMATORS,
        max_depth=settings.RF_MAX_DEPTH,
        seed=settings.RANDOM_SEED,
    )

    logger.info("Treinando pipeline (TF-IDF + RandomForest)...")
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    report = classification_report(y_test, y_pred)
    logger.info("Relatório de classificação (holdout):\n%s", report)

    model_path = BASE_DIR / settings.MODEL_PATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Pipeline salvo em %s", model_path)


if __name__ == "__main__":
    main()
