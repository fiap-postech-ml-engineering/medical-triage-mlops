"""DAG de retreino do classificador de especialidade médica.

Pipeline simples exigido pela Etapa 2 do Tech Challenge: ingestão -> treino ->
salvamento do modelo. Reaproveita `src.training.train` (mesmo código usado por
`make train`), então a lógica de treino tem um único ponto de manutenção.

Para rodar localmente:
  1. `export AIRFLOW_HOME=$(pwd)/.airflow`
  2. `uv sync --extra airflow`
  3. `uv run airflow standalone` (sobe webserver + scheduler + cria usuário admin)
  4. Aponte `dags_folder` (airflow.cfg) para esta pasta `dags/`, ou copie/symlink
     este arquivo para dentro do `dags_folder` padrão.
"""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task


@dag(
    dag_id="triage_training_pipeline",
    description="Ingestão de dados -> treino -> salvamento do classificador de especialidade.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "triage", "tech-challenge-fase-3"],
)
def triage_training_pipeline():
    @task
    def load_data() -> str:
        from src.config import BASE_DIR, get_settings
        from src.training.train import load_dataset

        settings = get_settings()
        dataset_path = BASE_DIR / settings.DATASET_PATH
        df = load_dataset(dataset_path, settings.TEXT_COLUMN, settings.LABEL_COLUMN)
        return f"{len(df)} amostras carregadas de {dataset_path}"

    @task
    def train_and_save(_: str) -> str:
        from src.training.train import main as train_main

        train_main()
        return "Modelo treinado e salvo em models/classifier.joblib"

    train_and_save(load_data())


triage_training_pipeline()
