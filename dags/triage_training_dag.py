"""DAG de retreino periódico do classificador de triagem.

Orquestra, em duas tasks sequenciais, o mesmo fluxo que pode ser rodado
manualmente via `uv run python -m src.binary_triage.dataset` e
`uv run python -m src.binary_triage.train`: garante o dataset em
`data/raw/laudos.csv` e, em seguida, retreina e sobrescreve
`models/modelo.joblib`.
"""

from airflow.sdk import dag, task
import pendulum


@dag(
    dag_id="triage_training_dag",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    tags=["triage", "training"],
)
def triage_training_dag():
    @task
    def load_data() -> None:
        from src.binary_triage.dataset import prepare_dataset

        prepare_dataset(force=False)

    @task
    def train_and_save() -> None:
        from src.binary_triage.train import main

        main()

    load_data() >> train_and_save()


triage_training_dag()
