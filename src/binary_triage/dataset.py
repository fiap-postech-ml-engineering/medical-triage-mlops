"""Prepara data/raw/laudos.csv a partir do Medical Abstracts TC Corpus (Kaggle).

Baixa o dataset via kagglehub (não exige credenciais para este dataset
específico — validado manualmente), mapeia os rótulos originais do corpus
para as especialidades usadas no projeto, e junta treino+teste num único
CSV. O split treino/validação/teste do pipeline de triagem é feito depois,
em `train.py`, então os splits originais do Kaggle não precisam ser
preservados aqui.
"""

import logging
from pathlib import Path

import kagglehub
import pandas as pd

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "saharalaa/medical-abstracts-tc-corpus"

LABEL_TO_ESPECIALIDADE = {
    1: "oncologia",  # neoplasms
    2: "gastroenterologia",  # digestive system diseases
    3: "neurologia",  # nervous system diseases
    4: "cardiologia",  # cardiovascular diseases
    5: "clinica_geral",  # general pathological conditions
}

RAW_PATH = Path("data/raw/laudos.csv")


def _to_triage_format(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "texto": df["medical_abstract"].str.strip(),
            "especialidade": df["condition_label"].map(LABEL_TO_ESPECIALIDADE),
        }
    )
    missing = out["especialidade"].isna().sum()
    if missing:
        raise ValueError(f"condition_label fora do mapeamento conhecido em {missing} linhas")
    return out


def prepare_dataset(output_path: Path = RAW_PATH, force: bool = False) -> Path:
    """Garante que `output_path` existe, baixando e mapeando o dataset se preciso."""
    if output_path.exists() and not force:
        return output_path

    dataset_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    df_train = pd.read_csv(dataset_dir / "medical_tc_train.csv")
    df_test = pd.read_csv(dataset_dir / "medical_tc_test.csv")

    full_df = pd.concat(
        [_to_triage_format(df_train), _to_triage_format(df_test)], ignore_index=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = prepare_dataset()
    logger.info("Dataset pronto em %s", path)
