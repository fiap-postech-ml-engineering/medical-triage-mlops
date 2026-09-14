"""EDA para o classificador binário CLINICO_GERAL vs ESPECIALISTA.

Lê data/raw/laudos.csv (gerado por `dataset.prepare_dataset`) e produz um
diagnóstico do dataset antes de qualquer modelagem: distribuição de classes,
nulos, duplicidade/vazamento de rótulo, tamanho das observações e padrões
linguísticos por classe.
"""

from collections import Counter
import logging
import re

import numpy as np
import pandas as pd

from src.binary_triage.dataset import prepare_dataset

logger = logging.getLogger(__name__)

pd.set_option("display.max_colwidth", 120)

STOPWORDS = set(
    "the a an of and to in with was were is are for on by as at be this that "
    "which from or not have has had it its patients patient study we results "
    "than these there also may can been between such among into two after "
    "using used all no both during but our their most however each other one "
    "cases case group groups significantly significant compared shown showed found".split()
)


def top_terms(texts: pd.Series, n: int = 25) -> list[tuple[str, int]]:
    words: list[str] = []
    for text in texts:
        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        words.extend(token for token in tokens if token not in STOPWORDS)
    return Counter(words).most_common(n)


def run(df: pd.DataFrame) -> None:
    logger.info("=" * 80)
    logger.info("1. ESTRUTURA DO DATASET")
    logger.info("=" * 80)
    logger.info("Shape: %s", df.shape)
    logger.info("Colunas: %s", list(df.columns))

    logger.info("=" * 80)
    logger.info("2. TARGET E DISTRIBUIÇÃO DE CLASSES (multiclasse original)")
    logger.info("=" * 80)
    vc = df["especialidade"].value_counts()
    logger.info("%s", vc)

    df = df.copy()
    df["target"] = np.where(
        df["especialidade"] == "clinica_geral", "CLINICO_GERAL", "ESPECIALISTA"
    )
    vc_bin = df["target"].value_counts()
    logger.info("Target binário derivado: %s", vc_bin.to_dict())
    logger.info("Razão de desbalanceamento: %.2fx", vc_bin.max() / vc_bin.min())

    logger.info("=" * 80)
    logger.info("3. NULOS E VAZIOS")
    logger.info("=" * 80)
    logger.info("Nulos por coluna: %s", df.isnull().sum().to_dict())
    logger.info("Textos vazios/whitespace: %d", (df["texto"].str.strip() == "").sum())

    logger.info("=" * 80)
    logger.info("4. DUPLICIDADE E RISCO DE VAZAMENTO")
    logger.info("=" * 80)
    full_dupes = df.duplicated().sum()
    text_dupes = df.duplicated(subset=["texto"]).sum()
    logger.info("Linhas 100%% duplicadas (texto+especialidade): %d", full_dupes)
    logger.info("Textos duplicados (ignorando label): %d linhas envolvidas", text_dupes)

    dupe_mask = df.duplicated(subset=["texto"], keep=False)
    dupes = df[dupe_mask]
    g_bin = dupes.groupby("texto")["target"].nunique()
    logger.info("Grupos de texto duplicado com target binário conflitante: %d", (g_bin > 1).sum())

    logger.info("=" * 80)
    logger.info("5. TAMANHO DAS OBSERVAÇÕES (texto)")
    logger.info("=" * 80)
    df["n_chars"] = df["texto"].str.len()
    df["n_words"] = df["texto"].str.split().str.len()
    logger.info("%s", df.groupby("target")[["n_chars", "n_words"]].describe().T)

    logger.info("=" * 80)
    logger.info("6. PADRÕES LINGUÍSTICOS POR CLASSE")
    logger.info("=" * 80)
    for cls in ["CLINICO_GERAL", "ESPECIALISTA"]:
        terms = top_terms(df.loc[df["target"] == cls, "texto"])
        logger.info("Top termos %s: %s", cls, ", ".join(f"{w}({c})" for w, c in terms))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    path = prepare_dataset()
    df = pd.read_csv(path)
    run(df)


if __name__ == "__main__":
    main()
