"""Compara direct (tfidf_lr_balanced) vs two-stage (multiclasse->binário)
em paridade: ambos com threshold ajustado na validação, não só o padrão 0.5.

Script de análise auxiliar (não faz parte do pipeline de produção) — serve
para justificar por que train.py escolhe o candidato direto em vez da
arquitetura em duas etapas.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.binary_triage.train import (
    RANDOM_SEED,
    build_candidates,
    evaluate_from_proba,
    load_and_clean,
    make_tfidf_only,
    split_data,
    two_stage_proba,
)

logger = logging.getLogger(__name__)


def main() -> None:
    df = load_and_clean()
    train_df, val_df, _ = split_data(df)
    x_train, y_train = train_df[["texto"]], train_df["target"]
    x_val, y_val = val_df[["texto"]], val_df["target"]

    direct = build_candidates()["tfidf_lr_balanced"]
    direct.fit(x_train, y_train)
    p_direct_val = direct.predict_proba(x_val)[:, list(direct.classes_).index("ESPECIALISTA")]

    stage1 = make_tfidf_only(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
    stage1.fit(x_train, train_df["especialidade"])
    p_2stage_val = two_stage_proba(stage1, x_val)

    rows = []
    for t in np.arange(0.20, 0.81, 0.05):
        t = round(t, 2)
        m_direct, _ = evaluate_from_proba(p_direct_val, y_val, threshold=t)
        m_2stage, _ = evaluate_from_proba(p_2stage_val, y_val, threshold=t)
        rows.append({"threshold": t, "modelo": "direct_tfidf_lr_balanced", **m_direct})
        rows.append({"threshold": t, "modelo": "two_stage_multiclass", **m_2stage})

    out = pd.DataFrame(rows)
    pivot = out.pivot(
        index="threshold",
        columns="modelo",
        values=["recall_especialista", "precision_especialista", "f1_macro"],
    ).round(4)
    logger.info("\n%s", pivot.to_string())

    logger.info(
        "--- No ponto de recall_especialista ~0.93 (threshold final escolhido no direct), "
        "qual a precisão de cada um? ---"
    )
    for modelo, p in [("direct", p_direct_val), ("two_stage", p_2stage_val)]:
        best_t, best_diff, best_m = None, 999.0, None
        for t in np.arange(0.05, 0.96, 0.01):
            m, _ = evaluate_from_proba(p, y_val, threshold=round(t, 2))
            diff = abs(m["recall_especialista"] - 0.9377)
            if diff < best_diff:
                best_diff, best_t, best_m = diff, round(t, 2), m
        logger.info("%s: threshold=%s -> %s", modelo, best_t, best_m)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
