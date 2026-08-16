"""Compara direct (tfidf_lr_balanced) vs two-stage (multiclasse->binário)
em paridade: ambos com threshold ajustado na validação, não só o padrão 0.5."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from binary_triage.train import (  # noqa: E402
    build_candidates,
    evaluate_from_proba,
    load_and_clean,
    make_tfidf_only,
    split_data,
    two_stage_proba,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402

RANDOM_SEED = 42

df = load_and_clean()
train_df, val_df, test_df = split_data(df)
X_train, y_train = train_df[["texto"]], train_df["target"]
X_val, y_val = val_df[["texto"]], val_df["target"]
X_test, y_test = test_df[["texto"]], test_df["target"]

direct = build_candidates()["tfidf_lr_balanced"]
direct.fit(X_train, y_train)
p_direct_val = direct.predict_proba(X_val)[:, list(direct.classes_).index("ESPECIALISTA")]

stage1 = make_tfidf_only(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
stage1.fit(X_train, train_df["especialidade"])
p_2stage_val = two_stage_proba(stage1, X_val)

rows = []
for t in np.arange(0.20, 0.81, 0.05):
    t = round(t, 2)
    m_direct, _ = evaluate_from_proba(p_direct_val, y_val, threshold=t)
    m_2stage, _ = evaluate_from_proba(p_2stage_val, y_val, threshold=t)
    rows.append({"threshold": t, "modelo": "direct_tfidf_lr_balanced", **m_direct})
    rows.append({"threshold": t, "modelo": "two_stage_multiclass", **m_2stage})

out = pd.DataFrame(rows)
print(out.pivot(index="threshold", columns="modelo", values=["recall_especialista", "precision_especialista", "f1_macro"]).round(4).to_string())

print("\n--- No ponto de recall_especialista ~0.93 (o threshold final escolhido no direct), qual a precisão de cada um? ---")
for modelo, p in [("direct", p_direct_val), ("two_stage", p_2stage_val)]:
    best_t, best_diff = None, 999
    for t in np.arange(0.05, 0.96, 0.01):
        m, _ = evaluate_from_proba(p, y_val, threshold=round(t, 2))
        diff = abs(m["recall_especialista"] - 0.9377)
        if diff < best_diff:
            best_diff, best_t, best_m = diff, round(t, 2), m
    print(f"{modelo}: threshold={best_t} -> {best_m}")
