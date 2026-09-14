"""Treino do classificador binário CLINICO_GERAL vs ESPECIALISTA.

Pipeline completo: limpeza (remove conflitos de rótulo e duplicidade de
texto) -> split treino/val/teste estratificado -> comparação de múltiplos
candidatos (TF-IDF+LogisticRegression, TF-IDF+LinearSVC, LSA/embeddings,
texto+features estruturadas, arquitetura em duas etapas) -> tuning de
threshold na validação -> refit no treino+validação -> avaliação final,
única e não enviesada, no teste -> interpretabilidade -> salva modelo.joblib.

Uso: `uv run python -m src.binary_triage.train` (a partir da raiz do projeto).
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.binary_triage.dataset import prepare_dataset
from src.binary_triage.features import TextMetaFeatures
from src.binary_triage.model_wrapper import ThresholdedBinaryClassifier

logger = logging.getLogger(__name__)

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "models"
REPORTS_DIR = OUT_DIR / "reports"
MODEL_PATH = OUT_DIR / "modelo.joblib"

RANDOM_SEED = 42
POSITIVE = "ESPECIALISTA"
NEGATIVE = "CLINICO_GERAL"


# ---------------------------------------------------------------------------
# 1. Dados: carregar, limpar (conflitos de rótulo + duplicidade), splitar
# ---------------------------------------------------------------------------
def load_and_clean() -> pd.DataFrame:
    raw_path = prepare_dataset()
    df = pd.read_csv(raw_path)
    df["target"] = np.where(df["especialidade"] == "clinica_geral", NEGATIVE, POSITIVE)

    n0 = len(df)
    dupe_mask = df.duplicated(subset=["texto"], keep=False)
    conflict_texts = df[dupe_mask].groupby("texto")["target"].nunique()
    conflict_texts = conflict_texts[conflict_texts > 1].index
    df = df[~df["texto"].isin(conflict_texts)]
    n1 = len(df)
    df = df.drop_duplicates(subset=["texto"], keep="first")
    n2 = len(df)

    logger.info(
        "Limpeza: %d linhas originais -> %d após remover %d linhas com rótulo binário "
        "conflitante entre duplicatas do mesmo texto -> %d após deduplicar texto exato "
        "(%d linhas removidas por duplicidade pura).",
        n0,
        n1,
        n0 - n1,
        n2,
        n1 - n2,
    )
    return df.reset_index(drop=True)


def split_data(df: pd.DataFrame):
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=RANDOM_SEED, stratify=df["especialidade"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["especialidade"]
    )
    logger.info(
        "Split -> treino: %d (%.1f%%) | validação: %d (%.1f%%) | teste: %d (%.1f%%)",
        len(train_df),
        100 * len(train_df) / len(df),
        len(val_df),
        100 * len(val_df) / len(df),
        len(test_df),
        100 * len(test_df) / len(df),
    )
    for name, part in [("treino", train_df), ("validação", val_df), ("teste", test_df)]:
        dist = (part["target"].value_counts(normalize=True) * 100).round(1).to_dict()
        logger.info("  Distribuição target (%s): %s", name, dist)
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# 2. Candidatos de modelo
# ---------------------------------------------------------------------------
def make_tfidf_only(clf, max_features=20000):
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=max_features, ngram_range=(1, 2), min_df=2
                            ),
                            "texto",
                        )
                    ]
                ),
            ),
            ("clf", clf),
        ]
    )


def make_combined(clf, max_features=20000):
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(
                                max_features=max_features, ngram_range=(1, 2), min_df=2
                            ),
                            "texto",
                        ),
                        (
                            "meta",
                            Pipeline(
                                [("extract", TextMetaFeatures()), ("scale", StandardScaler())]
                            ),
                            "texto",
                        ),
                    ]
                ),
            ),
            ("clf", clf),
        ]
    )


def make_lsa(clf, n_components=100):
    # TruncatedSVD sobre uma matriz TF-IDF grande (30k features) é caro em
    # CPU sem GPU. Reduzimos o vocabulário e o número de componentes deste
    # candidato especificamente (comparado aos outros, que usam TF-IDF
    # esparso direto sem redução de dimensionalidade) para manter o tempo de
    # comparação de candidatos razoável — não afeta os demais candidatos.
    return Pipeline(
        [
            (
                "features",
                ColumnTransformer(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2),
                            "texto",
                        )
                    ]
                ),
            ),
            ("svd", TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)),
            ("clf", clf),
        ]
    )


def build_candidates():
    return {
        "tfidf_lr": make_tfidf_only(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        "tfidf_lr_balanced": make_tfidf_only(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        ),
        "tfidf_linearsvc_calibrated": make_tfidf_only(
            CalibratedClassifierCV(LinearSVC(random_state=RANDOM_SEED), method="sigmoid", cv=3)
        ),
        "lsa_embeddings_lr": make_lsa(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)),
        "tfidf_meta_combined_lr": make_combined(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
        ),
    }


# ---------------------------------------------------------------------------
# 3. Avaliação
# ---------------------------------------------------------------------------
def evaluate(pipeline, x, y_true, threshold=0.5):
    proba = pipeline.predict_proba(x)
    classes = list(pipeline.classes_)
    pos_idx = classes.index(POSITIVE)
    p_pos = proba[:, pos_idx]
    y_pred = np.where(p_pos >= threshold, POSITIVE, NEGATIVE)
    return evaluate_from_proba(p_pos, y_true, threshold)[0], y_pred, p_pos


def two_stage_proba(stage1_pipeline, x):
    """p(ESPECIALISTA) = 1 - p(clinica_geral) a partir do classificador multiclasse."""
    proba = stage1_pipeline.predict_proba(x)
    classes = list(stage1_pipeline.classes_)
    cg_idx = classes.index("clinica_geral")
    return 1.0 - proba[:, cg_idx]


def evaluate_from_proba(p_pos, y_true, threshold=0.5):
    y_pred = np.where(p_pos >= threshold, POSITIVE, NEGATIVE)
    y_true_bin = (y_true == POSITIVE).astype(int)
    return {
        "precision_especialista": precision_score(
            y_true, y_pred, pos_label=POSITIVE, zero_division=0
        ),
        "recall_especialista": recall_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "f1_especialista": f1_score(y_true, y_pred, pos_label=POSITIVE, zero_division=0),
        "precision_clinico": precision_score(y_true, y_pred, pos_label=NEGATIVE, zero_division=0),
        "recall_clinico": recall_score(y_true, y_pred, pos_label=NEGATIVE, zero_division=0),
        "f1_clinico": f1_score(y_true, y_pred, pos_label=NEGATIVE, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y_true_bin, p_pos),
        "pr_auc": average_precision_score(y_true_bin, p_pos),
    }, y_pred


def interpret_model(pipeline: Pipeline, model_name: str):
    lines = [f"# Interpretabilidade — {model_name}", ""]
    clf = pipeline.named_steps.get("clf")
    features_step = pipeline.named_steps.get("features")

    if hasattr(clf, "calibrated_classifiers_"):
        lines.append("Modelo calibrado (LinearSVC + sigmoid) — coeficientes do estimador base.")
        base = clf.calibrated_classifiers_[0].estimator
        coefs = base.coef_[0]
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
    else:
        lines.append("Modelo não-linear ou pós-SVD: coeficientes não interpretáveis por termo.")
        return lines

    names = None
    if features_step is not None and hasattr(features_step, "get_feature_names_out"):
        try:
            names = features_step.get_feature_names_out()
        except (AttributeError, ValueError):
            names = None
    if names is None:
        names = [f"f{i}" for i in range(len(coefs))]

    if len(names) != len(coefs):
        lines.append("(dimensão pós-SVD — sem nomes de termo originais)")
        return lines

    order = np.argsort(coefs)
    top_pos = order[::-1][:30]
    top_neg = order[:30]

    lines.append(f"\n## Top termos associados a {POSITIVE} (coeficiente positivo)\n")
    for i in top_pos:
        lines.append(f"- `{names[i]}`: {coefs[i]:+.4f}")
    lines.append(f"\n## Top termos associados a {NEGATIVE} (coeficiente negativo)\n")
    for i in top_neg:
        lines.append(f"- `{names[i]}`: {coefs[i]:+.4f}")
    return lines


def build_examples_and_error_analysis(test_df, y_pred, p_pos, threshold):
    df = test_df.copy().reset_index(drop=True)
    df["y_pred"] = y_pred
    df["p_especialista"] = p_pos
    df["acerto"] = df["target"] == df["y_pred"]

    lines = [f"# Exemplos de previsão e análise de erros (threshold={threshold:.2f})\n"]

    lines.append("## Exemplos corretos\n")
    n_ok = min(3, int(df["acerto"].sum()))
    for _, row in df[df["acerto"]].sample(n_ok, random_state=RANDOM_SEED).iterrows():
        lines.append(
            f"- **Real**: {row['target']} (especialidade original: {row['especialidade']}) | "
            f"**Previsto**: {row['y_pred']} | p(ESPECIALISTA)={row['p_especialista']:.3f}\n"
            f"  > {row['texto'][:220]}...\n"
        )

    fn = df[(df["target"] == POSITIVE) & (df["y_pred"] == NEGATIVE)]
    fp = df[(df["target"] == NEGATIVE) & (df["y_pred"] == POSITIVE)]

    lines.append(
        f"\n## Falsos negativos (real=ESPECIALISTA, previsto=CLINICO_GERAL) — "
        f"{len(fn)} de {len(df)} no teste\n"
    )
    lines.append(
        "Impacto clínico: paciente que precisava de especialista é encaminhado ao "
        "clínico geral — risco de atraso diagnóstico/terapêutico. É o erro mais "
        "custoso neste contexto.\n"
    )
    for _, row in fn.sample(min(5, len(fn)), random_state=RANDOM_SEED).iterrows():
        lines.append(
            f"- especialidade original: {row['especialidade']} | "
            f"p(ESPECIALISTA)={row['p_especialista']:.3f}\n"
            f"  > {row['texto'][:220]}...\n"
        )

    lines.append(
        f"\n## Falsos positivos (real=CLINICO_GERAL, previsto=ESPECIALISTA) — "
        f"{len(fp)} de {len(df)} no teste\n"
    )
    lines.append(
        "Impacto: paciente sem necessidade clara de especialista é encaminhado a um — "
        "custo de recurso/tempo de especialista, mas sem risco direto ao paciente.\n"
    )
    for _, row in fp.sample(min(5, len(fp)), random_state=RANDOM_SEED).iterrows():
        lines.append(
            f"- p(ESPECIALISTA)={row['p_especialista']:.3f}\n  > {row['texto'][:220]}...\n"
        )

    return "\n".join(lines)


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_clean()
    train_df, val_df, test_df = split_data(df)

    x_train, y_train = train_df[["texto"]], train_df["target"]
    x_val, y_val = val_df[["texto"]], val_df["target"]
    x_test, y_test = test_df[["texto"]], test_df["target"]

    logger.info("=" * 80)
    logger.info("COMPARAÇÃO DE CANDIDATOS (treino -> avaliação em validação, threshold=0.5)")
    logger.info("=" * 80)

    candidates = build_candidates()
    results = {}
    fitted = {}
    for name, pipe in candidates.items():
        logger.info("Treinando candidato: %s", name)
        pipe.fit(x_train, y_train)
        metrics, _, _ = evaluate(pipe, x_val, y_val)
        results[name] = metrics
        fitted[name] = pipe
        logger.info("  %s -> %s", name, {k: round(v, 4) for k, v in metrics.items()})

    logger.info("Treinando candidato: two_stage_multiclass")
    stage1 = make_tfidf_only(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
    stage1.fit(x_train, train_df["especialidade"])
    p_pos_val_2stage = two_stage_proba(stage1, x_val)
    metrics_2stage, _ = evaluate_from_proba(p_pos_val_2stage, y_val)
    results["two_stage_multiclass"] = metrics_2stage
    logger.info(
        "  two_stage_multiclass -> %s", {k: round(v, 4) for k, v in metrics_2stage.items()}
    )

    comparison_df = pd.DataFrame(results).T.sort_values("recall_especialista", ascending=False)
    comparison_df.to_csv(REPORTS_DIR / "model_comparison_validation.csv")
    logger.info("\n%s", comparison_df.round(4).to_string())

    direct_df = comparison_df.drop(index="two_stage_multiclass")
    best_name = direct_df["f1_macro"].idxmax()
    best_pipeline = fitted[best_name]
    logger.info(">>> Melhor candidato direto por F1-macro em validação: %s", best_name)

    logger.info("=" * 80)
    logger.info("TUNING DE THRESHOLD (modelo: %s, conjunto: validação)", best_name)
    logger.info("=" * 80)
    proba_val = best_pipeline.predict_proba(x_val)
    classes = list(best_pipeline.classes_)
    p_pos_val = proba_val[:, classes.index(POSITIVE)]

    threshold_rows = []
    for t in np.arange(0.30, 0.71, 0.05):
        m, _ = evaluate_from_proba(p_pos_val, y_val, threshold=round(t, 2))
        m["threshold"] = round(t, 2)
        threshold_rows.append(m)
    threshold_df = pd.DataFrame(threshold_rows).set_index("threshold")
    threshold_df.to_csv(REPORTS_DIR / "threshold_sweep_validation.csv")
    logger.info("\n%s", threshold_df.round(4).to_string())

    # Critério: manter recall_especialista alto (evitar falso negativo = não
    # encaminhar quem precisa de especialista) sem deixar a precisão desabar.
    # Entre os thresholds com recall_especialista >= 0.90, escolhe o que
    # maximiza f1_macro; se nenhum atingir 0.90, usa o de maior recall.
    safe = threshold_df[threshold_df["recall_especialista"] >= 0.90]
    final_threshold = (
        safe["f1_macro"].idxmax()
        if len(safe) > 0
        else threshold_df["recall_especialista"].idxmax()
    )
    logger.info(">>> Threshold final escolhido: %.2f", final_threshold)

    logger.info("=" * 80)
    logger.info("REFIT EM TREINO+VALIDAÇÃO E AVALIAÇÃO FINAL NO TESTE")
    logger.info("=" * 80)
    x_trainval = pd.concat([x_train, x_val])
    y_trainval = pd.concat([y_train, y_val])

    final_pipeline = build_candidates()[best_name]
    final_pipeline.fit(x_trainval, y_trainval)

    test_metrics, y_pred_test, p_pos_test = evaluate(
        final_pipeline, x_test, y_test, threshold=final_threshold
    )
    logger.info(
        "Métricas finais no TESTE (threshold=%.2f): %s",
        final_threshold,
        {k: round(v, 4) for k, v in test_metrics.items()},
    )
    logger.info("\n%s", classification_report(y_test, y_pred_test, digits=4))

    with open(REPORTS_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {"model": best_name, "threshold": float(final_threshold), "metrics": test_metrics},
            f,
            indent=2,
            ensure_ascii=False,
        )

    test_metrics_05, _, _ = evaluate(final_pipeline, x_test, y_test, threshold=0.5)
    logger.info(
        "Para comparação — métricas no TESTE com threshold padrão 0.5: %s",
        {k: round(v, 4) for k, v in test_metrics_05.items()},
    )

    cm = confusion_matrix(y_test, y_pred_test, labels=[NEGATIVE, POSITIVE])
    disp = ConfusionMatrixDisplay(cm, display_labels=[NEGATIVE, POSITIVE])
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusão — teste (threshold={final_threshold:.2f})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    precisions, recalls, _ = precision_recall_curve((y_test == POSITIVE).astype(int), p_pos_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recalls, precisions)
    ax.set_xlabel("Recall (ESPECIALISTA)")
    ax.set_ylabel("Precision (ESPECIALISTA)")
    ax.set_title(f"Curva Precision-Recall — teste (PR-AUC={test_metrics['pr_auc']:.3f})")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "pr_curve_test.png", dpi=150)
    plt.close(fig)

    logger.info("=" * 80)
    logger.info("INTERPRETABILIDADE — TOP TERMOS DO MODELO FINAL (%s)", best_name)
    logger.info("=" * 80)
    interpret_lines = interpret_model(final_pipeline, best_name)
    with open(REPORTS_DIR / "interpretability.md", "w", encoding="utf-8") as f:
        f.write("\n".join(interpret_lines))

    examples_md = build_examples_and_error_analysis(
        test_df, y_pred_test, p_pos_test, final_threshold
    )
    with open(REPORTS_DIR / "examples_and_errors.md", "w", encoding="utf-8") as f:
        f.write(examples_md)

    # A decisão binária só diz "precisa de especialista" — não diz qual. Isso
    # é pouco acionável numa triagem real, então treinamos um segundo
    # classificador, só nas linhas ESPECIALISTA, para responder "qual
    # especialidade". Mesmo split, mas o sub-modelo nunca vê CLINICO_GERAL.
    logger.info("=" * 80)
    logger.info("SUB-CLASSIFICAÇÃO POR ESPECIALIDADE (apenas linhas ESPECIALISTA)")
    logger.info("=" * 80)
    esp_train = train_df[train_df["target"] == POSITIVE]
    esp_val = val_df[val_df["target"] == POSITIVE]
    esp_test = test_df[test_df["target"] == POSITIVE]
    esp_trainval = pd.concat([esp_train, esp_val])

    specialty_pipeline = make_tfidf_only(
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
    )
    specialty_pipeline.fit(esp_trainval[["texto"]], esp_trainval["especialidade"])

    y_pred_specialty = specialty_pipeline.predict(esp_test[["texto"]])
    specialty_report = classification_report(esp_test["especialidade"], y_pred_specialty, digits=4)
    logger.info("Relatório de classificação por especialidade (teste):\n%s", specialty_report)
    with open(REPORTS_DIR / "specialty_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(specialty_report)

    cm_specialty = confusion_matrix(
        esp_test["especialidade"], y_pred_specialty, labels=list(specialty_pipeline.classes_)
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm_specialty, display_labels=specialty_pipeline.classes_).plot(
        ax=ax, cmap="Oranges", colorbar=False, xticks_rotation=45
    )
    ax.set_title("Matriz de confusão — sub-classificação por especialidade (teste)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix_specialty_test.png", dpi=150)
    plt.close(fig)

    wrapped = ThresholdedBinaryClassifier(
        pipeline=final_pipeline,
        threshold=float(final_threshold),
        specialty_pipeline=specialty_pipeline,
    )
    joblib.dump(wrapped, MODEL_PATH)
    logger.info(">>> Modelo final salvo em %s", MODEL_PATH)

    summary = {
        "modelo_escolhido": best_name,
        "threshold_final": float(final_threshold),
        "n_treino": len(train_df),
        "n_validacao": len(val_df),
        "n_teste": len(test_df),
        "distribuicao_target_completa": df["target"].value_counts().to_dict(),
        "metricas_teste": test_metrics,
        "metricas_teste_threshold_0.5": test_metrics_05,
        "comparacao_validacao": comparison_df.round(4).to_dict(orient="index"),
        "specialty_classification_report_teste": specialty_report,
    }
    with open(REPORTS_DIR / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Resumo completo salvo em %s", REPORTS_DIR / "run_summary.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
