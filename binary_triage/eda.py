"""EDA para o classificador binário CLINICO_GERAL vs ESPECIALISTA.

Lê data/raw/laudos.csv (fonte já usada pelo projeto) e produz um diagnóstico
completo do dataset ANTES de qualquer modelagem: colunas, target, distribuição
de classes, nulos, duplicidade, vazamento potencial, tamanho das observações
e padrões linguísticos por classe.
"""

import re
from collections import Counter

import numpy as np
import pandas as pd

pd.set_option("display.max_colwidth", 120)

RAW_PATH = "data/raw/laudos.csv"


def main():
    df = pd.read_csv(RAW_PATH)

    print("=" * 80)
    print("1. ESTRUTURA DO DATASET")
    print("=" * 80)
    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(df.dtypes)
    print()
    print("Únicas colunas candidatas a target: 'especialidade' (categórica, 5 valores).")
    print("'texto' é a única feature disponível (texto livre). NÃO há colunas")
    print("estruturadas adicionais (idade, sexo, duração, etc.) neste dataset —")
    print("apenas texto + rótulo de especialidade.")
    print()

    print("=" * 80)
    print("2. TARGET E DISTRIBUIÇÃO DE CLASSES (multiclasse original)")
    print("=" * 80)
    vc = df["especialidade"].value_counts()
    print(vc)
    print((vc / len(df) * 100).round(2).astype(str) + "%")
    print()

    df["target"] = np.where(df["especialidade"] == "clinica_geral", "CLINICO_GERAL", "ESPECIALISTA")
    print("Target binário derivado (clinica_geral -> CLINICO_GERAL; demais -> ESPECIALISTA):")
    vc_bin = df["target"].value_counts()
    print(vc_bin)
    print((vc_bin / len(df) * 100).round(2).astype(str) + "%")
    print(f"Razão de desbalanceamento: {vc_bin.max() / vc_bin.min():.2f}x (desbalanceamento leve/moderado)")
    print()

    print("=" * 80)
    print("3. NULOS E VAZIOS")
    print("=" * 80)
    print(df.isnull().sum())
    print(f"Textos vazios/whitespace: {(df['texto'].str.strip() == '').sum()}")
    print()

    print("=" * 80)
    print("4. DUPLICIDADE E RISCO DE VAZAMENTO")
    print("=" * 80)
    full_dupes = df.duplicated().sum()
    text_dupes = df.duplicated(subset=["texto"]).sum()
    print(f"Linhas 100% duplicadas (texto+especialidade): {full_dupes}")
    print(f"Textos duplicados (mesmo texto, ignorando label): {text_dupes} linhas envolvidas")

    dupe_mask = df.duplicated(subset=["texto"], keep=False)
    dupes = df[dupe_mask]
    g_multi = dupes.groupby("texto")["especialidade"].nunique()
    g_bin = dupes.groupby("texto")["target"].nunique()
    print(f"Grupos de texto duplicado com especialidade DIFERENTE entre duplicatas: {(g_multi > 1).sum()}")
    print(f"Grupos de texto duplicado com TARGET BINÁRIO conflitante (mesmo texto -> classes binárias diferentes): {(g_bin > 1).sum()}")
    print()
    print(">>> Isso é sinal de rótulo multi-especialidade forçado em single-label")
    print(">>> (o corpus original mapeia 'condition_name', um artigo pode tratar de")
    print(">>> mais de uma condição). Textos idênticos com target BINÁRIO conflitante")
    print(">>> representam ruído de rótulo irresolvível e serão removidos antes do split.")
    print(">>> Textos idênticos com target binário CONSISTENTE serão agrupados no mesmo")
    print(">>> split (GroupShuffleSplit) para não vazar o mesmo texto entre treino/teste.")
    print()

    print("=" * 80)
    print("5. TAMANHO DAS OBSERVAÇÕES (texto)")
    print("=" * 80)
    df["n_chars"] = df["texto"].str.len()
    df["n_words"] = df["texto"].str.split().str.len()
    print(df.groupby("target")[["n_chars", "n_words"]].describe().T)
    print()

    print("=" * 80)
    print("6. PADRÕES LINGUÍSTICOS POR CLASSE (top termos, stopwords em inglês removidas)")
    print("=" * 80)
    stop = set(
        "the a an of and to in with was were is are for on by as at be this that "
        "which from or not have has had it its patients patient study we results "
        "than these there also may can been between such among into two after "
        "using used all no both during but our their most however each other one "
        "cases case group groups significantly significant compared shown showed found"
        .split()
    )

    def top_terms(texts, n=25):
        words = []
        for t in texts:
            toks = re.findall(r"[a-zA-Z]{3,}", t.lower())
            words.extend(w for w in toks if w not in stop)
        return Counter(words).most_common(n)

    for cls in ["CLINICO_GERAL", "ESPECIALISTA"]:
        print(f"\n--- Top termos: {cls} ---")
        terms = top_terms(df.loc[df["target"] == cls, "texto"])
        print(", ".join(f"{w}({c})" for w, c in terms))

    print()
    print("--- Top termos por especialidade original (dentro de ESPECIALISTA) ---")
    for esp in ["oncologia", "cardiologia", "neurologia", "gastroenterologia"]:
        terms = top_terms(df.loc[df["especialidade"] == esp, "texto"], n=15)
        print(f"{esp}: " + ", ".join(f"{w}({c})" for w, c in terms))

    print()
    print("=" * 80)
    print("7. RESUMO")
    print("=" * 80)
    print(f"N total: {len(df)} | features: texto (livre) + especialidade (target multiclasse)")
    print(f"Target binário definido: clinica_geral -> CLINICO_GERAL, demais 4 -> ESPECIALISTA")
    print(f"Desbalanceamento: {vc_bin.to_dict()}")


if __name__ == "__main__":
    main()
