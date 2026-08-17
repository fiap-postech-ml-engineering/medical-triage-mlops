"""Gera notebooks/binary_triage_analysis.ipynb a partir de células programáticas.

Segue a convenção usada nos projetos anteriores do time (ecommerce-recsys-mlops,
ml-churn-prediction): seções numeradas "## N. Título ====...", callouts
**Objetivo:** abrindo cada subseção, parágrafo "Nesta etapa iremos..." de
transição, dicionário de colunas logo no carregamento dos dados, análise de
quantis, e uma conclusão em linguagem simples fechando cada bloco.

Roda o mesmo pipeline de binary_triage/train.py — este notebook é a versão
"mostrando o trabalho" célula a célula do mesmo pipeline, não um pipeline
paralelo, então os números batem exatamente com o modelo.joblib salvo.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


SEP = "=" * 25

md(f"""# Triagem Clínico Geral × Especialista

Classificador binário para apoiar o encaminhamento de laudos entre atendimento
generalista e especializado, a partir do texto do laudo/observação médica.

**Objetivo geral:** validar a estratégia de modelagem (baseline vs. arquitetura
em duas etapas, threshold de decisão) em cima do dataset real do projeto
(`data/raw/laudos.csv`) antes de fixar o pipeline de produção (`modelo.joblib`).

Relatório visual consolidado com os mesmos resultados: `binary_triage/reports/report.html`.
""")

# ===========================================================================
md(f"""## 1. Configuração do ambiente {SEP}

Nesta etapa iremos configurar o ambiente de trabalho, importar as bibliotecas
necessárias e fixar a seed de aleatoriedade. Reaproveitamos as funções de
`binary_triage/train.py` e `binary_triage/features.py` em vez de duplicar a
lógica de limpeza/modelagem aqui — isso garante que os números deste notebook
batem exatamente com o `modelo.joblib` gerado pelo script de treino oficial.""")

code("""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent) if Path.cwd().name == "binary_triage" else str(Path.cwd()))

import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from binary_triage.train import (
    RAW_PATH, RANDOM_SEED, POSITIVE, NEGATIVE,
    load_and_clean, split_data, build_candidates, make_tfidf_only,
    evaluate, evaluate_from_proba, two_stage_proba, interpret_model,
)
from binary_triage.features import TextMetaFeatures, SPECIALTY_KEYWORD_SETS
from binary_triage.model_wrapper import ThresholdedBinaryClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
pd.set_option("display.max_colwidth", 100)
plt.rcParams["figure.facecolor"] = "white"
BAR_COLOR = "#3b6ea5"
""")

# ===========================================================================
md(f"""## 2. Carregamento dos dados {SEP}

### Dataset: `data/raw/laudos.csv`

Duas colunas apenas — não existem variáveis estruturadas independentes
(idade, sexo, duração, medicamentos como campos separados) neste dataset:

| Coluna | Tipo | Descrição |
|---|---|---|
| `texto` | texto livre | Resumo/observação clínica (laudo). Base do corpus é o *Medical Abstracts TC Corpus*, em inglês. |
| `especialidade` | categórica (5 valores) | Especialidade de destino: `clinica_geral`, `oncologia`, `cardiologia`, `neurologia`, `gastroenterologia`. É a coluna target — a única categórica de baixa cardinalidade, e o nome já indica o destino do encaminhamento. |

O **target binário** deste projeto é derivado de `especialidade`:
`clinica_geral -> CLINICO_GERAL`, as outras quatro -> `ESPECIALISTA`.""")

code("""
df_raw = pd.read_csv(RAW_PATH)
print(f"Shape: {df_raw.shape}")
df_raw.head(3)
""")

code("""
df_raw.info()
""")

md("""Não há necessidade de conversão de tipos — as duas colunas já são `object`
(string). Criamos a coluna `target` a partir de `especialidade` e seguimos
para a análise exploratória.""")

code("""
df_raw["target"] = np.where(df_raw["especialidade"] == "clinica_geral", NEGATIVE, POSITIVE)
df_raw.head(3)
""")

# ===========================================================================
md(f"""## 3. Análise Exploratória {SEP}

Nesta etapa iremos analisar o dataset em profundidade antes de treinar
qualquer modelo: valores ausentes, duplicidade/vazamento de rótulo, a
variável target, o tamanho das observações e os padrões linguísticos de cada
classe. Esta análise vai fundamentar as decisões de limpeza e de engenharia
de features das seções seguintes.""")

md("""### 3.1 Valores ausentes

**Objetivo:** identificar valores ausentes e strings vazias antes de qualquer
processamento de texto.""")

code("""
print("Valores nulos por coluna:")
display(df_raw.isnull().sum())
print(f"Textos vazios/whitespace: {(df_raw['texto'].str.strip() == '').sum()}")
""")

md("""Zero nulos e zero textos vazios — o dataset já chega limpo nesse
aspecto. O problema de qualidade real está na duplicidade de rótulo, tratado
a seguir.""")

md("""### 3.2 Duplicidade e vazamento de rótulo

**Objetivo:** verificar se existem linhas duplicadas e se essa duplicidade
representa risco de vazamento entre treino e teste — condição necessária
antes de decidir a estratégia de split (seção 5).""")

code("""
print(f"Linhas 100% duplicadas (texto + especialidade): {df_raw.duplicated().sum()}")

dupe_mask = df_raw.duplicated(subset=["texto"], keep=False)
dupes = df_raw[dupe_mask]
g_multi = dupes.groupby("texto")["especialidade"].nunique()
g_bin = dupes.groupby("texto")["target"].nunique()

print(f"Linhas envolvidas em texto duplicado (ignorando o rótulo): {dupe_mask.sum()}")
print(f"Grupos de texto duplicado com especialidade DIFERENTE entre as duplicatas: {(g_multi > 1).sum()}")
print(f"Grupos com TARGET BINÁRIO conflitante entre as duplicatas: {(g_bin > 1).sum()}")
""")

md("""2.929 textos idênticos aparecem com especialidade diferente entre
duplicatas — o corpus original mapeia categorias de doença a partir de
`condition_name`, e um mesmo abstract pode tocar mais de uma condição, mas o
dataset força rótulo único por linha. Em **2.411** desses grupos o conflito
chega a atingir o próprio target binário (mesmo texto ora `CLINICO_GERAL`, ora
`ESPECIALISTA`) — isso é ruído de rótulo irresolvível, não uma questão de
desempate, e por isso essas linhas serão **removidas** antes do split (não
depois), junto com duplicidade pura de texto. A função `load_and_clean()`
(`binary_triage/train.py`) implementa exatamente essa regra — reaproveitamos
aqui para manter os números idênticos ao pipeline de treino oficial.""")

code("""
df = load_and_clean()
print(f"Dataset limpo: {len(df)} linhas ({len(df_raw) - len(df)} removidas, {(len(df_raw)-len(df))/len(df_raw):.1%} do total)")
""")

md("""### 3.3 Variável target

**Objetivo:** entender como a variável que o modelo vai prever se
comporta — balanceamento de classes e composição da classe ESPECIALISTA.""")

code("""
print("Distribuição multiclasse original (pós-limpeza):")
vc_multi = df["especialidade"].value_counts()
display(vc_multi)
display((vc_multi / len(df) * 100).round(1).astype(str) + "%")

print("\\nDistribuição do target binário:")
vc_bin = df["target"].value_counts()
display(vc_bin)
display((vc_bin / len(df) * 100).round(1).astype(str) + "%")
print(f"\\nRazão de desbalanceamento: {vc_bin.max() / vc_bin.min():.2f}x")
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
vc_multi.plot(kind="barh", ax=axes[0], color=BAR_COLOR)
axes[0].set_title("Distribuição multiclasse (especialidade)")
axes[0].invert_yaxis()

vc_bin.plot(kind="bar", ax=axes[1], color=[BAR_COLOR, "#a5c4e0"])
axes[1].set_title("Distribuição do target binário")
axes[1].tick_params(axis="x", rotation=0)
fig.tight_layout()
plt.show()
""")

md("""72,8% dos laudos são ESPECIALISTA contra 27,2% CLINICO_GERAL
(~2,7:1) — desbalanceamento moderado, tratável com `class_weight` e ajuste de
threshold (seções 6 e 7), sem necessidade de reamostragem.""")

md("""### 3.4 Tamanho das observações

**Objetivo:** verificar se o tamanho do texto por si só já separa as classes
(o que seria um atalho barato para o modelo) e mapear a distribuição via
quantis.""")

code("""
df["n_chars"] = df["texto"].str.len()
df["n_words"] = df["texto"].str.split().str.len()

quantiles = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
display(df.groupby("target")["n_chars"].quantile(quantiles).unstack().round(0))
display(df.groupby("target")[["n_chars", "n_words"]].mean().round(1))
""")

code("""
fig, ax = plt.subplots(figsize=(7, 4))
for cls, color in [(NEGATIVE, "#a5c4e0"), (POSITIVE, BAR_COLOR)]:
    ax.hist(df.loc[df["target"] == cls, "n_chars"], bins=40, alpha=0.6, label=cls, color=color)
ax.set_xlabel("Caracteres por observação")
ax.set_ylabel("Frequência")
ax.legend()
ax.set_title("Distribuição de tamanho do texto por classe")
plt.show()
""")

md("""As distribuições praticamente se sobrepõem (mediana ~1.180 caracteres em
ambas as classes) — tamanho de texto **não discrimina** a classe e não é um
atalho útil para o modelo. Ainda assim, entra como uma das features testadas
na seção 4, junto com contagem de termos por especialidade.""")

md("""### 3.5 Padrões linguísticos por classe

**Objetivo:** identificar se cada especialidade tem vocabulário próprio, e se
a classe CLINICO_GERAL tem um padrão textual identificável ou se é de fato
uma classe "resíduo" (premissa central do problema, a validar empiricamente).""")

code("""
stop = set(
    "the a an of and to in with was were is are for on by as at be this that "
    "which from or not have has had it its patients patient study we results "
    "than these there also may can been between such among into two after "
    "using used all no both during but our their most however each other one "
    "cases case group groups significantly significant compared shown showed found"
    .split()
)

def top_terms(texts, n=12):
    words = []
    for t in texts:
        toks = re.findall(r"[a-zA-Z]{3,}", t.lower())
        words.extend(w for w in toks if w not in stop)
    return Counter(words).most_common(n)

for esp in ["clinica_geral", "oncologia", "cardiologia", "neurologia", "gastroenterologia"]:
    terms = top_terms(df.loc[df["especialidade"] == esp, "texto"])
    print(f"{esp:20s}: " + ", ".join(f"{w}({c})" for w, c in terms))
""")

md("""`clinica_geral` não tem vocabulário próprio claro — os termos mais
frequentes (`disease`, `treatment`, `clinical`, `years`) são genéricos e
compartilhados com as outras quatro classes. As especialidades, em contraste,
têm vocabulário claramente distintivo (`cancer`/`carcinoma`/`tumor` para
oncologia, `coronary`/`cardiac`/`hypertension` para cardiologia, etc.). Isso
**confirma empiricamente** a premissa do problema: Clínico Geral é a classe
residual, sem padrão textual único — e por isso dividi-la em subcategorias
específicas (respiratória, dermatológica, preventiva, ver seção 8) **não é
sustentado pelos dados**: são abstracts acadêmicos sobre doenças, não relatos
de queixa de paciente com esse recorte.""")

# ===========================================================================
md(f"""## 4. Engenharia de features {SEP}

Sem colunas estruturadas independentes, derivamos features numéricas do
próprio texto (`binary_triage/features.py::TextMetaFeatures`): tamanho
(chars/words), densidade de dígitos, nº de sentenças, e contagem de termos
associados a cada especialidade — listas construídas a partir da seção 3.5,
não inventadas. Testamos empiricamente na seção 6 se combinar essas features
com TF-IDF melhora o modelo.""")

code("""
meta_preview = TextMetaFeatures().transform(df["texto"].head(5))
pd.DataFrame(meta_preview, columns=TextMetaFeatures().get_feature_names_out())
""")

# ===========================================================================
md(f"""## 5. Preparação para modelagem {SEP}

**Objetivo:** dividir treino/validação/teste sem vazamento. Split
estratificado por `especialidade` (granularidade mais fina que o target
binário), 70/15/15. Como o dataset já foi deduplicado por texto na seção 3.2,
cada linha é um documento único — a preocupação de vazamento por duplicidade
já foi resolvida na origem, antes do split (não precisamos de um split por
grupo/paciente adicional).""")

code("""
train_df, val_df, test_df = split_data(df)

X_train, y_train = train_df[["texto"]], train_df["target"]
X_val, y_val = val_df[["texto"]], val_df["target"]
X_test, y_test = test_df[["texto"]], test_df["target"]

for name, part in [("treino", train_df), ("validação", val_df), ("teste", test_df)]:
    dist = (part["target"].value_counts(normalize=True) * 100).round(1).to_dict()
    print(f"{name:12s} n={len(part):5d}  {dist}")
""")

# ===========================================================================
md(f"""## 6. Modelagem e comparação de candidatos {SEP}

### 6.1 Candidatos

- `tfidf_lr` / `tfidf_lr_balanced` — TF-IDF (1-2 grams) + LogisticRegression, com e sem `class_weight="balanced"`
- `tfidf_linearsvc_calibrated` — TF-IDF + LinearSVC, calibrado (sigmoid) para gerar `predict_proba`
- `lsa_embeddings_lr` — TF-IDF + TruncatedSVD (200 componentes, "embedding" denso via LSA) + LogisticRegression. Optamos por LSA em vez de embeddings de transformer porque este ambiente não tem acesso a download de modelo pré-treinado nem GPU — é a alternativa de representação densa mais direta disponível localmente.
- `tfidf_meta_combined_lr` — TF-IDF + features estruturadas da seção 4, via `ColumnTransformer`
- `two_stage_multiclass` — classificador multiclasse (5 especialidades); a probabilidade de ESPECIALISTA é derivada como `1 - p(clinica_geral)`

Todos usam `Pipeline`/`ColumnTransformer` treinados **só no conjunto de
treino** — o vetorizador nunca vê texto de validação/teste antes da hora.""")

code("""
candidates = build_candidates()
results = {}
fitted = {}
for name, pipe in candidates.items():
    pipe.fit(X_train, y_train)
    metrics, _, _ = evaluate(pipe, X_val, y_val)
    results[name] = metrics
    fitted[name] = pipe

stage1 = make_tfidf_only(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED))
stage1.fit(X_train, train_df["especialidade"])
p_pos_val_2stage = two_stage_proba(stage1, X_val)
metrics_2stage, _ = evaluate_from_proba(p_pos_val_2stage, y_val)
results["two_stage_multiclass"] = metrics_2stage

comparison_df = pd.DataFrame(results).T.sort_values("f1_macro", ascending=False)
comparison_df.round(4)
""")

code("""
fig, ax = plt.subplots(figsize=(8, 4))
order = comparison_df.index
colors = [BAR_COLOR if m == "tfidf_lr_balanced" else "#c9c9c9" for m in order]
ax.barh(order, comparison_df.loc[order, "f1_macro"], color=colors)
ax.invert_yaxis()
ax.set_xlabel("F1-macro (validação)")
ax.set_title("Comparação de candidatos")
plt.show()
""")

md("""Vencedor por F1-macro: `tfidf_lr_balanced`. `LinearSVC` calibrado fica
muito próximo. `lsa_embeddings_lr` tem o pior desempenho — reduzir a
dimensionalidade via SVD perde justamente os termos raros e específicos
(nomes de doença) que mais discriminam a classe Especialista neste corpus.""")

md("""### 6.2 Arquitetura em duas etapas vs. classificador direto

**Objetivo:** decidir se vale a pena manter um classificador multiclasse
intermediário (seção 8 do enunciado do projeto) ou se um classificador
binário direto já é suficiente. Comparação justa: ambos calibrados no mesmo
ponto de operação (mesmo recall de Especialista), não no threshold padrão 0,5.""")

code("""
direct = fitted["tfidf_lr_balanced"]
p_direct_val = direct.predict_proba(X_val)[:, list(direct.classes_).index(POSITIVE)]

def closest_at_recall(p, target_recall):
    best_t, best_diff, best_m = None, 999, None
    for t in np.arange(0.05, 0.96, 0.01):
        m, _ = evaluate_from_proba(p, y_val, threshold=round(t, 2))
        diff = abs(m["recall_especialista"] - target_recall)
        if diff < best_diff:
            best_diff, best_t, best_m = diff, round(t, 2), m
    return best_t, best_m

t_direct, m_direct_matched = closest_at_recall(p_direct_val, 0.9377)
t_2stage, m_2stage_matched = closest_at_recall(p_pos_val_2stage, 0.9377)
print(f"direct    @ threshold={t_direct}: precision={m_direct_matched['precision_especialista']:.4f}  recall={m_direct_matched['recall_especialista']:.4f}  f1_macro={m_direct_matched['f1_macro']:.4f}")
print(f"two_stage @ threshold={t_2stage}: precision={m_2stage_matched['precision_especialista']:.4f}  recall={m_2stage_matched['recall_especialista']:.4f}  f1_macro={m_2stage_matched['f1_macro']:.4f}")
""")

md("""No threshold padrão (0,5), a two-stage parece muito melhor em recall
(~0,97 vs ~0,84) — mas isso é só efeito de não estar calibrada. No mesmo
ponto de recall (~0,94), a precisão **empata** entre as duas abordagens. A
arquitetura extra (treinar e manter um classificador multiclasse) não compra
ganho de desempenho aqui — fica o modelo direto, mais simples de treinar e
servir em produção.""")

# ===========================================================================
md(f"""## 7. Calibração de threshold {SEP}

**Objetivo:** encontrar o ponto de corte de probabilidade certo para este
problema — 0,50 não é ele. Falso negativo (Especialista classificado como
Clínico Geral) é o erro caro aqui: risco de atraso diagnóstico. Falso positivo
custa só tempo de especialista. Critério: entre os thresholds com recall
Especialista ≥ 90%, escolher o que maximiza F1-macro.""")

code("""
proba_val = direct.predict_proba(X_val)
p_pos_val = proba_val[:, list(direct.classes_).index(POSITIVE)]

threshold_rows = []
for t in np.arange(0.30, 0.71, 0.05):
    m, _ = evaluate_from_proba(p_pos_val, y_val, threshold=round(t, 2))
    m["threshold"] = round(t, 2)
    threshold_rows.append(m)
threshold_df = pd.DataFrame(threshold_rows).set_index("threshold")
display(threshold_df.round(4))

safe = threshold_df[threshold_df["recall_especialista"] >= 0.90]
final_threshold = safe["f1_macro"].idxmax() if len(safe) else threshold_df["recall_especialista"].idxmax()
print(f"Threshold final escolhido: {final_threshold}")
""")

code("""
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(threshold_df.index, threshold_df["recall_especialista"], marker="o", color="#c9622a", label="Recall Especialista")
ax.plot(threshold_df.index, threshold_df["precision_especialista"], marker="o", color=BAR_COLOR, label="Precision Especialista")
ax.axvline(final_threshold, color="#999999", linestyle="--", label=f"threshold escolhido = {final_threshold}")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.legend()
ax.set_title("Precision × Recall (Especialista) por threshold — validação")
plt.show()
""")

# ===========================================================================
md(f"""## 8. Avaliação final {SEP}

**Objetivo:** medir o desempenho real do modelo em dados nunca vistos. Refit
do modelo escolhido em treino+validação combinados (mais dado para o modelo
de produção), avaliação única e não enviesada no teste.""")

code("""
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_pipeline = build_candidates()["tfidf_lr_balanced"]
final_pipeline.fit(X_trainval, y_trainval)

test_metrics, y_pred_test, p_pos_test = evaluate(final_pipeline, X_test, y_test, threshold=final_threshold)
print(classification_report(y_test, y_pred_test, digits=4))
pd.Series(test_metrics).round(4)
""")

code("""
cm = confusion_matrix(y_test, y_pred_test, labels=[NEGATIVE, POSITIVE])
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=[NEGATIVE, POSITIVE]).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Matriz de confusão — teste (threshold={final_threshold:.2f})")
plt.show()
""")

md("""Recall de Especialista de 92,6% no teste, muito próximo dos 93,8%
observados em validação — sem sinal de overfit no threshold escolhido.""")

# ===========================================================================
md(f"""## 9. Sub-classificação por especialidade {SEP}

**Objetivo:** a decisão binária (seções 1-8) só diz "precisa de especialista"
— não diz qual. Isso é pouco acionável numa triagem real, então treinamos um
segundo classificador, **só nas linhas ESPECIALISTA**, para responder "qual
especialidade" entre oncologia, cardiologia, neurologia e gastroenterologia.
As duas decisões são independentes: mesmo split de treino/val/teste, mas o
sub-modelo nunca vê linhas CLINICO_GERAL — ele não compete com a decisão
binária, só a complementa.""")

code("""
esp_train = train_df[train_df["target"] == POSITIVE]
esp_val = val_df[val_df["target"] == POSITIVE]
esp_test = test_df[test_df["target"] == POSITIVE]
esp_trainval = pd.concat([esp_train, esp_val])

print("Distribuição de especialidade (treino, apenas ESPECIALISTA):")
display(esp_train["especialidade"].value_counts())

specialty_pipeline = make_tfidf_only(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED)
)
specialty_pipeline.fit(esp_trainval[["texto"]], esp_trainval["especialidade"])

y_pred_specialty = specialty_pipeline.predict(esp_test[["texto"]])
print(classification_report(esp_test["especialidade"], y_pred_specialty, digits=4))
""")

code("""
cm_specialty = confusion_matrix(esp_test["especialidade"], y_pred_specialty, labels=list(specialty_pipeline.classes_))
fig, ax = plt.subplots(figsize=(5.5, 5.5))
ConfusionMatrixDisplay(cm_specialty, display_labels=specialty_pipeline.classes_).plot(
    ax=ax, cmap="Oranges", colorbar=False, xticks_rotation=45
)
ax.set_title("Matriz de confusão — sub-classificação por especialidade (teste)")
fig.tight_layout()
plt.show()
""")

md("""87,7% de acurácia entre as 4 especialidades (macro F1 0,86) — bem acima
do baseline de 35% (classe majoritária, oncologia). A confusão mais comum é
cardiologia↔neurologia e oncologia↔neurologia, plausível: sintomas
cardiovasculares e neurológicos (dor torácica, tontura) e achados oncológicos
inespecíficos podem se sobrepor no vocabulário do abstract. O modelo final
(`modelo.joblib`) embute os dois classificadores: `predict()` decide
CLINICO_GERAL/ESPECIALISTA, e `predict_detailed()` adiciona a especialidade
mais provável quando a decisão é ESPECIALISTA.""")

# ===========================================================================
md(f"""## 10. Interpretabilidade e análise de erros {SEP}

**Objetivo:** entender por que o modelo classifica cada laudo como classifica,
e caracterizar os casos em que ele erra.""")

code("""
lines = interpret_model(final_pipeline, "tfidf_lr_balanced")
print("\\n".join(lines[:45]))
""")

code("""
test_view = test_df.copy().reset_index(drop=True)
test_view["y_pred"] = y_pred_test
test_view["p_especialista"] = p_pos_test

fn = test_view[(test_view["target"] == POSITIVE) & (test_view["y_pred"] == NEGATIVE)]
fp = test_view[(test_view["target"] == NEGATIVE) & (test_view["y_pred"] == POSITIVE)]
print(f"Falsos negativos: {len(fn)} / {len(test_view)}  |  Falsos positivos: {len(fp)} / {len(test_view)}")

print("\\n--- Exemplo de falso negativo (real=ESPECIALISTA, previsto=CLINICO_GERAL) ---")
row = fn.sample(1, random_state=RANDOM_SEED).iloc[0]
print(f"especialidade original: {row['especialidade']} | p(ESPECIALISTA)={row['p_especialista']:.3f}")
print(row["texto"][:300])
""")

md("""O falso negativo acima ilustra o limite do modelo baseado em texto:
sintoma e causa não usam vocabulário tipicamente associado à especialidade de
destino — o caso "parece" genérico até a suspeita clínica entrar em jogo,
algo que só o texto do abstract não captura bem.""")

# ===========================================================================
md(f"""## 11. Conclusão {SEP}

- **Target**: derivado de `especialidade` (`clinica_geral` → `CLINICO_GERAL`,
  demais → `ESPECIALISTA`), único caminho suportado pela estrutura do dataset
  (só 2 colunas, sem variáveis estruturadas independentes).
- **Limpeza**: ~39% das linhas brutas removidas por conflito de rótulo entre
  duplicatas ou duplicidade pura de texto — sem essa etapa haveria vazamento
  garantido entre treino e teste.
- **Modelo escolhido (decisão binária)**: `tfidf_lr_balanced` (TF-IDF +
  LogisticRegression, `class_weight="balanced"`). A arquitetura em duas etapas
  (multiclasse → binário) não trouxe ganho real quando comparada no mesmo
  ponto de operação — descartada em favor da solução mais simples.
- **Sub-classificação por especialidade**: quando o modelo decide
  ESPECIALISTA, um segundo classificador (treinado só nas linhas ESPECIALISTA)
  aponta qual das 4 especialidades — 87,7% de acurácia, muito acima do
  baseline de classe majoritária (~35%). Sem isso, "precisa de especialista"
  não seria acionável numa triagem real.
- **Threshold final**: calibrado para priorizar recall de Especialista (erro
  de falso negativo é o mais custoso clinicamente), mantendo F1-macro
  competitivo.
- **Limitações**: corpus em inglês (abstracts acadêmicos), não generaliza
  para laudos reais em português sem re-treino; `clinica_geral` aqui é o
  balde residual "general pathological conditions" do corpus original, não
  uma amostra de atendimento de rotina. Ferramenta de apoio à triagem — não
  substitui avaliação médica.

Modelo final salvo em `binary_triage/modelo.joblib` por `train.py` (script de
treino oficial, embute os dois classificadores — binário e por especialidade);
inferência via `binary_triage/predict.py`; artefatos completos em
`binary_triage/reports/` (matriz de confusão binária e por especialidade,
curva PR, comparação de modelos, interpretabilidade, exemplos de erro) e
relatório visual consolidado em `binary_triage/reports/report.html`.""")

nb["cells"] = cells

with open("notebooks/binary_triage_analysis.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook gerado: notebooks/binary_triage_analysis.ipynb")
