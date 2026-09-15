# Otimização de Latência (ONNX)

## 🗺️ Visão geral

O classificador de triagem roda nativamente via scikit-learn
(`Pipeline.predict_proba()`), que tem overhead de interpretação Python. Para
quantificar o ganho de rodar a inferência via um motor compilado, os dois
pipelines internos do `ThresholdedBinaryClassifier` — `pipeline` (decisão
binária CLINICO_GERAL/ESPECIALISTA) e `specialty_pipeline` (qual
especialidade) — são exportados para **ONNX** e comparados via **ONNX
Runtime**.

| Módulo | O que faz |
|---|---|
| `src/binary_triage/export_onnx.py` | Carrega `models/modelo.joblib` e exporta `pipeline`/`specialty_pipeline` para `models/modelo.onnx`/`models/specialty.onnx` via `skl2onnx.to_onnx()` |
| `src/binary_triage/benchmark_onnx.py` | Mede latência (p50/p95/p99, 300 chamadas, inferência única) de `pipeline.predict_proba()` vs. a sessão ONNX equivalente, salvando `models/reports/onnx_benchmark.json` |

## 🤔 O porquê da escolha dessa arquitetura

- Ambos os pipelines internos são `Pipeline` scikit-learn puro (`ColumnTransformer` com `TfidfVectorizer` → `LogisticRegression`) — o candidato final (`tfidf_lr_balanced`) não usa o `TextMetaFeatures` customizado (esse só é usado no candidato `make_combined`, descartado na comparação de `train.py`). Isso torna a conversão via `skl2onnx` direta, sem precisar de um conversor customizado.
- Exportação separada do pipeline binário e do `specialty_pipeline` (em vez de só o binário) porque a resposta de triagem completa da API inclui a especialidade provável — servir só metade do modelo via ONNX deixaria a demonstração incompleta em relação ao que a API real entrega.
- **Escopo desta branch**: só exportação + benchmark/comparação, sem integrar ONNX à API de inferência (`feat/api-inferencia-docker`) — trocar o motor de inferência em produção é uma decisão de deploy separada, que merece sua própria branch/PR para não misturar escopo com um PR já em revisão.

## 📊 Resultado do benchmark

Medido localmente (300 chamadas após 20 de warmup, uma requisição por vez — não em lote):

| | p50 | p95 | p99 |
|---|---|---|---|
| **scikit-learn** | 2.69ms | 4.87ms | 7.91ms |
| **ONNX Runtime** | 0.16ms | 0.44ms | 0.83ms |

**Speedup no p50: ~16.4x.** Medição local (CPU, sem rede real) — serve como piso relativo entre as duas abordagens, não como SLA de produção. Reproduzível via `uv run python -m src.binary_triage.benchmark_onnx` (requer `models/modelo.joblib` e `models/modelo.onnx` já gerados).

## ⚠️ Limitações e observações

| Limitação | Impacto |
|---|---|
| Threshold calibrado (0.40) não faz parte do grafo ONNX | O `ThresholdedBinaryClassifier.predict()` aplica o threshold em Python puro sobre as probabilidades — a sessão ONNX devolve só `predict_proba()`; qualquer consumidor do `.onnx` precisa reaplicar essa lógica externamente |
| Exige locale `en_US.UTF-8` instalado no sistema | O `StringNormalizer` que o `onnxruntime` usa para o `TfidfVectorizer` falha sem esse locale (`RUNTIME_EXCEPTION`) — relevante caso o ONNX venha a ser servido em produção containerizada no futuro |
| Artefatos `.onnx` não são versionados no git | Reproduzíveis via `export_onnx.py`, seguindo a mesma decisão já tomada para `models/*.joblib` |
| Benchmark mede só o pipeline binário | O `specialty_pipeline` também foi exportado e validado (paridade numérica em teste), mas não entrou na medição de latência comparativa |
