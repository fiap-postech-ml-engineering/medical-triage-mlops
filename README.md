# Medical Triage MLOps

Sistema de triagem automática de laudos médicos: recebe o texto de um laudo/relato clínico e recomenda **para qual especialidade médica** o paciente deve ser encaminhado (ex.: `cardiologia`, `ortopedia`, `neurologia`, `dermatologia`, `pediatria`, `clinica_geral`, `ginecologia`, `psiquiatria` — as classes reais vêm do dataset de treino, não são fixas no código).

FIAP MLE Tech Challenge — **Fase 3**. Tema central: deploy de modelo em produção com pipeline CI/CD, monitoramento e otimização de latência.

> Este projeto reaproveita a base de engenharia (API, config, logging, testes, Docker, tooling) construída na [Fase 2](https://github.com/fiap-postech-ml-engineering/ecommerce-recsys-mlops) (sistema de recomendação de e-commerce), adaptada para o novo domínio (classificação de texto) e com os componentes exigidos nesta fase: CI/CD, Airflow, Prometheus/Grafana e otimização ONNX.

## Arquitetura

Ver [`docs/deploy_architecture.md`](docs/deploy_architecture.md) para a decisão de arquitetura de deploy em nuvem (batch vs. real-time), o funcionamento da API e as limitações atuais.

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Status geral do serviço |
| `GET` | `/health` | Health check (API + modelo carregado) |
| `POST` | `/classify` | Classifica um laudo e retorna a especialidade recomendada |
| `GET` | `/metrics` | Métricas Prometheus |

## Stack

- **API**: FastAPI + Uvicorn
- **Modelo**: scikit-learn (`TfidfVectorizer` + `RandomForestClassifier`), com otimização via **ONNX Runtime**
- **Orquestração de retreino**: Apache Airflow (`dags/triage_training_dag.py`)
- **CI/CD**: GitHub Actions (`.github/workflows/ci.yml`) — lint → test → build
- **Monitoramento**: Prometheus + Grafana via Docker Compose
- **Empacotamento**: Docker (multi-stage), `uv` para dependências

## Quickstart

```bash
# 1. Instalar dependências
uv sync --extra dev

# 2. Configurar variáveis de ambiente
cp .env.example .env

# 3. Treinar o classificador (usa o dataset de exemplo em data/raw/laudos_sample.csv)
#    Antes de treinar, ajuste DATASET_PATH no .env para o dataset real do hospital
#    (mínimo de 2.000 amostras — ver "Dataset" abaixo).
make train

# 4. (Opcional) Converter para ONNX e comparar latência
make onnx-export
make latency-compare

# 5. Subir a API localmente
make init
# -> http://localhost:8000/docs
```

### Subindo a stack completa (API + Prometheus + Grafana)

```bash
make docker-up
```

- API: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (login `admin` / `admin`, dashboard "Medical Triage API" já provisionado)

Gere requisições para ver os gráficos populando:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"laudo_texto": "Paciente relata dor torácica intensa com irradiação para o braço esquerdo."}'
```

### Testes e lint

```bash
make check        # lint + format check + testes
make test-cov      # testes com relatório de cobertura
```

## Dataset

O treino espera um CSV com pelo menos duas colunas (nomes configuráveis via `TEXT_COLUMN`/`LABEL_COLUMN` no `.env`):

| texto | especialidade |
|---|---|
| "Paciente relata dor torácica..." | cardiologia |

Dois datasets estão disponíveis:

- **`data/raw/laudos_sample.csv`** (48 linhas, versionado) — só para smoke-test do pipeline ponta a ponta, não usar para avaliar qualidade do modelo.
- **[Medical Abstracts TC Corpus](https://www.kaggle.com/datasets/saharalaa/medical-abstracts-tc-corpus)** (Kaggle, ~14.400 amostras, em inglês) — dataset real sugerido no PDF do Tech Challenge. `notebooks/01_eda_medical_abstracts.ipynb` baixa o dataset (endpoint público do Kaggle, sem precisar de `kaggle.json`), faz a EDA, mapeia as 5 classes originais (`condition_label`) para especialidade (`cardiologia`, `clinica_geral`, `gastroenterologia`, `neurologia`, `oncologia`), grava `data/raw/laudos.csv` no formato esperado pelo pipeline e já roda o treino real — usa essa saída como `MODEL_PATH` padrão da API.

  **Caveat de idioma**: os abstracts são em inglês; o modelo treinado com esse dataset não generaliza bem para laudos em português. Ver nota no topo do notebook.

Rode o notebook antes de `make train` se quiser o modelo treinado no dataset real em vez do dataset de exemplo:

```bash
uv run jupyter notebook notebooks/01_eda_medical_abstracts.ipynb
```

## Retreino via Airflow

```bash
export AIRFLOW_HOME=$(pwd)/.airflow
uv sync --extra airflow
uv run airflow standalone
```

Aponte o `dags_folder` do `airflow.cfg` gerado para a pasta `dags/` deste repositório (ou copie/symlink `dags/triage_training_dag.py`). A DAG `triage_training_pipeline` roda `load_data -> train_and_save`, reaproveitando `src/training/train.py` — o mesmo código usado por `make train`.

## CI/CD

`.github/workflows/ci.yml` roda em todo push/PR: `ruff check` + `ruff format --check` → `pytest --cov` → `docker build`. Cada etapa depende da anterior (falha rápida).

## Estrutura do projeto

```
├── src/
│   ├── api/            # FastAPI: app, rotas, schemas, inferência, middleware
│   ├── training/        # Treino, conversão ONNX, comparação de latência
│   ├── config.py         # Settings (pydantic-settings)
│   ├── logging_config.py # Logging estruturado (texto ou JSON) com request_id
│   └── metrics.py        # Instrumentação Prometheus
├── dags/                 # DAG Airflow de retreino
├── monitoring/            # prometheus.yml + provisioning/dashboard do Grafana
├── .github/workflows/     # CI (lint -> test -> build)
├── tests/                 # pytest
├── docs/                  # Arquitetura de deploy, model card
├── data/raw/               # Dataset de treino (CSV)
└── models/                  # Artefatos treinados (.joblib / .onnx), git-ignored
```

## O que foi reaproveitado da Fase 2

| Componente | Origem | Adaptação |
|---|---|---|
| Skeleton FastAPI (`app.py`, `middleware.py`, `routes/health.py`) | Fase 2 | Domínio trocado (recomendação → triagem); `middleware.py` ganhou instrumentação Prometheus |
| `config.py` / `logging_config.py` | Fase 2 | Hiperparâmetros de recsys/MLflow/DVC removidos; campos do classificador de texto adicionados |
| `Dockerfile` | Fase 2 | Reaproveitado sem alterações (build multi-stage já genérico) |
| `docker-compose.yml` | Fase 2 | Serviços `prometheus`/`grafana` adicionados |
| `Makefile`, `.pre-commit-config.yaml`, `pyproject.toml` (lint/test config) | Fase 2 | Targets de DVC/MLflow removidos; dependências trocadas (recsys → NLP/ONNX/Prometheus) |
| Estrutura de testes (`tests/test_api.py`, `test_config.py`, `test_logging_config.py`) | Fase 2 | Casos reescritos para os novos endpoints/config; padrão de teste mantido |
| `docs/deploy_architecture.md` (template) | Fase 2 | Reescrito para o cenário hospitalar + decisão batch vs. real-time |

Itens que **não existiam** na Fase 2 e foram criados do zero: pipeline CI/CD (`.github/workflows/ci.yml`), DAG Airflow (`dags/`), stack Prometheus/Grafana (`monitoring/`), treino do classificador NLP e exportação/benchmark ONNX (`src/training/`).
