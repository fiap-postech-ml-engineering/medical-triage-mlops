# Arquitetura de Deploy

## 🗺️ Visão geral

Microsserviço stateless exposto via API REST síncrona, implementado com **FastAPI + Uvicorn**, containerizado via **Docker** (build multi-stage) e orquestrado localmente via **docker-compose** junto com **Prometheus** e **Grafana**. O serviço carrega o pipeline de classificação de especialidade médica (TF-IDF + RandomForest, com opção de servir a versão convertida para **ONNX**) direto do disco (`models/`), sem depender de um registry externo.

A API expõe os seguintes endpoints (ver `src/api/routes/`):

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/` | Informações da API e status do serviço |
| `GET` | `/health` | Health check — status da API e se o modelo está carregado (`triage_service.is_ready`) |
| `POST` | `/classify` | Classificação da especialidade recomendada para um laudo/relato de texto |
| `GET` | `/metrics` | Métricas no formato Prometheus (`prometheus_client`) |

A documentação interativa é gerada automaticamente pelo FastAPI e pode ser acessada em `/docs` (Swagger) ou `/redoc` quando a API estiver rodando.

## ☁️ Estratégia de deploy em nuvem (batch vs. real-time)

Cenário: hospital precisa direcionar o paciente para a especialidade certa a partir do texto do laudo/relato, no momento do atendimento — a decisão alimenta o fluxo de encaminhamento em tempo real, não um relatório periódico.

- **Escolha: real-time, serviço containerizado em nuvem (ex.: AWS ECS Fargate ou Cloud Run no GCP)**, atrás de um load balancer, com autoscaling horizontal baseado em CPU/latência.
- **Por que não batch**: encaminhar o paciente é uma decisão pontual, síncrona, que precisa da resposta em milissegundos para o profissional de triagem seguir o atendimento — processar em lote (ex.: uma vez por hora) atrasaria o fluxo assistencial sem necessidade, já que o modelo é leve (TF-IDF + RandomForest/ONNX) e a inferência é barata.
- **Por que não Kubernetes desde já**: a carga esperada (um posto de triagem, poucas requisições por minuto) não justifica a complexidade operacional de um cluster K8s; um serviço serverless/gerenciado (Fargate/Cloud Run) cobre autoscaling e recuperação de falhas com muito menos overhead. K8s vira a escolha natural se o número de hospitais atendidos crescer e exigir múltiplos serviços/namespaces compartilhando o cluster.
- **Observabilidade em nuvem**: a mesma stack local (Prometheus + Grafana) pode ser substituída por Amazon Managed Prometheus/Grafana ou Google Cloud Monitoring sem mudar a instrumentação da API (`prometheus_client` expõe `/metrics` de forma agnóstica ao ambiente).
- **Retreino**: a DAG do Airflow (`dags/triage_training_dag.py`) roda em um Airflow gerenciado (ex.: MWAA na AWS ou Cloud Composer no GCP), publicando um novo `models/classifier.joblib`/`.onnx` que o serviço recarrega em um novo deploy — não há promoção automática em runtime nesta fase (ver limitações).

## 🤔 O porquê da escolha dessa arquitetura

FastAPI foi escolhido pela validação automática de payload via Pydantic, performance assíncrona nativa e curva de instrumentação baixa com `prometheus_client`.

- O modelo é carregado uma única vez no startup (`lifespan`, ver `src/api/app.py`) e mantido em memória, evitando I/O de disco a cada request.
- A opção `USE_ONNX` (ver `src/config.py`) permite comparar, sem mudar código de rota, a latência do pipeline scikit-learn original contra a versão convertida para ONNX Runtime (`make latency-compare`).

## 🔁 Qual o funcionamento da API

1. **Startup (`lifespan`)** — `get_settings()` carrega a configuração e `triage_service.load()` carrega o artefato configurado: `models/classifier.joblib` (sklearn) ou `models/classifier.onnx` + `models/classifier_labels.json` (ONNX), conforme `USE_ONNX`.
2. **Middleware de observabilidade** — gera ou propaga o header `X-Request-ID` (UUID), mede a latência da requisição, loga `request.completed` (método, path, status, latência, IP) e registra as métricas Prometheus `http_requests_total` e `http_request_duration_seconds` (ver `src/api/middleware.py`, `src/metrics.py`).
3. **Validação Pydantic** — `TriageRequest` exige `laudo_texto` não vazio (payload inválido retorna `422`, ver `tests/test_api.py`).
4. **Checagem de disponibilidade** — se o modelo não estiver carregado (`is_ready=False`), a API retorna `503` antes de tentar inferir (degradação graciosa, ver `src/api/routes/classify.py`).
5. **Inferência** — `triage_service.classify(texto)` roda o pipeline (ou sessão ONNX) e retorna a especialidade mais provável, a confiança e a distribuição de probabilidade por classe.
6. **Resposta** — `TriageResponse` com `especialidade_recomendada`, `confianca` e `probabilidades`.

## ⚙️ Configurações e ambiente

Configuração via `.env` / `Pydantic Settings` (`src/config.py`), sem valores hardcoded nas rotas:

| Variável | Descrição |
|---|---|
| `APP_ENV` | Ambiente da aplicação (`development`/`production`, etc.) |
| `MODEL_VERSION` | Versão exposta no título/health da API |
| `USE_ONNX` | Se `true`, serve o runtime ONNX; se `false`, serve o pipeline sklearn |
| `MODEL_PATH` / `ONNX_MODEL_PATH` | Caminho do artefato treinado |
| `DATASET_PATH` / `TEXT_COLUMN` / `LABEL_COLUMN` | Onde e como ler os dados de treino |
| `LATENCY_WARN_MS` | Limiar de latência para log em nível `WARNING` |

## 🐳 Containerização e monitoramento

- **`Dockerfile`** — multi-stage: `base` (Python 3.13-slim + `uv`) → `builder` (`uv sync --frozen --no-dev`) → `runtime` (usuário não-root, apenas `.venv` resolvido + código necessário).
- **`docker-compose.yml`** — serviços `app` (FastAPI), `prometheus` (scrape em `/metrics` a cada 5s, config em `monitoring/prometheus.yml`) e `grafana` (dashboard provisionado automaticamente em `monitoring/grafana/dashboards/triage-api-dashboard.json`, acessível em `http://localhost:3000`, login `admin`/`admin`).
- Comandos: `make docker-build`, `make docker-up[-detached]`, `make docker-down`/`make stop`, `make docker-logs`, `make docker-check` (smoke test: sobe, valida `/health`, derruba).

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Processo único (sem workers paralelos) | Sem paralelismo real de CPU em picos de carga |
| Sem autoscaling (execução local) | Gargalo sob alta demanda — endereçado na estratégia de nuvem acima |
| Sem versionamento de endpoint (`/v1/`) | Breaking changes afetam todos os clientes |
| Sem autenticação | API aberta, sem controle de acesso — inadequado para dado clínico real em produção |
| Sem rate limiting | Vulnerável a abuso ou sobrecarga acidental |
| Modelo carregado só no startup (`lifespan`) | Retreinar via Airflow não afeta uma API já em execução — exige reiniciar o processo/redeploy para recarregar |
| Dataset de exemplo pequeno (`data/raw/laudos_sample.csv`) | Serve para validar o pipeline ponta a ponta; substituir por um dataset real com 2.000+ amostras antes de avaliar qualidade do modelo |
