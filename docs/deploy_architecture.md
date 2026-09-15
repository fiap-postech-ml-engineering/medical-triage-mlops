# Arquitetura de Deploy

## 🗺️ Visão geral

Microsserviço stateless exposto via API REST síncrona, implementada com **FastAPI + Uvicorn**, containerizado via **Docker** (build multi-stage) e orquestrado localmente via **docker-compose**. O serviço carrega o `ThresholdedBinaryClassifier` treinado (`src/binary_triage/train.py`) e encapsula toda a lógica de pré-processamento, threshold calibrado e sub-classificação por especialidade — o cliente só envia o texto do laudo e recebe a triagem já decidida.

A API expõe os seguintes endpoints (ver `src/api/routes.py`):

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/health` | Health check da API |
| `POST` | `/classify` | Classifica um laudo em `CLINICO_GERAL`/`ESPECIALISTA`, com especialidade provável quando aplicável |

A documentação interativa é gerada automaticamente pelo FastAPI e pode ser acessada em `/docs` (Swagger) ou `/redoc` quando a API estiver rodando.

## 🤔 O porquê da escolha dessa arquitetura

FastAPI foi escolhido pela validação automática de payload via Pydantic, performance assíncrona nativa e integração direta com o ecossistema Python usado no restante do pipeline (scikit-learn, joblib).

- O modelo é carregado **uma única vez no startup** (`lifespan`, ver `src/api/api.py`) e mantido em memória durante toda a vida do processo, evitando I/O de disco a cada request.
- Inferência com um `Pipeline` scikit-learn (TF-IDF + features estruturadas + classificador) sobre um único texto é da ordem de dezenas de milissegundos em CPU — não há necessidade de processamento assíncrono/em lote para o caso de uso de triagem individual.

## 🔁 Qual o funcionamento da API

Ciclo de vida de uma requisição `POST /classify`:

1. **Startup (`lifespan`)** — `inference.load_model(settings.model_path)` carrega o `models/modelo.joblib` via `joblib.load`. Se o artefato não existir ou não puder ser desserializado, a aplicação **falha no startup** (fail-fast) em vez de subir e falhar silenciosamente na primeira requisição.
2. **Middleware de logging** (`src/api/middleware.py`) — loga método, path, status code e latência de cada requisição.
3. **Validação Pydantic** (`ClassifyRequest`) — exige `texto` não vazio; payload inválido retorna `422`.
4. **Inferência** (`src/api/inference.py`) — `model.predict_detailed()` aplica o pipeline completo: threshold calibrado (0.40, não o padrão 0.5 do scikit-learn) para a decisão binária e, quando classificado como `ESPECIALISTA`, o sub-classificador de especialidade (oncologia/cardiologia/neurologia/gastroenterologia).
5. **Resposta** (`ClassifyResponse`) — classificação, probabilidades da decisão binária, especialidade mais provável (quando aplicável) e a distribuição completa por especialidade.

## ⚙️ Configurações e ambiente

Configuração via `.env` / `pydantic-settings` (`src/config/settings.py`), sem valores hardcoded nas rotas:

| Variável | Descrição |
|---|---|
| `MODEL_PATH` | Caminho do artefato `.joblib` do modelo treinado (default: `models/modelo.joblib`) |

## 🐳 Containerização

- **`Dockerfile`** — multi-stage: `base` (Python 3.12-slim + `uv`) → `builder` (`uv sync --frozen --no-dev`) → `runtime` (usuário não-root, apenas `.venv` resolvido + código necessário).
- **`docker-compose.yml`** — serviço `api`, build a partir do `Dockerfile` (`target: runtime`), bind mounts para hot-reload em dev, healthcheck via `GET /health`, porta `8000`.

## ⚠️ Limitações atuais

| Limitação | Impacto |
|---|---|
| Sem CI/CD de build/push da imagem Docker | `docker build` ainda não faz parte do `ci.yml`; validado manualmente |
| Processo único (sem workers paralelos) | Sem paralelismo real de CPU em picos de carga |
| Sem autoscaling | Gargalo sob alta demanda |
| Sem versionamento de endpoint (`/v1/`) | Breaking changes afetam todos os clientes |
| Sem autenticação | API aberta, sem controle de acesso |
| Sem rate limiting | Vulnerável a abuso ou sobrecarga acidental |
| Sem registry de modelos ativo | Troca de versão do modelo é feita substituindo o arquivo `.joblib` manualmente; exige reiniciar o processo para recarregar (`lifespan` só carrega no startup) |
| Sem métricas Prometheus ainda | Previsto para a próxima branch (`feat/monitoramento-observabilidade`) |
