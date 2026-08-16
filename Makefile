#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = medical-triage-mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements create_environment train onnx-export latency-compare test test-slow test-cov lint lint-fix lint-fix-unsafe format format-fix format-diff format-verbose check check-slow clean init docker-build docker-up docker-up-detached docker-down docker-logs docker-check stop

.DEFAULT_GOAL := help

help:
	@echo "Comandos disponíveis:"
	@echo "  make requirements        - Instala dependências Python (uv sync)"
	@echo "  make create_environment  - Cria ambiente virtual com uv"
	@echo "  make train               - Treina o classificador (TF-IDF + RandomForest) e salva em models/"
	@echo "  make onnx-export         - Converte o modelo treinado para ONNX"
	@echo "  make latency-compare     - Compara latência do modelo sklearn vs. ONNX"

	@echo "  make test                - Roda testes com output verboso (exclui slow)"
	@echo "  make test-slow           - Roda apenas os testes marcados como slow"
	@echo "  make test-cov            - Roda testes com cobertura (relatório HTML)"

	@echo "  make lint                - Verifica estilo do código com Ruff"
	@echo "  make lint-fix            - Corrige automaticamente issues de linting"
	@echo "  make lint-fix-unsafe     - Corrige automaticamente issues com unsafe-fixes"
	@echo "  make format              - Verifica formatação sem modificar (Ruff)"
	@echo "  make format-fix          - Formata código com Ruff"
	@echo "  make format-diff         - Mostra diferenças de formatação sem modificar"
	@echo "  make format-verbose      - Formata código com output verboso"

	@echo "  make check               - Executa lint, format e testes (sequencial, exclui slow)"
	@echo "  make check-slow          - Executa lint, format e todos os testes (incluindo slow)"

	@echo "  make clean               - Remove arquivos temporários"
	@echo "  make init                - Inicia API local com uvicorn"
	@echo "  make stop                - Para serviços Docker"

	@echo "  make docker-build        - Builda os serviços Docker"
	@echo "  make docker-up           - Sobe os serviços Docker"
	@echo "  make docker-up-detached  - Sobe os serviços Docker em background"
	@echo "  make docker-down         - Para e remove os containers"
	@echo "  make docker-logs         - Mostra logs do serviço app"
	@echo "  make docker-check        - Sobe, valida a API e derruba os containers"

## Dependências

requirements:
	uv sync

create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Ambiente virtual criado. Ative com:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Modelo

train:
	uv run python -m src.training.train

onnx-export:
	uv run python -m src.training.onnx_export

latency-compare:
	uv run python -m src.training.latency_compare

## Testes

test:
	uv run python -m pytest tests/ -v --no-cov -m "not slow"

test-slow:
	uv run python -m pytest tests/ -v --no-cov -m "slow"

test-cov:
	uv run python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Lint

lint:
	uv run ruff check src/ tests/

lint-fix:
	uv run ruff check src/ tests/ --fix

lint-fix-unsafe:
	uv run ruff check src/ tests/ --fix --unsafe-fixes

## Formatação

format:
	uv run ruff format --check src/ tests/

format-fix:
	uv run ruff format src/ tests/

format-diff:
	uv run ruff format --diff src/ tests/

format-verbose:
	uv run ruff format -v src/ tests/

## Checks combinados

check: lint format test
	@echo "✓ Todos os checks passaram!"

check-slow: lint format test test-slow
	@echo "✓ Todos os checks (incluindo slow) passaram!"

## Limpeza

clean:
	rm -rf .pytest_cache .coverage htmlcov .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

#################################################################################
# SERVIÇOS                                                                      #
#################################################################################

init:
	uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up

docker-up-detached:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f app

docker-check:
	@set -e; \
	trap 'docker compose down' EXIT; \
	docker compose up -d --build; \
	curl --fail --silent --show-error --retry 12 --retry-delay 5 --retry-connrefused http://localhost:8000/health

stop: docker-down
