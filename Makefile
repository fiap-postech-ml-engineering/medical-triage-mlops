#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = medical-triage-mlops
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

.PHONY: help requirements create_environment test test-slow test-cov lint lint-fix lint-fix-unsafe format format-fix format-diff format-verbose check check-slow clean

.DEFAULT_GOAL := help

help:
	@echo "Comandos disponíveis:"
	@echo "  make requirements        - Instala dependências Python (uv sync)"
	@echo "  make create_environment  - Cria ambiente virtual com uv"

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

## Dependências

requirements:
	uv sync

create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Ambiente virtual criado. Ative com:"
	@echo ">>> Windows: .\\.venv\\Scripts\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

## Testes

test:
	uv run python -m pytest tests/ -v --no-cov -m "not slow"

test-slow:
	uv run python -m pytest tests/ -v --no-cov -m "slow"

test-cov:
	uv run python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Lint

lint:
	uv run ruff check src/ tests/ dags/

lint-fix:
	uv run ruff check src/ tests/ dags/ --fix

lint-fix-unsafe:
	uv run ruff check src/ tests/ dags/ --fix --unsafe-fixes

## Formatação

format:
	uv run ruff format --check src/ tests/ dags/

format-fix:
	uv run ruff format src/ tests/ dags/

format-diff:
	uv run ruff format --diff src/ tests/ dags/

format-verbose:
	uv run ruff format -v src/ tests/ dags/

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
