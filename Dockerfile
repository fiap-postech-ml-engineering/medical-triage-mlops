# Base comum usada pelos stages de build.
# Mantém configurações globais do Python e instala o uv.
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /usr/local/bin/


# Stage responsável por resolver e instalar dependências.
# Tudo que é necessário apenas para build fica fora da imagem final.
FROM base AS builder

COPY pyproject.toml uv.lock LICENSE README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev


# Stage final da aplicação.
# Recebe somente o ambiente resolvido e o código necessário para execução.
FROM python:3.13-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Usuário não-root reduz riscos caso a aplicação tenha alguma falha.
RUN addgroup --system app \
    && adduser --system --ingroup app app

COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY pyproject.toml README.md LICENSE ./

RUN mkdir -p /app/data /app/models /app/logs \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
