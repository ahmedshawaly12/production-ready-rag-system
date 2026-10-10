FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# 1) Manifests only, so the dependency layer is cached until deps change
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/
COPY api/pyproject.toml api/
COPY data_pipeline/pyproject.toml data_pipeline/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --no-dev --package api

# 2) Source code, then install the workspace members themselves
COPY shared/shared/ shared/shared/
COPY api/api/ api/api/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --package api

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
