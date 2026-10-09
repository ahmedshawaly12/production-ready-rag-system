# flows/Dockerfile - Prefect worker with data pipeline dependencies
FROM prefecthq/prefect:3-python3.13

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_HTTP_TIMEOUT=300 \
    UV_PROJECT_ENVIRONMENT=/usr/local
#     # PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install pipeline-specific dependencies
COPY pyproject.toml uv.lock /app/
COPY /shared/pyproject.toml /app/shared/
COPY /data_pipeline/pyproject.toml /app/data_pipeline/


# RUN uv sync --frozen --no-install-workspace --no-dev --package data-pipeline
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-install-workspace --no-dev --package data-pipeline

RUN uv sync --frozen --no-install-workspace --no-dev --package data-pipeline

COPY shared/shared/ /app/shared/shared/
COPY data_pipeline/data_pipeline/ /app/data_pipeline/data_pipeline/
# RUN uv sync --frozen --no-dev --package data-pipeline
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --no-dev --package data-pipeline

RUN uv sync --frozen --no-dev --package data-pipeline

CMD ["prefect", "worker", "start", "--pool", "default-agent-pool", "--type", "process"]
