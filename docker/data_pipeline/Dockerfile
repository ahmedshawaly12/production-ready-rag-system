# flows/Dockerfile - Prefect worker with data pipeline dependencies
FROM prefecthq/prefect:2-python3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app

# Install pipeline-specific dependencies
# COPY pyproject.toml .
# COPY data_pipeline/pyproject.toml ./data_pipeline/
# RUN uv sync --package data-pipeline
# Copy flow definitions
# COPY data_pipeline/etl.py /app/data_pipeline/
# COPY data_pipeline/deploy.py /app/data_pipeline/
# COPY ["data_pipeline/deploy copy.py" "/app/data_pipeline/"]

CMD ["prefect", "worker", "start", "--pool", "default-agent-pool", "--type", "process"]
