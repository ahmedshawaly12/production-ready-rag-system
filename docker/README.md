# RAG System: Docker Stack

A production-style RAG system run with Docker Compose. It has a FastAPI chat API, a Prefect data pipeline, a vector database, a semantic cache, and a full monitoring stack.

## Architecture

```
                    ┌──────────────── Chat ────────────────┐
 Client ──► nginx ──► FastAPI ──┬──► Redis    (semantic cache)
                                ├──► Qdrant   (vector search)
                                └──► LiteLLM  (LLM / embeddings proxy)

                    ┌────────────── Ingestion ─────────────┐
 Prefect server ──► Prefect worker ──► MinIO (files) ──► Qdrant

                    ┌────────────── Monitoring ────────────┐
 Prometheus ◄── FastAPI, Qdrant, Node Exporter, Postgres Exporter
     └──► Grafana (dashboards)
```

## Folder structure

```
docker/
├── docker-compose.yml        # all services
├── api/Dockerfile            # FastAPI image
├── data_pipeline/Dockerfile  # Prefect worker image
├── nginx/default.conf        # reverse proxy config
├── litellm_proxy/config.yml  # LiteLLM model config
├── prometheus/prometheus.yml # scrape targets
└── env/
    ├── .env.example.*        # templates (committed)
    └── .env.*                # your real values (git-ignored)
```

## Quick start

All commands run from the `docker/` folder.

**1. Create your env files** from the templates:

```bash
cd docker/env
for f in .env.example.*; do cp -n "$f" "${f/.example/}"; done
```

Open each `.env.*` file and fill in your values (API keys, passwords).

**2. Start everything:**

```bash
cd ..
docker compose up -d --build
```

**3. Check that it works:**

```bash
docker compose ps
curl http://localhost/api/v1/health
```

Expected response: `{"status":"healthy"}`

**4. Ask a question:**

```bash
curl -N -X POST http://localhost/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is vitamin C?"}'
```

The answer streams back as Server-Sent Events (`documents`, `token`, `done`).

## Services

| Service | Purpose | URL / Port |
|---|---|---|
| **nginx** | Public entry point, proxies to the API | http://localhost |
| **fastapi** | Chat and health API | http://localhost:8000 |
| **litellm** | LLM gateway | http://localhost:4000 |
| **qdrant** | Vector database | http://localhost:6333/dashboard |
| **redis** | Semantic cache (Redis Stack) | `localhost:6379` |
| **redis insight** | Redis web UI | http://localhost:8001 |
| **minio** | Object storage for source files | API `:9000`, console http://localhost:9001 |
| **postgres** | Metadata database (Prefect, LiteLLM) | internal only |
| **prefect-server** | Pipeline orchestration UI | http://localhost:4200 |
| **worker** | Runs the data pipeline flows | internal only |
| **prometheus** | Metrics collection | http://localhost:9090 |
| **grafana** | Dashboards | http://localhost:3000 |
| **node-exporter** | Host metrics | http://localhost:9100 |
| **postgres-exporter** | Postgres metrics | http://localhost:9187 |

### API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `POST` | `/api/v1/chat` | Ask a question (streams SSE) |

The metrics endpoint is only reachable inside the Docker network, where Prometheus scrapes it. Nginx blocks it from outside.

## Monitoring with Grafana

Open http://localhost:3000 and sign in with the credentials from `env/.env.grafana`.

### 1. Add the Prometheus data source

**Connections → Data sources → Add data source → Prometheus**

- URL: `http://prometheus:9090`
- Click **Save & test**

### 2. Import the dashboards

**Dashboards → New → Import**, type the ID, click **Load**, choose the **Prometheus** data source, then **Import**.

| Dashboard | ID | What it shows |
|---|---|---|
| [FastAPI Observability](https://grafana.com/grafana/dashboards/18739-fastapi-observability/) | `18739` | Request rate, latency, errors |
| [Node Exporter Full](https://grafana.com/grafana/dashboards/1860-node-exporter-full/) | `1860` | CPU, memory, disk, network |
| [Qdrant](https://grafana.com/grafana/dashboards/23033-qdrant/) | `23033` | Collections, search performance |
| [PostgreSQL Exporter](https://grafana.com/grafana/dashboards/12485-postgresql-exporter/) | `12485` | Connections, queries, database size |

### 3. Verify the scrape targets

Open http://localhost:9090/targets. The `fastapi`, `qdrant`, `node-exporter`, `postgres` and `prometheus` jobs should all be **UP**. If a dashboard is empty, check here first.

## Everyday commands

```bash
docker compose up -d                  # start
docker compose ps                     # status
docker compose logs -f fastapi        # follow one service's logs
docker compose up -d --build fastapi  # rebuild and restart one service
docker compose restart nginx          # restart one service
docker compose down                   # stop (keeps your data)
docker compose down -v                # stop and DELETE all volumes (data is lost)
```

## Running the tests

From the repository root (outside `docker/`):

```bash
uv sync --all-packages
uv run pytest api/tests -v
```

The tests use fakes, so they don't need Docker or any running service.

## Troubleshooting

| Problem | Fix |
|---|---|
| `http://localhost` shows an Apache page | Another web server owns port 80. Stop it with `sudo systemctl disable --now apache2`, or change nginx to `"8080:80"` in the compose file. |
| `502 Bad Gateway` from nginx | The API isn't up yet. Check `docker compose logs fastapi`. |
| nginx restarts with `host not found in upstream` | Run `docker compose down && docker compose up -d` so all containers share the same network. |
| Prometheus target is **DOWN** | Check the container logs. Names in `prometheus.yml` must match the compose service names. |
| API can't reach Redis or Qdrant | In `env/.env.app`, use the service names (`redis`, `qdrant`), not `localhost`. |
| Changes to `.env.*` are ignored | Recreate the container: `docker compose up -d --force-recreate <service>`. |
