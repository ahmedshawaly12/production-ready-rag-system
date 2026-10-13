# Load Testing

A [Locust](https://locust.io/) load test for the RAG API. It shows how many users the system handles and how latency changes as load grows.

## Folder content

| File | Purpose |
|---|---|
| `locustfile.py` | The test: simulated users calling `/api/v1/chat` and `/api/v1/health` |
| `results_*.csv` | Output of earlier headless runs (stats, history, failures, exceptions) |

## Prerequisites

- The API is running and reachable (see the [root README](../README.md#setup-from-zero-to-first-answer)), and Qdrant already contains documents. Without data, every answer is a fast "not enough information" reply and the test shows nothing useful.
- The `rpm` limits in `docker/litellm_proxy/config.yml` are set to your provider plan's real limits. Every chat request calls the embedding model, even a cache hit.
- Locust is installed. From the repository root:

```bash
uv add --dev locust
```

## Run the test

### Web UI

```bash
uv run locust -f loadtests/locustfile.py --host http://localhost
```

Open **http://localhost:8089**, enter the number of users and the spawn rate, and start the test. Charts and tables update live.

### Headless mode

Run with 20 users, adding 2 users per second, for 5 minutes:

```bash
uv run locust -f loadtests/locustfile.py --host http://localhost \
    --headless -u 20 -r 2 -t 5m --csv loadtests/results
```

The results are saved as CSV files with the prefix `loadtests/results`.

| Flag | Meaning |
|---|---|
| `--host` | Base URL of the API (nginx, port 80, in the Docker setup) |
| `-u` | Number of simulated users |
| `-r` | Users started per second |
| `-t` | Test duration (`30s`, `5m`, `1h`) |
| `--csv` | Prefix for the result files |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `CACHE_HIT_RATIO` | `0.3` | Probability of picking a repeated question that may hit the semantic cache. |
| `REQUEST_TIMEOUT` | `120` | Maximum streaming read timeout in seconds. |

Example: a run without intentionally repeated questions, to measure the worst case (everything is generated):

```bash
CACHE_HIT_RATIO=0 uv run locust -f loadtests/locustfile.py --host http://localhost
```

## Understanding the results

The test sends requests to `/api/v1/chat` and `/api/v1/health`. Chat responses are streamed with Server-Sent Events, so latency is reported in two ways.

| Metric | Meaning |
|---|---|
| `chat first token [cache hit]` | Time until the first answer text arrives for a cached response |
| `chat first token [cache miss]` | Time until the first answer text arrives for a response that needs generation |
| `chat total [cache hit]` | Total time to receive a cached response |
| `chat total [cache miss]` | Total time to receive a generated response |
| `chat total [failed]` | Requests that fail: HTTP error, stream error, timeout, or missing completion event |

For generation performance, look at the **cache-miss latency**, the failure count, and the p95 response times. First-token time is what users feel as "how fast it starts answering".

### What to watch while it runs

- **Grafana** (http://localhost:3000): request rate, latency and errors in the FastAPI dashboard, CPU and memory in Node Exporter, search time in Qdrant.
- **Langfuse**: traces show where time goes (embedding, retrieval, generation).
- **Failures with 429 or rate-limit messages** point to the provider limits (`rpm`), not to your code.

## Notes

- Start with a small number of users, then increase.
- Chat requests call your configured LLM provider, so they can cost money or hit rate limits.
- Repeated questions are served from the shared semantic cache and make the results look faster than real traffic. Use `CACHE_HIT_RATIO` to control this.
- Clear the cache between runs if you want comparable numbers.
