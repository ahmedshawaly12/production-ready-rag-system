# Load Testing

This directory contains a [Locust](https://locust.io/) load test for the RAG API.

## Prerequisites

- The API must be running and accessible.
- Install Locust from the repository root:

```bash
uv add --dev locust
```

## Run the Test

### Web UI

Start Locust:

```bash
uv run locust -f loadtests/locustfile.py --host http://localhost
```

Open **http://localhost:8089** in your browser, enter the desired number of users and spawn rate, and start the test.

### Headless Mode

Run the test with 20 users, increasing the load by 2 users per second, for 5 minutes:

```bash
uv run locust -f loadtests/locustfile.py --host http://localhost \
    --headless -u 20 -r 2 -t 5m --csv loadtests/results
```

The results are saved as CSV files with the prefix `loadtests/results`.

## Configuration

The following environment variables are supported:

| Variable | Default | Description |
|---|---|---|
| `CACHE_HIT_RATIO` | `0.3` | Probability of selecting a repeated question that may hit the semantic cache. |
| `REQUEST_TIMEOUT` | `120` | Maximum streaming read timeout in seconds. |

Example: run a test without intentionally repeating questions:

```bash
CACHE_HIT_RATIO=0 uv run locust -f loadtests/locustfile.py --host http://localhost
```

## Understanding the Results

The test sends requests to `/api/v1/chat` and `/api/v1/health`. Chat responses are streamed using Server-Sent Events (SSE).

Locust reports the following chat metrics:

- **`chat first token [cache hit]`** — Time until the first answer text arrives for a cached response.
- **`chat first token [cache miss]`** — Time until the first answer text arrives for a response that requires generation.
- **`chat total [cache hit]`** — Total time to receive a cached response.
- **`chat total [cache miss]`** — Total time to receive a generated response.
- **`chat total [failed]`** — Requests that fail due to an HTTP error, stream error, timeout, or missing completion event.

For generation performance, pay particular attention to **cache-miss latency**, failure counts, and p95 response times.

## Notes

- Start with a small number of users before increasing the load.
- Chat requests may call your configured LLM provider and incur costs or hit rate limits.
- Repeated questions may be served from the shared semantic cache, affecting the results.
- Ensure the API and its dependencies are running before starting the test.
