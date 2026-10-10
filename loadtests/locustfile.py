"""
Locust load test for the RAG API.

Run with the web UI:
    uv run locust -f loadtests/locustfile.py --host http://localhost

Run headless:
    uv run locust -f loadtests/locustfile.py --host http://localhost \
        --headless -u 20 -r 2 -t 5m --csv loadtests/results

Environment variables:
    CACHE_HIT_RATIO: Share of questions drawn from a repeated pool (default 0.3).
    REQUEST_TIMEOUT: Streaming read timeout in seconds (default 120).
"""

import codecs
import json
import os
import random
import re
import time
from contextlib import suppress

import requests
from locust import HttpUser, between, events, task

CACHE_HIT_RATIO = float(os.getenv("CACHE_HIT_RATIO", "0.3"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))

CHAT_PATH = "/api/v1/chat"
HEALTH_PATH = "/api/v1/health"

HTTP_OK = 200
CONNECT_TIMEOUT = 5
SSE_CONTENT_MARKER = "event: token\ndata: "

EVENT_NAME_RE = re.compile(r"^event: (\w+)$", re.MULTILINE)
DONE_RE = re.compile(r"event: done\ndata: (\{.*?\})")
ERROR_RE = re.compile(r"event: error\ndata: (\{.*?\})")

REPEATED_QUESTIONS = [
    "Which foods are rich in vitamin C?",
    "What are good sources of protein for vegetarians?",
    "How much fiber should an adult eat per day?",
    "What is the difference between saturated and unsaturated fat?",
    "Which foods contain a lot of iron?",
]

FOODS = [
    "spinach",
    "lentils",
    "salmon",
    "oats",
    "almonds",
    "eggs",
    "yogurt",
    "broccoli",
    "chickpeas",
    "bananas",
    "avocado",
    "quinoa",
    "tofu",
    "sweet potatoes",
    "oranges",
    "walnuts",
    "brown rice",
    "mushrooms",
    "cottage cheese",
    "blueberries",
    "tuna",
    "beef liver",
    "kale",
    "peanuts",
]

NUTRIENTS = [
    "protein",
    "fiber",
    "iron",
    "calcium",
    "vitamin C",
    "vitamin D",
    "magnesium",
    "potassium",
    "omega-3",
    "zinc",
    "vitamin B12",
    "folate",
]

TEMPLATES = [
    "How much {nutrient} is in {food}?",
    "Is {food} a good source of {nutrient}?",
    "Which has more {nutrient}, {food} or {food2}?",
    "Can {food} help me get enough {nutrient}?",
]


def pick_question() -> str:
    """Generate a question for either a likely cache hit or cache miss."""
    if random.random() < CACHE_HIT_RATIO:
        return random.choice(REPEATED_QUESTIONS)

    food, food2 = random.sample(FOODS, 2)
    return random.choice(TEMPLATES).format(
        food=food,
        food2=food2,
        nutrient=random.choice(NUTRIENTS),
    )


def report(
    name: str,
    response_time_ms: float,
    length: int = 0,
    exception: Exception | None = None,
) -> None:
    """Record a custom request metric in Locust."""
    events.request.fire(
        request_type="POST",
        name=name,
        response_time=response_time_ms,
        response_length=length,
        exception=exception,
        context={},
    )


def elapsed_ms(start: float) -> float:
    """Return elapsed time in milliseconds."""
    return (time.perf_counter() - start) * 1000


def find_first_token(buffer: str) -> float | None:
    """Return whether answer text has started, based on the current buffer."""
    marker_index = buffer.find(SSE_CONTENT_MARKER)
    if marker_index == -1:
        return None

    answer = buffer[marker_index + len(SSE_CONTENT_MARKER) :].lstrip()
    if answer and not answer.startswith("event:"):
        return 0.0

    return None


def read_stream(
    response: requests.Response,
    start: float,
) -> tuple[str, float | None]:
    """Read the SSE response and measure time to the first answer token."""
    buffer = ""
    first_token_ms = None
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    for chunk in response.iter_content(chunk_size=None):
        if not chunk:
            continue

        buffer += decoder.decode(chunk)

        if first_token_ms is None and find_first_token(buffer) is not None:
            first_token_ms = elapsed_ms(start)

    # Flush any incomplete UTF-8 sequence remaining at EOF.
    buffer += decoder.decode(b"", final=True)
    return buffer, first_token_ms


def report_failure(
    start: float,
    exception: Exception,
    length: int = 0,
) -> None:
    """Record a failed chat request."""
    report(
        "chat total [failed]",
        elapsed_ms(start),
        length,
        exception,
    )


def report_stream_result(
    buffer: str,
    first_token_ms: float | None,
    start: float,
) -> None:
    """Validate the SSE stream and report timing metrics."""
    total_ms = elapsed_ms(start)
    length = len(buffer.encode("utf-8"))
    event_names = EVENT_NAME_RE.findall(buffer)

    if "error" in event_names or "done" not in event_names:
        error = ERROR_RE.search(buffer)
        detail = error.group(1) if error else "Stream ended without a done event"
        report_failure(start, RuntimeError(detail), length)
        return

    cached = False
    done = DONE_RE.search(buffer)

    if done:
        with suppress(json.JSONDecodeError):
            cached = bool(json.loads(done.group(1)).get("cached"))

    cache_label = "cache hit" if cached else "cache miss"

    if first_token_ms is not None:
        report(f"chat first token [{cache_label}]", first_token_ms)

    report(f"chat total [{cache_label}]", total_ms, length)


def execute_chat_request(
    session: requests.Session,
    url: str,
    question: str,
) -> None:
    """Send a request, consume its SSE stream, and record metrics."""
    start = time.perf_counter()

    try:
        with session.post(
            url,
            json={"question": question},
            stream=True,
            timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
        ) as response:
            if response.status_code != HTTP_OK:
                report_failure(
                    start,
                    RuntimeError(f"HTTP {response.status_code}"),
                )
                return

            buffer, first_token_ms = read_stream(response, start)

    except requests.RequestException as exc:
        report_failure(start, exc)
        return

    report_stream_result(buffer, first_token_ms, start)


class ChatUser(HttpUser):
    """Simulate users calling the RAG API."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        self.http = requests.Session()
        self.http.headers.update({"Accept": "text/event-stream"})

    @task(1)
    def health(self) -> None:
        self.client.get(HEALTH_PATH, name="health")

    @task(10)
    def chat(self) -> None:
        url = self.host.rstrip("/") + CHAT_PATH
        execute_chat_request(self.http, url, pick_question())
