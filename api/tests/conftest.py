from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import chat as chat_module
from api.routes.chat import chat_route
from api.routes.health import health_route
from api.services.guardrails_service import GuardrailsService
from api.services.rag_service import QueryEmbedding

filterwarnings = ["ignore:The anyio.abc.BlockingPortal alias:DeprecationWarning"]


# ---------- plain data stand-ins ----------
@dataclass
class Doc:
    score: float
    text: str
    metadata: dict | None = None


@dataclass
class Chunk:
    content: str = ""
    model: str | None = "fake-model"
    usage: dict | None = None


@pytest.fixture
def make_doc():
    return Doc


@pytest.fixture
def make_chunk():
    return Chunk


# ---------- fakes ----------
class FakeRAGService:
    def __init__(self):
        self.embedding = QueryEmbedding(
            vector=[0.1, 0.2, 0.3], model="fake-embed", prompt_tokens=5
        )
        self.documents = []
        self.chunks = []
        self.stream_error = None
        self.embed_calls = []
        self.retrieve_calls = []
        self.stream_calls = []

    async def embed(self, text):
        self.embed_calls.append(text)
        return self.embedding

    async def retrieve(self, vector):
        self.retrieve_calls.append(vector)
        return self.documents

    async def stream_generate(self, messages, **kwargs):
        self.stream_calls.append(messages)
        for chunk in self.chunks:
            yield chunk
        if self.stream_error:
            raise self.stream_error


class FakeCache:
    def __init__(self):
        self.cached = None
        self.get_error = None
        self.set_error = None
        self.get_calls = []
        self.set_calls = []

    async def get(self, **kwargs):
        self.get_calls.append(kwargs)
        if self.get_error:
            raise self.get_error
        return self.cached

    async def set(self, **kwargs):
        self.set_calls.append(kwargs)
        if self.set_error:
            raise self.set_error


class FakePromptManager:
    def __init__(self):
        self.calls = []

    def compile_prompt(self, **kwargs):
        self.calls.append(kwargs)
        prompt = SimpleNamespace(name=kwargs["name"], version=1)
        messages = [
            {"role": "system", "content": kwargs["context"]},
            {"role": "user", "content": kwargs["question"]},
        ]
        return prompt, messages


class FakeLangfuse:
    def __init__(self):
        self.names = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.names.append(kwargs.get("name"))
        yield MagicMock()


@pytest.fixture
def settings():
    return SimpleNamespace(
        TOP_K=3,
        USED_COLLECTION_NAME="test_collection",
        MIN_RETRIEVAL_SCORE=0.5,
        PROMPT_NAME="rag-prompt",
        TEMPERATURE=0.0,
        MAX_OUTPUT_TOKENS=256,
    )


@pytest.fixture
def fake_rag():
    return FakeRAGService()


@pytest.fixture
def fake_cache():
    return FakeCache()


@pytest.fixture
def fake_prompt_manager():
    return FakePromptManager()


@pytest.fixture
def fake_langfuse():
    return FakeLangfuse()


# ---------- app + client ----------
@pytest.fixture
def app(
    monkeypatch, settings, fake_rag, fake_cache, fake_prompt_manager, fake_langfuse
):
    # propagate_attributes is imported straight from langfuse in chat.py
    monkeypatch.setattr(chat_module, "propagate_attributes", lambda **kw: nullcontext())

    app = FastAPI()  # no lifespan: nothing connects to real services
    app.include_router(health_route)
    app.include_router(chat_route)
    app.state.settings = settings
    app.state.langfuse = fake_langfuse
    app.state.prompt_manager = fake_prompt_manager
    app.state.rag_service = fake_rag
    app.state.cache_service = fake_cache
    app.state.guardrails_service = GuardrailsService()  # real, it's pure
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def parse_sse():
    """Turn an SSE body into [(event_name, data_string), ...]."""

    def _parse(body: str):
        events = []
        for block in body.split("\n\n"):
            if not block.strip():
                continue
            event, data = "message", []
            for line in block.split("\n"):
                if line.startswith("event: "):
                    event = line[len("event: ") :]
                elif line.startswith("data: "):
                    data.append(line[len("data: ") :])
            events.append((event, "\n".join(data)))
        return events

    return _parse
