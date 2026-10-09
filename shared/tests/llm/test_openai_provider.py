import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import RateLimitError

from shared.llm import openai_provider as mod
from shared.llm.openai_provider import EmbeddingResult, OpenAIProvider


# ---------- helpers / fixtures ----------
def make_rate_limit_error() -> RateLimitError:
    req = httpx.Request("POST", "https://api.test/v1/embeddings")
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=req), body=None
    )


def embedding_response(vectors, model="real-embed-model", tokens=5):
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors],
        model=model,
        usage=SimpleNamespace(prompt_tokens=tokens),
    )


class FakeClock:
    """Replaces time.monotonic and asyncio.sleep inside the provider module only,
    so tests never actually wait (and the real event loop is untouched)."""

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self):
        return self.now

    async def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=c.monotonic))
    monkeypatch.setattr(
        mod,
        "asyncio",
        SimpleNamespace(
            Lock=asyncio.Lock,
            sleep=c.sleep,
            CancelledError=asyncio.CancelledError,
        ),
    )
    return c


@pytest.fixture
def provider():
    p = OpenAIProvider(api_key="test-key")
    p.set_embedding_model("embed-model")
    p.set_generation_model("gen-model")
    p.client = MagicMock()  # never hits the network
    return p


# ---------- init / small helpers ----------
def test_empty_api_url_becomes_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        mod, "AsyncOpenAI", lambda **kw: captured.update(kw) or MagicMock()
    )
    OpenAIProvider(api_key="k", api_url="")
    assert captured["base_url"] is None


def test_estimate_tokens(provider):
    # sum(len)//3 + len(texts)  ->  (6 + 3) // 3 + 2 = 5
    assert provider._estimate_tokens(["abcdef", "abc"]) == 5


def test_process_text_truncates_and_strips(provider):
    provider.default_max_input_tokens = 5
    assert provider.process_text("  abc def") == "abc"


# ---------- token budget ----------
async def test_wait_for_budget_records_usage_without_sleeping(provider, clock):
    await provider._wait_for_budget(1000)
    assert clock.slept == []
    assert list(provider._token_window) == [(clock.now, 1000)]


async def test_wait_for_budget_caps_request_at_limit(provider, clock):
    await provider._wait_for_budget(10_000_000)
    assert provider._token_window[0][1] == int(provider.tpm_limit * 0.8)


async def test_wait_for_budget_sleeps_until_window_frees(provider, clock):
    limit = int(provider.tpm_limit * 0.8)
    provider._token_window.append((clock.now, limit))  # budget fully used

    await provider._wait_for_budget(1000)

    assert clock.slept == [pytest.approx(60.1)]
    assert len(provider._token_window) == 1  # old entry evicted, new one added


# ---------- _embed_batch ----------
async def test_embed_batch_success(provider):
    provider._wait_for_budget = AsyncMock()
    resp = embedding_response([[0.1]])
    provider.client.embeddings.create = AsyncMock(return_value=resp)

    assert await provider._embed_batch(["a"]) is resp
    provider.client.embeddings.create.assert_awaited_once_with(
        model="embed-model", input=["a"]
    )


async def test_embed_batch_retries_on_rate_limit(provider, clock):
    provider._wait_for_budget = AsyncMock()
    resp = embedding_response([[0.1]])
    provider.client.embeddings.create = AsyncMock(
        side_effect=[make_rate_limit_error(), resp]
    )

    assert await provider._embed_batch(["a"]) is resp
    assert provider.client.embeddings.create.await_count == 2
    assert len(clock.slept) == 1
    assert 5 <= clock.slept[0] < 6  # first backoff: 5 * 2**0 + jitter


async def test_embed_batch_gives_up_after_max_retries(provider, clock):
    provider._wait_for_budget = AsyncMock()
    provider.client.embeddings.create = AsyncMock(side_effect=make_rate_limit_error())

    assert await provider._embed_batch(["a"], max_retries=3) is None
    assert provider.client.embeddings.create.await_count == 3


# ---------- embed_text_with_meta / embed_text ----------
async def test_embed_with_meta_batches_and_aggregates(provider):
    provider._embed_batch = AsyncMock(
        side_effect=[
            embedding_response([[1.0], [2.0]], tokens=4),
            embedding_response([[3.0]], tokens=6),
        ]
    )

    result = await provider.embed_text_with_meta(["a", "b", "c"], batch_size=2)

    assert isinstance(result, EmbeddingResult)
    assert result.vectors == [[1.0], [2.0], [3.0]]
    assert result.prompt_tokens == 10
    assert result.model == "real-embed-model"
    assert provider._embed_batch.await_count == 2


async def test_embed_with_meta_wraps_single_string(provider):
    provider._embed_batch = AsyncMock(return_value=embedding_response([[1.0]]))
    result = await provider.embed_text_with_meta("hello")
    provider._embed_batch.assert_awaited_once_with(["hello"])
    assert result.vectors == [[1.0]]


async def test_embed_with_meta_returns_none_without_model(provider):
    provider.embedding_model_id = None
    assert await provider.embed_text_with_meta("x") is None


async def test_embed_with_meta_returns_none_if_a_batch_fails(provider):
    provider._embed_batch = AsyncMock(side_effect=[embedding_response([[1.0]]), None])
    assert await provider.embed_text_with_meta(["a", "b"], batch_size=1) is None


async def test_embed_text_returns_only_vectors(provider):
    provider._embed_batch = AsyncMock(return_value=embedding_response([[1.0]]))
    assert await provider.embed_text("x") == [[1.0]]


async def test_embed_text_returns_none_on_failure(provider):
    provider._embed_batch = AsyncMock(return_value=None)
    assert await provider.embed_text("x") is None


# ---------- generate_text ----------
def completion(content="hi", model="real-gen"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model=model,
    )


async def test_generate_text_success_uses_defaults(provider):
    provider.client.chat.completions.create = AsyncMock(return_value=completion())
    msgs = [{"role": "user", "content": "yo"}]

    assert await provider.generate_text(msgs) == "hi"
    assert provider.actual_gen_model_name == "real-gen"
    provider.client.chat.completions.create.assert_awaited_once_with(
        model="gen-model",
        max_tokens=provider.default_max_output_tokens,
        temperature=provider.default_temperature,
        messages=msgs,
    )


async def test_generate_text_zero_temperature_is_respected(provider):
    provider.client.chat.completions.create = AsyncMock(return_value=completion())
    await provider.generate_text([], temperature=0.0)
    assert (
        provider.client.chat.completions.create.call_args.kwargs["temperature"] == 0.0
    )


async def test_generate_text_rate_limit_returns_none(provider):
    provider.client.chat.completions.create = AsyncMock(
        side_effect=make_rate_limit_error()
    )
    assert await provider.generate_text([]) is None


async def test_generate_text_empty_choices_returns_none(provider):
    provider.client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(choices=[], model="m")
    )
    assert await provider.generate_text([]) is None


async def test_generate_text_requires_model(provider):
    provider.generation_model_id = None
    assert await provider.generate_text([]) is None


# ---------- stream_generate_text ----------
def stream_chunk(content=None, usage=None, choices=True, model="m"):
    return SimpleNamespace(
        usage=usage,
        model=model,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content))]
        if choices
        else [],
    )


def make_stream(chunks, error=None):
    async def gen():
        for c in chunks:
            yield c
        if error:
            raise error

    return gen()


async def test_stream_yields_content_then_usage(provider):
    usage = SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    stream = make_stream(
        [
            stream_chunk("Hel"),
            stream_chunk("lo"),
            stream_chunk(usage=usage, choices=False),
        ]
    )
    provider.client.chat.completions.create = AsyncMock(return_value=stream)

    out = [c async for c in provider.stream_generate_text([])]

    assert [c.content for c in out if c.content] == ["Hel", "lo"]
    assert out[-1].usage == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    kwargs = provider.client.chat.completions.create.call_args.kwargs
    assert kwargs["stream"] is True
    assert kwargs["stream_options"] == {"include_usage": True}


async def test_stream_swallows_rate_limit(provider):
    provider.client.chat.completions.create = AsyncMock(
        side_effect=make_rate_limit_error()
    )
    assert [c async for c in provider.stream_generate_text([])] == []


async def test_stream_swallows_generic_errors_after_partial_output(provider):
    stream = make_stream([stream_chunk("partial")], error=RuntimeError("boom"))
    provider.client.chat.completions.create = AsyncMock(return_value=stream)

    out = [c async for c in provider.stream_generate_text([])]
    assert [c.content for c in out] == ["partial"]


async def test_stream_reraises_cancellation(provider):
    stream = make_stream([], error=asyncio.CancelledError())
    provider.client.chat.completions.create = AsyncMock(return_value=stream)

    with pytest.raises(asyncio.CancelledError):
        async for _ in provider.stream_generate_text([]):
            pass
