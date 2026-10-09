import json

import pytest

from api.routes.chat import NO_CONTEXT_ANSWER
from api.routes.schemas.chat import ChatPayloadSchema
from api.services.rag_service import QueryEmbedding

URL = "/api/v1/chat"
QUESTION = "Where do I get vitamin C?"


def names(events):
    return [name for name, _ in events]


# ---------- schema / validation ----------
def test_schema_accepts_question():
    assert ChatPayloadSchema(question="hi").question == "hi"


@pytest.mark.parametrize("body", [{}, {"question": None}, {"question": ["a"]}])
def test_invalid_payload_is_rejected(client, fake_rag, body):
    r = client.post(URL, json=body)
    assert r.status_code == 422
    assert fake_rag.embed_calls == []


# ---------- happy path ----------
def test_streams_answer_with_sources(
    client, fake_rag, fake_cache, make_doc, make_chunk, parse_sse
):
    fake_rag.documents = [
        make_doc(
            0.9, "Citrus fruits contain vitamin C.", {"source": "a.pdf", "page": 3}
        ),
        make_doc(0.8, "Peppers are rich in vitamin C.", {"source": "b.pdf", "page": 1}),
    ]
    fake_rag.chunks = [
        make_chunk("Vitamin C "),
        make_chunk("is in citrus."),
        make_chunk(
            "",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        ),
    ]

    r = client.post(URL, json={"question": QUESTION})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["cache-control"] == "no-cache"

    events = parse_sse(r.text)
    assert names(events) == ["documents", "token", "done"]

    docs = json.loads(events[0][1])
    assert [d["metadata"]["source"] for d in docs] == ["a.pdf", "b.pdf"]
    assert events[1][1] == "Vitamin C is in citrus."
    assert json.loads(events[2][1]) == {"cached": False}


def test_answer_is_written_to_cache(client, fake_rag, fake_cache, make_doc, make_chunk):
    fake_rag.documents = [make_doc(0.9, "text", {"source": "a.pdf", "page": 1})]
    fake_rag.chunks = [make_chunk("answer")]

    client.post(URL, json={"question": QUESTION})

    assert len(fake_cache.set_calls) == 1
    call = fake_cache.set_calls[0]
    assert call["prompt"] == QUESTION
    assert call["response"] == "answer"
    assert call["vector"] == fake_rag.embedding.vector
    assert {"user_id", "session_id", "conversation_id"} <= set(call)


def test_prompt_is_compiled_from_retrieved_context(
    client, settings, fake_rag, fake_prompt_manager, make_doc, make_chunk
):
    fake_rag.documents = [make_doc(0.9, "Citrus text", {"source": "a.pdf", "page": 3})]
    fake_rag.chunks = [make_chunk("ok")]

    client.post(URL, json={"question": QUESTION})

    call = fake_prompt_manager.calls[0]
    assert call["name"] == settings.PROMPT_NAME
    assert call["question"] == QUESTION
    assert call["chat_history"] == []
    assert "[Document 1]" in call["context"]
    assert "a.pdf" in call["context"] and "Citrus text" in call["context"]
    # the compiled messages are what the generator receives
    assert fake_rag.stream_calls[0][-1]["content"] == QUESTION


def test_langfuse_observations_for_cache_miss(
    client, fake_rag, fake_langfuse, make_doc, make_chunk
):
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("a")]

    client.post(URL, json={"question": QUESTION})

    assert fake_langfuse.names == [
        "rag-chat",
        "question-embedding",
        "semantic-cache-lookup",
        "vector-retrieval",
        "prompt-compilation",
        "rag-generation",
        "semantic-cache-write",
    ]


# ---------- guardrails ----------
def test_question_is_masked_before_embedding(client, fake_rag):
    client.post(URL, json={"question": "Email john@example.com about iron"})
    assert fake_rag.embed_calls == ["Email [EMAIL] about iron"]


# ---------- cache ----------
def test_cache_hit_skips_retrieval_and_generation(
    client, fake_rag, fake_cache, parse_sse
):
    fake_cache.cached = {
        "response": "Cached answer",
        "documents": [
            {"score": 0.9, "text": "t", "metadata": {"source": "a.pdf", "page": 1}}
        ],
    }

    r = client.post(URL, json={"question": QUESTION})
    events = parse_sse(r.text)

    assert names(events) == ["documents", "token", "done"]
    # NOTE: on a cache hit the token data is JSON-encoded (quoted)
    assert json.loads(events[1][1]) == "Cached answer"
    assert json.loads(events[2][1]) == {"cached": True}
    assert fake_rag.retrieve_calls == []
    assert fake_rag.stream_calls == []
    assert fake_cache.set_calls == []


def test_cache_is_scoped_by_user_session_conversation(client, fake_cache):
    client.post(URL, json={"question": QUESTION})
    assert set(fake_cache.get_calls[0]) == {
        "vector",
        "user_id",
        "session_id",
        "conversation_id",
    }


def test_cache_lookup_failure_behaves_like_miss(
    client, fake_rag, fake_cache, make_doc, make_chunk, parse_sse
):
    fake_cache.get_error = ConnectionError("redis down")
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("answer")]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events) == ["documents", "token", "done"]
    assert json.loads(events[-1][1]) == {"cached": False}


def test_cache_write_failure_does_not_fail_request(
    client, fake_rag, fake_cache, make_doc, make_chunk, parse_sse
):
    fake_cache.set_error = ConnectionError("redis down")
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("answer")]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert "error" not in names(events)
    assert names(events)[-1] == "done"


# ---------- retrieval edge cases ----------
def test_no_documents_returns_fallback_answer(client, fake_rag, parse_sse):
    fake_rag.documents = []

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events) == ["documents", "token", "done"]
    assert json.loads(events[0][1]) == []
    assert json.loads(events[1][1]) == NO_CONTEXT_ANSWER
    assert fake_rag.stream_calls == []


def test_low_score_documents_are_filtered_out(client, fake_rag, make_doc, parse_sse):
    fake_rag.documents = [make_doc(0.2, "weak", {}), make_doc(0.3, "weaker", {})]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert json.loads(events[1][1]) == NO_CONTEXT_ANSWER  # MIN_RETRIEVAL_SCORE=0.5


def test_only_documents_above_min_score_are_used(
    client, fake_rag, fake_prompt_manager, make_doc, make_chunk, parse_sse
):
    fake_rag.documents = [make_doc(0.9, "strong", {}), make_doc(0.1, "weak", {})]
    fake_rag.chunks = [make_chunk("a")]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert len(json.loads(events[0][1])) == 1
    assert "weak" not in fake_prompt_manager.calls[0]["context"]


# ---------- error paths ----------
def test_embedding_failure_returns_error_event(client, fake_rag, fake_cache, parse_sse):
    fake_rag.embedding = QueryEmbedding(vector=[])

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events) == ["error"]
    assert json.loads(events[0][1]) == {"detail": "Failed to embed the question"}
    assert fake_cache.get_calls == []
    assert fake_rag.retrieve_calls == []


def test_empty_generation_returns_error_and_is_not_cached(
    client, fake_rag, fake_cache, make_doc, make_chunk, parse_sse
):
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("")]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events)[-1] == "error"
    assert json.loads(events[-1][1]) == {"detail": "Generation returned no content"}
    assert fake_cache.set_calls == []


def test_generation_exception_is_reported_not_raised(
    client, fake_rag, fake_cache, make_doc, make_chunk
):
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("partial")]
    fake_rag.stream_error = RuntimeError("boom")

    r = client.post(URL, json={"question": QUESTION})

    assert r.status_code == 200
    assert "Generation failed" in r.text
    assert fake_cache.set_calls == []


# ---------- known issues (strict xfail: remove the marker once fixed) ----------
@pytest.mark.xfail(
    strict=True,
    reason="Tokens are written raw, so a blank line in the LLM output ends the SSE event",
)
def test_newlines_in_streamed_tokens_keep_sse_framing(
    client, fake_rag, make_doc, make_chunk, parse_sse
):
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("Line one\n\nLine two")]

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events) == ["documents", "token", "done"]


@pytest.mark.xfail(
    strict=True,
    reason="'event: token' / 'data: ' is left unterminated, so the error event is glued onto it",
)
def test_mid_stream_error_arrives_as_its_own_event(
    client, fake_rag, make_doc, make_chunk, parse_sse
):
    fake_rag.documents = [make_doc(0.9, "t", {})]
    fake_rag.chunks = [make_chunk("partial")]
    fake_rag.stream_error = RuntimeError("boom")

    events = parse_sse(client.post(URL, json={"question": QUESTION}).text)

    assert names(events)[-1] == "error"
