from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.services import rag_service as rag_module
from api.services.rag_service import QueryEmbedding, RAGService


@pytest.fixture
def clients():
    return SimpleNamespace(
        vectordb=MagicMock(), embedding=MagicMock(), generation=MagicMock()
    )


@pytest.fixture
def service(monkeypatch, settings, clients):
    monkeypatch.setattr(rag_module, "get_settings", lambda: settings)
    return RAGService(
        vectordb_client=clients.vectordb,
        embedding_client=clients.embedding,
        generation_client=clients.generation,
    )


async def test_embed_returns_first_vector_with_metadata(service, clients):
    clients.embedding.embed_text_with_meta = AsyncMock(
        return_value=SimpleNamespace(vectors=[[0.1, 0.2]], model="m", prompt_tokens=7)
    )

    result = await service.embed("hello")

    assert result == QueryEmbedding(vector=[0.1, 0.2], model="m", prompt_tokens=7)
    clients.embedding.embed_text_with_meta.assert_awaited_once_with("hello")


@pytest.mark.parametrize(
    "bad", [None, SimpleNamespace(vectors=[], model="m", prompt_tokens=0)]
)
async def test_embed_failure_gives_empty_vector(service, clients, bad):
    clients.embedding.embed_text_with_meta = AsyncMock(return_value=bad)

    assert (await service.embed("hello")).vector == []


async def test_retrieve_uses_collection_and_top_k_from_settings(
    service, clients, settings
):
    clients.vectordb.search = AsyncMock(return_value=["doc"])

    assert await service.retrieve([0.1]) == ["doc"]
    clients.vectordb.search.assert_awaited_once_with(
        collection_name=settings.USED_COLLECTION_NAME,
        query_vector=[0.1],
        top_k=settings.TOP_K,
    )


async def test_generate_passes_arguments_through(service, clients):
    clients.generation.generate_text = AsyncMock(return_value="text")

    out = await service.generate(
        messages=[{"role": "user", "content": "q"}],
        max_output_tokens=50,
        temperature=0.2,
    )

    assert out == "text"
    clients.generation.generate_text.assert_awaited_once_with(
        messages=[{"role": "user", "content": "q"}],
        max_output_tokens=50,
        temperature=0.2,
    )


async def test_stream_generate_yields_chunks_in_order(service, clients):
    seen = {}

    async def fake_stream(**kwargs):
        seen.update(kwargs)
        for c in ("a", "b", "c"):
            yield c

    clients.generation.stream_generate_text = fake_stream

    chunks = [c async for c in service.stream_generate(messages=["m"], temperature=0.5)]

    assert chunks == ["a", "b", "c"]
    assert seen == {"messages": ["m"], "max_output_tokens": None, "temperature": 0.5}
