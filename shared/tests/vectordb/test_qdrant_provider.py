from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import NAMESPACE_DNS, uuid5

import pytest
from qdrant_client import AsyncQdrantClient, models

from shared.vectordb.providers import qdrant_provider as mod
from shared.vectordb.providers.qdrant_provider import (
    QdrantProvider,
    VectorSearchError,
)


@pytest.fixture
def provider():
    """Provider wired to a mocked client."""
    p = QdrantProvider(
        "http://localhost:6333", embedding_size=3, distance_method="cosine"
    )
    p.client = AsyncMock()
    return p


# ---------- init ----------
@pytest.mark.parametrize(
    "name,expected",
    [("cosine", models.Distance.COSINE), ("DOT", models.Distance.DOT)],
)
def test_distance_method_is_case_insensitive(name, expected):
    assert QdrantProvider("u", 3, name).distance_method == expected


def test_unsupported_distance_method_raises():
    with pytest.raises(ValueError, match="Unsupported distance method 'euclid'"):
        QdrantProvider("u", 3, "euclid")


def test_require_client_before_connect():
    p = QdrantProvider("u", 3, "cosine")
    with pytest.raises(ConnectionError, match="connect"):
        p._require_client()


# ---------- connect / disconnect ----------
async def test_connect_success(monkeypatch):
    fake = AsyncMock()
    monkeypatch.setattr(mod, "AsyncQdrantClient", MagicMock(return_value=fake))
    p = QdrantProvider("http://x", 3, "cosine")

    await p.connect()

    assert p.client is fake
    fake.get_collections.assert_awaited_once()


async def test_connect_failure_closes_client_and_raises(monkeypatch):
    fake = AsyncMock()
    fake.get_collections.side_effect = RuntimeError("down")
    monkeypatch.setattr(mod, "AsyncQdrantClient", MagicMock(return_value=fake))
    p = QdrantProvider("http://x", 3, "cosine")

    with pytest.raises(ConnectionError, match="Cannot connect"):
        await p.connect()

    fake.close.assert_awaited_once()
    assert p.client is None


async def test_disconnect_closes_and_clears(provider):
    client = provider.client
    await provider.disconnect()
    client.close.assert_awaited_once()
    assert provider.client is None


async def test_disconnect_when_not_connected_is_noop():
    await QdrantProvider("u", 3, "cosine").disconnect()


# ---------- collections ----------
async def test_delete_collection_missing_returns_false(provider):
    provider.client.collection_exists.return_value = False
    assert await provider.delete_collection("c") is False
    provider.client.delete_collection.assert_not_awaited()


async def test_delete_collection_existing(provider):
    provider.client.collection_exists.return_value = True
    assert await provider.delete_collection("c") is True
    provider.client.delete_collection.assert_awaited_once_with(collection_name="c")


async def test_create_collection_new_uses_default_size(provider):
    provider.client.collection_exists.return_value = False

    assert await provider.create_collection("c") is True

    cfg = provider.client.create_collection.call_args.kwargs["vectors_config"]
    assert cfg.size == 3
    assert cfg.distance == models.Distance.COSINE


async def test_create_collection_custom_size(provider):
    provider.client.collection_exists.return_value = False
    await provider.create_collection("c", embedding_size=768)
    assert (
        provider.client.create_collection.call_args.kwargs["vectors_config"].size == 768
    )


async def test_create_collection_already_exists_returns_false(provider):
    provider.client.collection_exists.return_value = True
    assert await provider.create_collection("c") is False
    provider.client.create_collection.assert_not_awaited()


async def test_create_collection_do_reset_deletes_first(provider):
    # exists for the delete check, gone for the create check
    provider.client.collection_exists.side_effect = [True, False]

    assert await provider.create_collection("c", do_reset=True) is True
    provider.client.delete_collection.assert_awaited_once()
    provider.client.create_collection.assert_awaited_once()


# ---------- insert ----------
async def test_insert_many_missing_collection_returns_false(provider):
    provider.client.collection_exists.return_value = False
    ok = await provider.insert_many("c", ["t"], [[0.1] * 3], [{}], ["r1"])
    assert ok is False
    provider.client.upsert.assert_not_awaited()


async def test_insert_many_builds_points_with_deterministic_ids(provider):
    provider.client.collection_exists.return_value = True

    ok = await provider.insert_many(
        "c", ["text"], [[0.1, 0.2, 0.3]], [{"page": 1}], ["rec-1"]
    )

    assert ok is True
    (point,) = provider.client.upsert.call_args.kwargs["points"]
    assert point.id == str(uuid5(NAMESPACE_DNS, "rec-1"))
    assert point.payload == {"text": "text", "metadata": {"page": 1}}


async def test_insert_many_splits_into_batches(provider):
    provider.client.collection_exists.return_value = True
    n = 5

    await provider.insert_many(
        "c",
        [f"t{i}" for i in range(n)],
        [[0.0] * 3] * n,
        [{}] * n,
        [f"r{i}" for i in range(n)],
        batch_size=2,
    )

    sizes = [len(c.kwargs["points"]) for c in provider.client.upsert.call_args_list]
    assert sizes == [2, 2, 1]


async def test_insert_many_returns_false_when_upsert_fails(provider):
    provider.client.collection_exists.return_value = True
    provider.client.upsert.side_effect = RuntimeError("boom")
    assert await provider.insert_many("c", ["t"], [[0.0] * 3], [{}], ["r"]) is False


async def test_insert_many_length_mismatch_raises(provider):
    provider.client.collection_exists.return_value = True
    with pytest.raises(ValueError):  # zip(strict=True)
        await provider.insert_many("c", ["a", "b"], [[0.0] * 3], [{}], ["r1"])


async def test_insert_one_delegates_to_insert_many(provider):
    provider.insert_many = AsyncMock(return_value=True)
    assert await provider.insert_one("c", "t", [0.1], {"m": 1}, "r") is True
    provider.insert_many.assert_awaited_once_with(
        collection_name="c",
        texts=["t"],
        vectors=[[0.1]],
        metadata=[{"m": 1}],
        record_ids=["r"],
        batch_size=1,
    )


# ---------- search (mocked) ----------
async def test_search_maps_points_to_retrieved_documents(provider):
    provider.client.query_points.return_value = SimpleNamespace(
        points=[
            SimpleNamespace(score=0.9, payload={"text": "a", "metadata": {"k": 1}}),
            SimpleNamespace(score=0.5, payload={"text": "b", "metadata": None}),
        ]
    )

    docs = await provider.search("c", [0.1, 0.2, 0.3], top_k=2)

    assert [(d.score, d.text, d.metadata) for d in docs] == [
        (0.9, "a", {"k": 1}),
        (0.5, "b", None),
    ]
    provider.client.query_points.assert_awaited_once_with(
        collection_name="c", query=[0.1, 0.2, 0.3], limit=2, with_payload=True
    )


async def test_search_no_hits_returns_empty_list(provider):
    provider.client.query_points.return_value = SimpleNamespace(points=[])
    assert await provider.search("c", [0.0] * 3) == []


async def test_search_failure_raises_vector_search_error(provider):
    provider.client.query_points.side_effect = RuntimeError("collection missing")

    with pytest.raises(VectorSearchError, match="'c'") as exc:
        await provider.search("c", [0.0] * 3)

    assert isinstance(exc.value.__cause__, RuntimeError)  # original error preserved


# ---------- integration-style, using Qdrant's in-process :memory: mode ----------
@pytest.fixture
async def real_provider():
    p = QdrantProvider("http://unused", embedding_size=3, distance_method="cosine")
    p.client = AsyncQdrantClient(":memory:")
    yield p
    await p.disconnect()


async def test_roundtrip_insert_and_search(real_provider):
    p = real_provider
    assert await p.create_collection("docs") is True
    await p.insert_many(
        "docs",
        ["x-axis", "y-axis", "z-axis"],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [{"i": 0}, {"i": 1}, {"i": 2}],
        ["a", "b", "c"],
    )

    docs = await p.search("docs", [0.9, 0.1, 0], top_k=2)

    assert [d.text for d in docs] == ["x-axis", "y-axis"]
    assert docs[0].metadata == {"i": 0}
    assert docs[0].score > docs[1].score


async def test_reinserting_same_record_id_overwrites(real_provider):
    p = real_provider
    await p.create_collection("docs")
    await p.insert_one("docs", "old", [1, 0, 0], {}, "same-id")
    await p.insert_one("docs", "new", [1, 0, 0], {}, "same-id")

    docs = await p.search("docs", [1, 0, 0], top_k=10)
    assert [d.text for d in docs] == ["new"]


async def test_search_missing_collection_raises(real_provider):
    with pytest.raises(VectorSearchError):
        await real_provider.search("nope", [0.0] * 3)


async def test_create_collection_reset_drops_data(real_provider):
    p = real_provider
    await p.create_collection("docs")
    await p.insert_one("docs", "t", [1, 0, 0], {}, "r")

    await p.create_collection("docs", do_reset=True)

    assert await p.search("docs", [1, 0, 0]) == []
