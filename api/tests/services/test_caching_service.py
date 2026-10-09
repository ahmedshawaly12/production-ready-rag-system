import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.helpers.utils import serialize_documents
from api.services.caching_service import SemanticCacheService

SCOPE = dict(user_id="u1", session_id="s1", conversation_id="c1")


@pytest.fixture
def service():
    # bypass __init__ so no Redis connection is attempted
    svc = SemanticCacheService.__new__(SemanticCacheService)
    svc.cache = MagicMock()
    svc.cache.acheck = AsyncMock(return_value=[])
    svc.cache.astore = AsyncMock()
    svc.cache.adisconnect = AsyncMock()
    return svc


async def test_get_miss_returns_none(service):
    assert await service.get(vector=[0.1], **SCOPE) is None


async def test_get_hit_returns_response_and_documents(service):
    payload = {"response": "hi", "documents": [{"text": "t"}]}
    service.cache.acheck.return_value = [{"response": json.dumps(payload)}]

    assert await service.get(vector=[0.1], **SCOPE) == payload


async def test_get_sends_vector_and_scope_filter(service):
    await service.get(vector=[0.1, 0.2], **SCOPE)

    kwargs = service.cache.acheck.await_args.kwargs
    assert kwargs["vector"] == [0.1, 0.2]
    assert kwargs["num_results"] == 1
    rendered = str(kwargs["filter_expression"])
    assert "u1" in rendered and "s1" in rendered and "c1" in rendered


@pytest.mark.parametrize(
    "raw",
    ["not json", json.dumps({"response": "x"}), json.dumps(["a"]), "null"],
)
async def test_malformed_entries_are_treated_as_miss(service, raw):
    service.cache.acheck.return_value = [{"response": raw}]
    assert await service.get(vector=[0.1], **SCOPE) is None


async def test_set_stores_serialized_payload_with_scope(service, make_doc):
    docs = [make_doc(0.9, "text", {"source": "a.pdf", "page": 2})]

    await service.set(
        prompt="q", response="مرحبا", documents=docs, vector=[0.1], **SCOPE
    )

    kwargs = service.cache.astore.await_args.kwargs
    assert kwargs["prompt"] == "q"
    assert kwargs["vector"] == [0.1]
    assert kwargs["filters"] == SCOPE
    assert kwargs["ttl"] is None
    assert json.loads(kwargs["response"]) == {
        "response": "مرحبا",
        "documents": serialize_documents(docs),
    }


async def test_close_disconnects(service):
    await service.close()
    service.cache.adisconnect.assert_awaited_once()
