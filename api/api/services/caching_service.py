import json
import logging

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.query.filter import Tag
from redisvl.utils.vectorize import CustomTextVectorizer

from api.helpers.config import get_settings
from api.helpers.utils import serialize_documents

logger = logging.getLogger(__name__)


_SCOPE_FIELDS = ("user_id", "session_id", "conversation_id")


class SemanticCacheService:
    def __init__(self):
        settings = get_settings()
        redis_url = f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}"

        dims = settings.EMBEDDING_MODEL_SIZE

        # It only tells redisvl the vector dimensions. so this vectorizer never embeds anything.
        def _placeholder_embed(text: str, **kwargs) -> list[float]:
            return [0.0] * dims

        vectorizer = CustomTextVectorizer(embed=_placeholder_embed)

        self.cache = SemanticCache(
            name=settings.SEMANTIC_CACHE_SEARCH_INDEX_NAME,
            redis_url=redis_url,
            distance_threshold=settings.CACHE_DISTANCE_THRESHOLD,
            ttl=settings.CACHE_TTL,
            vectorizer=vectorizer,
            filterable_fields=[{"name": f, "type": "tag"} for f in _SCOPE_FIELDS],
        )

    @staticmethod
    def _build_filter(user_id: str, session_id: str, conversation_id: str):
        return (
            (Tag("user_id") == user_id)
            & (Tag("session_id") == session_id)
            & (Tag("conversation_id") == conversation_id)
        )

    async def get(
        self, vector: list[float], user_id: str, session_id: str, conversation_id: str
    ) -> dict | None:
        filter_expression = self._build_filter(user_id, session_id, conversation_id)

        results = await self.cache.acheck(
            vector=vector, num_results=1, filter_expression=filter_expression
        )

        if not results:
            return None

        try:
            payload = json.loads(results[0]["response"])

            return {
                "response": payload["response"],
                "documents": payload["documents"],
            }

        except (json.JSONDecodeError, KeyError, TypeError):
            # Entry written in an older format: treat it as a miss.
            logger.warning("Ignoring malformed semantic cache entry", exc_info=True)
            return None

    async def set(
        self,
        prompt: str,
        response: str,
        documents: list,
        vector: list[float],
        user_id: str,
        session_id: str,
        conversation_id: str,
        ttl: int | None = None,
    ):
        payload = {
            "response": response,
            "documents": serialize_documents(documents),
        }

        await self.cache.astore(
            prompt=prompt,
            response=json.dumps(payload, ensure_ascii=False),
            vector=vector,
            filters={
                "user_id": user_id,
                "session_id": session_id,
                "conversation_id": conversation_id,
            },
            ttl=ttl,
        )

    async def close(self) -> None:
        await self.cache.adisconnect()
