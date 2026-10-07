import json

from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import CustomTextVectorizer

from api.helpers.config import get_settings


class SemanticCacheService:
    def __init__(self):
        self.app_settings = get_settings()
        redis_url = (
            f"redis://{self.app_settings.REDIS_HOST}:{self.app_settings.REDIS_PORT}"
        )

        dims = self.app_settings.EMBEDDING_MODEL_SIZE

        def _placeholder_embed(text: str, **kwargs) -> list[float]:
            return [0.0] * dims

        vectorizer = CustomTextVectorizer(embed=_placeholder_embed)

        self.cache = SemanticCache(
            name=self.app_settings.SEMANTIC_CACHE_SEARCH_INDEX_NAME,
            redis_url=redis_url,
            distance_threshold=self.app_settings.CACHE_DISTANCE_THRESHOLD,
            ttl=self.app_settings.CACHE_TTL,
            vectorizer=vectorizer,
        )

    async def get(self, vector: list[float]):
        results = await self.cache.acheck(vector=vector, num_results=1)

        if not results:
            return None

        cached_data = json.loads(results[0]["response"])

        return {
            "response": cached_data["response"],
            "documents": cached_data["documents"],
        }

    async def set(
        self,
        prompt: str,
        response: str,
        documents: list,
        vector: list[float],
        ttl: int | None = None,
    ):
        documents_data = [
            {
                "score": doc.score,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        cache_data = {
            "response": response,
            "documents": documents_data,
        }

        await self.cache.astore(
            prompt=prompt, response=json.dumps(cache_data), vector=vector, ttl=ttl
        )
