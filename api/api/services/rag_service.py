from collections.abc import AsyncIterator
from dataclasses import dataclass

from shared.vectordb.schemas import RetrievedDocument

from api.helpers.config import get_settings


@dataclass
class QueryEmbedding:
    vector: list[float]  # empty list means embedding failed
    model: str | None = None  # real model that served the request
    prompt_tokens: int = 0


class RAGService:
    def __init__(self, vectordb_client, embedding_client, generation_client):
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.generation_client = generation_client
        self.app_settings = get_settings()

    async def embed(self, text: str) -> QueryEmbedding:
        result = await self.embedding_client.embed_text_with_meta(text)
        if not result or not result.vectors:
            return QueryEmbedding(vector=[])

        return QueryEmbedding(
            vector=result.vectors[0],
            model=result.model,
            prompt_tokens=result.prompt_tokens,
        )

    async def retrieve(self, vector: list[float]) -> list[RetrievedDocument]:
        return await self.vectordb_client.search(
            collection_name=self.app_settings.USED_COLLECTION_NAME,
            query_vector=vector,
            top_k=self.app_settings.TOP_K,
        )

    async def generate(
        self,
        messages: list[dict],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        return await self.generation_client.generate_text(
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    async def stream_generate(
        self,
        messages: list[dict],
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator:
        async for chunk in self.generation_client.stream_generate_text(
            messages=messages,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ):
            yield chunk
