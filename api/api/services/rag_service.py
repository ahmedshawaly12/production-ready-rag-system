from collections.abc import AsyncGenerator

from shared.vectordb.schemas import RetrievedDocument

from api.helpers.config import get_settings


class RAGService:
    def __init__(self, vectordb_client, embedding_client, generation_client):
        self.vectordb_client = vectordb_client
        self.embedding_client = embedding_client
        self.generation_client = generation_client
        self.app_settings = get_settings()

    async def embed(self, text: str) -> list[float]:
        embeded_text = await self.embedding_client.embed_text(text)
        return embeded_text[0]

    async def retrieve(self, vector: list[float]) -> list[RetrievedDocument] | None:
        return await self.vectordb_client.search(
            collection_name=self.app_settings.USED_COLLECTION_NAME,
            query_vector=vector,
            top_k=self.app_settings.TOP_K,
        )

    async def generate(
        self,
        prompt: str,
        chat_history: list,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        return await self.generation_client.generate_text(
            prompt=prompt,
            chat_history=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )

    async def stream_generate(
        self,
        prompt: str,
        chat_history: list,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        async for chunk in self.generation_client.stream_generate_text(
            prompt=prompt,
            chat_history=chat_history,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        ):
            yield chunk
