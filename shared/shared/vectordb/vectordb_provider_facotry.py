from shared.config import get_settings
from shared.vectordb.providers.qdrant_provider import QdrantProvider


class VectorDBProviderFactory:
    def __init__(
        self, embedding_size: str | None = None, distance_method: str | None = None
    ):
        self.settings = get_settings()
        self.embedding_size = embedding_size or self.settings.EMBEDDING_SIZE
        self.distance_method = distance_method or self.settings.VECTORDB_DISTANCE_METHOD

    async def create(self):
        if self.settings.VECTORDB_PROVDER_NAME.lower() == "qdrant":
            provider = QdrantProvider(
                provider_url=self.settings.VECTORDB_PROVDER_URL,
                distance_method=self.distance_method,
                embedding_size=self.embedding_size,
            )
            await provider.connect()
            return provider
