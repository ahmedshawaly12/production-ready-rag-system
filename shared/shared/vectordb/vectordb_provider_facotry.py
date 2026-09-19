from shared.config import get_settings
from shared.vectordb.providers.qdrant_provider import QdrantProvider


class VectorDBProviderFactory:
    def __init__(self):
        self.settings = get_settings()

    def create(self):
        if self.settings.VECTORDB_PROVDER_NAME.lower() == "qdrant":
            return QdrantProvider(
                provider_url=self.settings.VECTORDB_PROVDER_URL,
                distance_method=self.settings.VECTORDB_DISTANCE_METHOD,
                embedding_size=self.settings.EMBEDDING_SIZE,
            )
