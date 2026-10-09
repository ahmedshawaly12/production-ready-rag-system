import logging
from collections.abc import Sequence
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client import AsyncQdrantClient, models

from shared.vectordb.schemas import RetrievedDocument
from shared.vectordb.vectordb_interface import VectorDBInterface

DISTANCE_METHODS = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
}


class VectorSearchError(RuntimeError):
    """Search failed (outage, missing collection, ...). Different from 'no results'."""


class QdrantProvider(VectorDBInterface):
    def __init__(self, provider_url: str, embedding_size: int, distance_method: str):
        self.provider_url = provider_url
        self.client: AsyncQdrantClient | None = None
        self.embedding_size = embedding_size
        self.logger = logging.getLogger(__name__)

        try:
            self.distance_method = DISTANCE_METHODS[distance_method.lower()]
        except KeyError:
            supported = ", ".join(DISTANCE_METHODS)
            raise ValueError(
                f"Unsupported distance method '{distance_method}'. "
                f"Supported: {supported}"
            ) from None

    def _require_client(self) -> AsyncQdrantClient:
        if self.client is None:
            raise ConnectionError(
                "Qdrant client is not connected. Call connect() first."
            )
        return self.client

    # Connection
    async def connect(self) -> None:
        client = AsyncQdrantClient(url=self.provider_url, check_compatibility=False)
        try:
            await client.get_collections()  # health check
        except Exception as e:
            await client.close()
            self.logger.error("Cannot connect to Qdrant vector DB: %s", e)
            raise ConnectionError(f"Cannot connect to Qdrant: {e}") from e

        self.client = client
        self.logger.info("Connected to Qdrant successfully")

    async def disconnect(self) -> None:
        if self.client is not None:
            await self.client.close()
        self.client = None

    # Collections
    async def is_collection_exist(self, collection_name: str) -> bool:
        return await self._require_client().collection_exists(
            collection_name=collection_name
        )

    async def list_all_collections(self):
        return await self._require_client().get_collections()

    async def delete_collection(self, collection_name: str) -> bool:
        if not await self.is_collection_exist(collection_name):
            self.logger.warning(f"Collection '{collection_name}' does not exist")
            return False

        await self._require_client().delete_collection(collection_name=collection_name)
        self.logger.info(f"Deleted collection '{collection_name}'")
        return True

    async def create_collection(
        self,
        collection_name: str,
        embedding_size: int | None = None,
        do_reset: bool = False,
    ) -> bool:
        if do_reset:
            await self.delete_collection(collection_name)

        if await self.is_collection_exist(collection_name):
            return False

        embedding_size = embedding_size or self.embedding_size

        await self._require_client().create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embedding_size, distance=self.distance_method
            ),
        )
        self.logger.info(f"Created collection '{collection_name}'")
        return True

    # Insert
    async def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict,
        record_id: str,
    ) -> bool:
        return await self.insert_many(
            collection_name=collection_name,
            texts=[text],
            vectors=[vector],
            metadata=[metadata],
            record_ids=[record_id],
            batch_size=1,
        )

    async def insert_many(
        self,
        collection_name: str,
        texts: Sequence[str],
        vectors: Sequence[list],
        metadata: Sequence[dict],
        record_ids: Sequence[str],
        batch_size: int = 150,
    ) -> bool:
        client = self._require_client()

        if not await self.is_collection_exist(collection_name):
            self.logger.warning(
                f"Cannot insert records: collection '{collection_name}' does not exist"
            )
            return False

        points = [
            models.PointStruct(
                id=str(uuid5(NAMESPACE_DNS, record_id)),
                vector=vector,
                payload={"text": text, "metadata": meta},
            )
            for record_id, text, vector, meta in zip(
                record_ids, texts, vectors, metadata, strict=True
            )
        ]

        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            try:
                await client.upsert(collection_name=collection_name, points=batch)
            except Exception as e:
                self.logger.error(
                    f"Failed to insert batch into collection '{collection_name}': {e}"
                )
                return False

        return True

    # Search
    async def search(
        self, collection_name: str, query_vector: list, top_k: int = 10
    ) -> list[RetrievedDocument]:
        client = self._require_client()

        # No collection_exists() pre-check: it costs an extra network round trip on every query,
        # and query_points already fails if the collection is missing.

        # if not await self.is_collection_exist(collection_name):
        #     self.logger.warning(
        #         f"Cannot search: collection '{collection_name}' does not exist"
        #     )
        #     return []

        try:
            response = await client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        except Exception as e:
            self.logger.exception(
                f"Failed to search collection '{collection_name}': {e}"
            )
            # search now raises VectorSearchError. The generic handler in chat.py catches it and sends an error event, and the traceback is logged.
            raise VectorSearchError(
                f"Search failed on collection '{collection_name}'"
            ) from e

        return [
            RetrievedDocument(
                score=point.score,
                text=(point.payload or {}).get("text"),
                metadata=(point.payload or {}).get("metadata"),
            )
            for point in response.points
        ]
