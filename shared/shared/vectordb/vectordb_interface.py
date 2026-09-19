from abc import ABC, abstractmethod

from shared.vectordb.schemas import RetrievedDocument


class VectorDBInterface(ABC):
    @abstractmethod
    def connect(self):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    @abstractmethod
    def is_collection_exist(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def create_collection(
        self, collection_name: str, embedding_size: int, distance_method
    ):
        pass

    @abstractmethod
    def list_all_collections(self):
        pass

    @abstractmethod
    def insert_one(
        self,
        collection_name: str,
        text: str,
        vector: list,
        metadata: dict,
        record_id: str,
    ):
        pass

    @abstractmethod
    def insert_many(
        self,
        collection_name: str,
        texts: list,
        vectors: list,
        metadata: list,
        record_ids: list,
        batch_size: int = 50,
    ):
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: list,
        top_k: int,
    ) -> RetrievedDocument:
        pass
