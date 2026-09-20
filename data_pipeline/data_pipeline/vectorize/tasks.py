import json

from botocore.client import BaseClient
from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger
from shared.llm.embedding_model import get_embedding_model
from shared.vectordb.vectordb_provider_facotry import VectorDBProviderFactory

from data_pipeline.helpers.utils import download_bytes


async def get_vector_db_provider():
    vector_db_provider_factory = VectorDBProviderFactory()
    return await vector_db_provider_factory.create()


@task(name="extract_pages", retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def extract_pages(client: BaseClient, bucket: str, key: str):
    logger = get_run_logger()

    logger.info(f"Start Extraction from: {bucket}/{key}")
    data = download_bytes(client, bucket, key)
    logger.info(f"Data Loaded Successfully from: {bucket}/{key}")

    pages = json.loads(data.decode("utf-8", errors="ignore"))
    logger.info(f"Extracted {len(pages['pages'])} pages from: {bucket}/{key}")

    return pages["pages"], pages["source"]


@task(
    name="index_into_vectordb", retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE
)
async def index_into_vectordb(
    vector_db_provider,
    collection_name: str,
    chuncks: list[dict],
):
    logger = get_run_logger()

    if not chuncks:
        logger.warning(f"No chuncks to index into collection: {collection_name}")
        return 0

    logger.info(
        f"Starting vector indexing: {len(chuncks)} chuncks -> {collection_name}"
    )

    texts = [c["text"] for c in chuncks]
    metadata = [c["metadata"] for c in chuncks]

    logger.info(f"Generating embeddingds for {len(texts)} chuncks")

    vectors = get_embedding_model().embed_text(texts)
    record_ids = [
        f"page_{c['metadata']['page']}_id_{c['metadata']['chunck_id']}" for c in chuncks
    ]

    logger.info(f"Inserting {len(record_ids)} vectors into {collection_name}")

    await vector_db_provider.insert_many(
        collection_name=collection_name,
        texts=texts,
        vectors=vectors,
        metadata=metadata,
        record_ids=record_ids,
        batch_size=400,
    )

    logger.info(
        f"Successfully indexed {len(record_ids)} chunks into '{collection_name}'"
    )

    return len(record_ids)
