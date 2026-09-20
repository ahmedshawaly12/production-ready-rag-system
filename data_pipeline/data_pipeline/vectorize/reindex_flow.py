import asyncio

from prefect import flow
from prefect.logging import get_run_logger
from pydantic import BaseModel, Field

from data_pipeline.helpers.config import get_settings
from data_pipeline.helpers.utils import get_s3_client, list_files_to_extract, move_file
from data_pipeline.vectorize.chunkers import chunck_pages
from data_pipeline.vectorize.tasks import (
    extract_pages,
    get_vector_db_provider,
    index_into_vectordb,
)


class ReindexFlowConfigsSchema(BaseModel):
    bucket: str = Field(default_factory=lambda: get_settings().bucket_name)
    extracted_text_dest_path: str = "extracted"
    failed_index_files_dest_path: str = "failed_index"
    do_reset: bool = True
    collection_name: str = Field(
        default_factory=lambda: get_settings().default_collection_name
    )


@flow(name="reindex-pipeline", log_prints=False)
async def reindex_flow(configs: ReindexFlowConfigsSchema):
    logger = get_run_logger()

    logger.info("========== Starting Reindex pipeline ==========")

    vectordb_client = await get_vector_db_provider()
    s3_client = get_s3_client()
    logger.info("s3 client and vectordb client initialized")

    await vectordb_client.create_collection(
        collection_name=configs.collection_name, do_reset=configs.do_reset
    )

    # 1- list files
    files = await list_files_to_extract(
        s3_client, configs.bucket, configs.extracted_text_dest_path
    )

    if not files:
        await vectordb_client.disconnect()
        return []

    logger.info(f"Starting processing of {len(files)} extracted documents")

    results = []
    for key in files:
        logger.info(f"======== Processing document: {key} ========")
        try:
            # 2- Extract docs
            pages, source = await extract_pages(s3_client, configs.bucket, key)

            # 3- chuncking
            chuncks = await chunck_pages(pages, source)

            # 4- embedding + indexing
            _ = await index_into_vectordb(
                vectordb_client,
                collection_name=configs.collection_name,
                chuncks=chuncks,
            )
            results.append(key)

            logger.info(
                f"=== File processed successfully: key={key} | pages={len(pages)} | chunks={len(chuncks)} ==="
            )

        except Exception:
            logger.exception(f"=== Failed to process file: {key} ===")
            results.append(None)
            try:
                await move_file(
                    s3_client, configs.bucket, key, configs.failed_index_files_dest_path
                )

            except Exception:
                logger.exception(f"Failed to move {key} to unprocessed")

    successful = sum(result is not None for result in results)

    await vectordb_client.disconnect()

    logger.info("========== Pipeline completed ==========")
    logger.info(f"Total documents: {len(files)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Failed: {len(files) - successful}")


if __name__ == "__main__":
    asyncio.run(reindex_flow(ReindexFlowConfigsSchema()))
