from botocore.client import BaseClient
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger
from pydantic import BaseModel, Field

from data_pipeline.helpers.config import get_settings
from data_pipeline.helpers.utils import get_s3_client, list_files_to_extract, move_file
from data_pipeline.parse.tasks import (
    extract_text,
    save_extracted_text,
)
from data_pipeline.vectorize.chunkers import chunck_pages
from data_pipeline.vectorize.tasks import get_vector_db_provider, index_into_vectordb


class PipelineConfigsSchema(BaseModel):
    bucket: str = Field(default_factory=lambda: get_settings().bucket_name)
    raw_documents_path: str = "raw"
    extracted_text_dest_path: str = "extracted"
    processed_files_dest_path: str = "processed"
    unprocessed_files_dest_path: str = "unprocessed"
    do_reset: bool = False
    collection_name: str = Field(
        default_factory=lambda: get_settings().default_collection_name
    )


@task(name="process_file", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def process_file(
    client: BaseClient,
    bucket: str,
    key: str,
    vectordb_client,
    file_allowed_types: list,
    configs: PipelineConfigsSchema,
):
    logger = get_run_logger()

    logger.info(f"======== Processing file: {key} ========")
    try:
        # 1- extract text
        pages = await extract_text(client, bucket, key, file_allowed_types)
        if not pages:
            # logger.warning(
            #     f"No extracted content for {key}. Moving file to unprocessed."
            # )
            await move_file(client, bucket, key, configs.unprocessed_files_dest_path)
            return None

        logger.info(f"Extracted {len(pages)} page(s) from {key}")

        # 2- save extracted text
        extracted_key = await save_extracted_text(
            client,
            bucket,
            key,
            pages,
            destination_path=configs.extracted_text_dest_path,
        )

        # 3- chunck pages
        chuncks = await chunck_pages(pages, key)

        if not chuncks:
            logger.warning(
                f"No chunks generated for {key}. Moving file to unprocessed."
            )
            await move_file(client, bucket, key, configs.unprocessed_files_dest_path)
            return None

        # 4- Generate embedding + index into vectordb
        await index_into_vectordb(vectordb_client, configs.collection_name, chuncks)

        # 5- move the file to processed
        await move_file(client, bucket, key, configs.processed_files_dest_path)

        logger.info(
            f"=== File processed successfully: key={key} | pages={len(pages)} | chunks={len(chuncks)} ==="
        )
        return extracted_key

    except Exception:
        logger.exception(f"=== Failed to process file: {key} ===")
        try:
            await move_file(client, bucket, key, configs.unprocessed_files_dest_path)
        except Exception:
            logger.exception(f"Failed to move {key} to unprocessed")


@flow(name="rag-data-ingestion-pipeline", log_prints=False)
async def rag_data_ingestion_flow(configs: PipelineConfigsSchema):
    logger = get_run_logger()
    logger.info("========== Starting document extraction pipeline ==========")

    s3_client = get_s3_client()
    vectordb_client = await get_vector_db_provider()
    logger.info("s3 client and vectordb client initialized")

    file_allowed_types = get_settings().file_allowed_types

    if configs.do_reset or not await vectordb_client.is_collection_exist(
        configs.collection_name
    ):
        await vectordb_client.create_collection(
            collection_name=configs.collection_name, do_reset=configs.do_reset
        )
        logger.info(f"create a new collection: {configs.collection_name}")

    # 1- list files
    files = await list_files_to_extract(
        s3_client, configs.bucket, configs.raw_documents_path
    )

    if not files:
        await vectordb_client.disconnect()
        return []

    logger.info(f"Starting processing of {len(files)} file(s)")

    # 2- process files
    results = []

    for key in files:
        result = await process_file(
            s3_client,
            configs.bucket,
            key,
            vectordb_client,
            file_allowed_types,
            configs,
        )
        results.append(result)
    # futures = [
    #     process_file.submit(
    #         s3_client,
    #         configs.bucket,
    #         key,
    #         vectordb_client,
    #         file_allowed_types,
    #         configs,
    #     )
    #     for key in files
    # ]
    # results = [f.result() for f in futures]

    # 3- summary
    successful = sum(result is not None for result in results)

    await vectordb_client.disconnect()

    logger.info("========== Pipeline completed ==========")
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Successfully processed: {successful}")
    logger.info(f"Skipped: {len(files) - successful}")

    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(rag_data_ingestion_flow())
