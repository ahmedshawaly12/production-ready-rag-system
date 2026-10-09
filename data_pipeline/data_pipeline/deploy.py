import asyncio

from data_pipeline.data_ingestion_pipeline import rag_data_ingestion_flow
from data_pipeline.helpers.config import get_settings
from data_pipeline.vectorize.reindex_flow import reindex_flow


async def create_deployment():
    settings = get_settings()

    flow = await rag_data_ingestion_flow.afrom_source(
        source=".",
        entrypoint="data_pipeline/data_ingestion_pipeline.py:rag_data_ingestion_flow",
    )

    await flow.deploy(
        name="rag-data-ingestion",
        work_pool_name="default-agent-pool",
        parameters={
            "configs": {
                "bucket": settings.bucket_name,
                "collection_name": settings.default_collection_name,
                "raw_documents_path": "raw",
                "extracted_text_dest_path": "extracted",
                "processed_files_dest_path": "processed",
                "unprocessed_files_dest_path": "unprocessed",
                "do_reset": False,
            }
        },
    )

    flow2 = await reindex_flow.afrom_source(
        source=".",
        entrypoint="data_pipeline/vectorize/reindex_flow.py:reindex_flow",
    )

    await flow2.deploy(
        name="reindex-pipeline",
        work_pool_name="default-agent-pool",
        parameters={
            "configs": {
                "bucket": settings.bucket_name,
                "collection_name": settings.default_collection_name,
                "extracted_text_dest_path": "extracted",
                "failed_index_files_dest_path": "failed_index",
                "do_reset": True,
            }
        },
    )


if __name__ == "__main__":
    asyncio.run(create_deployment())
