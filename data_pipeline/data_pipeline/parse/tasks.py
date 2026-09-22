import json
from pathlib import PurePosixPath

from botocore.client import BaseClient
from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from data_pipeline.helpers.utils import (
    download_bytes,
    get_file_extention,
    upload_bytes,
)
from data_pipeline.parse.extractors import extract_pdf_text


@task(name="extract_text", retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def extract_text(
    client: BaseClient, bucket: str, key: str, file_allowed_types: list
):
    logger = get_run_logger()
    file_extention = get_file_extention(key)

    logger.info(f"Start text extraction from {bucket}/{key}")

    if file_extention not in file_allowed_types:
        logger.warning(f"unsupported file format {bucket}/{key}")
        # move_file(client, bucket, key, "unsupported")
        # logger.info(f"Moved unsupported file s3://{bucket}/{key}")
        return []

    data = download_bytes(client, bucket, key)
    logger.info(f"Downloaded {key} ({len(data)} bytes)")

    pages = []
    if file_extention == ".txt":
        pages = [{"text": data.decode("utf-8", errors="ignore"), "page": 1}]

    elif file_extention == ".pdf":
        pages_text = extract_pdf_text(data)
        pages = [
            {"page": page_num, "text": text}
            for page_num, text in enumerate(pages_text, start=1)
        ]

    else:
        logger.warning(f"No extractor implemented for {file_extention}: {key}")
        return []

    logger.info(f"Successfully extracted {len(pages)} pages from {key}")

    # move_file(settings.bucket_name, file_path, "processed")
    return pages


def get_extracted_s3_key(key: str, destination_path: str) -> str:
    path = PurePosixPath(key)
    filename = path.stem + ".json"
    return f"{destination_path}/{filename}"


@task(
    name="save_extracted_text", retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE
)
async def save_extracted_text(
    client: BaseClient,
    bucket: str,
    source_key: str,
    pages: list[dict],
    destination_path: str,
):
    logger = get_run_logger()
    destination_key = get_extracted_s3_key(source_key, destination_path)

    payload = {"source": source_key, "pages": pages}

    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")

    upload_bytes(client, bucket, destination_key, data, content_type="application/json")
    logger.info(f"Save Extracted Text: {bucket}/{destination_key}")
    return destination_key
