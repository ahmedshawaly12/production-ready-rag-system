import os
from pathlib import PurePosixPath

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError
from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.logging import get_run_logger

from data_pipeline.helpers.config import get_settings


def get_s3_client() -> BaseClient:
    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.minio_endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    return s3_client


def get_file_extention(key: str) -> str:
    return os.path.splitext(key)[-1].lower()


def download_bytes(client: BaseClient, bucket: str, key: str) -> bytes:
    obj = client.get_object(Bucket=bucket, Key=key)
    return obj["Body"].read()


def upload_bytes(
    client: BaseClient,
    bucket: str,
    key: str,
    data: bytes,
    content_type: str = "application/octet-stream",
) -> None:
    client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def _list_bucket_folder_files(client: BaseClient, bucket: str, path: str) -> list[str]:
    logger = get_run_logger()

    try:
        response = client.list_objects_v2(Bucket=bucket, Prefix=f"{path}/")

    except ClientError as e:
        error_code = e.response["Error"]["Code"]

        if error_code == "NoSuchBucket":
            logger.warning(f"bucket does not exist: {bucket}")
            return []
        raise

    contents = response.get("Contents", [])
    return [obj["Key"] for obj in contents]


@task(
    name="list_files_to_extract",
    retries=3,
    retry_delay_seconds=5,
    cache_policy=NO_CACHE,
)
async def list_files_to_extract(
    client: BaseClient, bucket: str, path: str
) -> list[str]:
    logger = get_run_logger()
    logger.info(f"Listing files from {bucket}/{path}")
    files = _list_bucket_folder_files(client, bucket, path)
    if not files:
        logger.info(f"No files found in {bucket}/{path}")
    return files


@task(name="move_file", retries=3, retry_delay_seconds=5, cache_policy=NO_CACHE)
async def move_file(client: BaseClient, bucket: str, source_key: str, destination: str):
    logger = get_run_logger()

    file_name = PurePosixPath(source_key).name
    new_key = f"{destination}/{file_name}"

    logger.info(f"Moving {bucket}/{source_key} to {bucket}/{new_key}")

    client.copy_object(
        Bucket=bucket,
        CopySource={"Bucket": bucket, "Key": source_key},
        Key=new_key,
    )

    # Delete the original
    client.delete_object(Bucket=bucket, Key=source_key)
    logger.info(f"Successfully moved file {bucket}/{source_key} to {bucket}/{new_key}")
    return new_key
