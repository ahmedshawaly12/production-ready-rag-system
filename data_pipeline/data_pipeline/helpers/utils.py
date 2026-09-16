import os

import boto3
from botocore.client import BaseClient

from data_pipeline.helpers.config import get_settings


def get_s3_client() -> BaseClient:
    settings = get_settings()
    s3_client = boto3.client(
        "s3",
        endpoint_url=settings.endpoint_url,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
    )

    return s3_client


def get_file_extention(key: str) -> str:
    return os.path.splitext(key)[-1].lower()


def list_bucket_folder_files(client: BaseClient, bucket: str, path: str):
    response = client.list_objects_v2(Bucket=bucket, Prefix=f"{path}/")
    contents = response.get("Contents")
    if contents is None:
        print(f"there is no file exists in {path}")
        return None

    return [obj["Key"] for obj in contents]


def move_file(client: BaseClient, bucket_name: str, source_key: str, dest_key: str):
    # Copy the object to the new location
    new_key = f"{dest_key}/{source_key.split('/')[-1]}"
    client.copy_object(
        Bucket=bucket_name,
        CopySource={"Bucket": bucket_name, "Key": source_key},
        Key=new_key,
    )
    # Delete the original
    client.delete_object(Bucket=bucket_name, Key=source_key)
    return new_key


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
