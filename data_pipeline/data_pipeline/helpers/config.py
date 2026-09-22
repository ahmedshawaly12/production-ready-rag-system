from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class DataPipelineSettings(BaseSettings):
    aws_access_key_id: str
    aws_secret_access_key: str
    minio_endpoint_url: str
    bucket_name: str
    file_allowed_types: list
    chunk_size: int = 250
    chunk_overlap: int = 50
    default_collection_name: str

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")


def get_settings():
    return DataPipelineSettings()
