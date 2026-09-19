from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class SharedSettings(BaseSettings):
    VECTORDB_PROVDER_NAME: str
    VECTORDB_PROVDER_URL: str
    VECTORDB_DISTANCE_METHOD: str

    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_API_KEY: str
    EMBEDDING_SIZE: int
    EMBEDDING_MODEL_PROVIDER_URL: str | None = None

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")


def get_settings():
    return SharedSettings()
