from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    USED_COLLECTION_NAME: str
    TOP_K: int

    GENERATION_MODEL_PROVIDER_URL: str
    GENERATION_MODEL_ID: str
    GENERATION_MODEL_API_KEY: str
    MAX_INPUT_CHARACTERS: int
    MAX_OUTPUT_TOKENS: int
    TEMPERATURE: float

    EMBEDDING_MODEL_SIZE: int

    REDIS_HOST: str
    REDIS_PORT: int
    SEMANTIC_CACHE_SEARCH_INDEX_NAME: str
    CACHE_DISTANCE_THRESHOLD: float
    CACHE_TTL: int

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env")


def get_settings():
    return Settings()
