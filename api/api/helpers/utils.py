from redis.asyncio import Redis

from api.helpers.config import get_settings


def get_redis_client() -> Redis:
    settings = get_settings()
    return Redis(
        host=settings.REDIS_HOST, port=settings.REDIS_PORT, decode_responses=True
    )
