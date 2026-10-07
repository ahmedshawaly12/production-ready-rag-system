from contextlib import asynccontextmanager

from fastapi import FastAPI
from shared.llm.embedding_model import get_embedding_model
from shared.llm.openai_provider import OpenAIProvider
from shared.vectordb.vectordb_provider_facotry import VectorDBProviderFactory

from api.helpers.config import get_settings
from api.helpers.utils import get_redis_client
from api.routes.chat import chat_route
from api.routes.health import health_route
from api.services.caching_service import SemanticCacheService
from api.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    # clients
    redis_client = get_redis_client()
    embedding_model = get_embedding_model()

    generation_model = OpenAIProvider(
        api_key=settings.GENERATION_MODEL_API_KEY,
        api_url=settings.GENERATION_MODEL_PROVIDER_URL,
        default_max_input_tokens=settings.MAX_INPUT_CHARACTERS,
        default_max_output_tokens=settings.MAX_OUTPUT_TOKENS,
        default_temperature=settings.TEMPERATURE,
    )
    generation_model.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    vector_provider_factory = VectorDBProviderFactory()
    vectordb_client = await vector_provider_factory.create()

    # services
    rag_service = RAGService(
        vectordb_client=vectordb_client,
        embedding_client=embedding_model,
        generation_client=generation_model,
    )

    cache_service = SemanticCacheService()

    app.state.rag_service = rag_service
    app.state.cache_service = cache_service

    try:
        yield
    finally:
        await vectordb_client.disconnect()
        await app.state.cache_service.cache.adisconnect()
        await redis_client.aclose()


app = FastAPI(lifespan=lifespan)

app.include_router(health_route)
app.include_router(chat_route)
