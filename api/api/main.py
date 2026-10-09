from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse
from shared.llm.embedding_model import get_embedding_model
from shared.llm.openai_provider import OpenAIProvider
from shared.prompts.prompt_manager import PromptManager
from shared.vectordb.vectordb_provider_facotry import VectorDBProviderFactory

from api.helpers.config import get_settings
from api.routes.chat import chat_route
from api.routes.health import health_route
from api.services.caching_service import SemanticCacheService
from api.services.rag_service import RAGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    async with AsyncExitStack() as stack:
        langfuse = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_BASE_URL,
        )
        stack.callback(langfuse.shutdown)

        prompt_manager = PromptManager(langfuse_client=langfuse)
        prompt_manager.load_prompt(name=settings.PROMPT_NAME, label="production")

        embedding_model = get_embedding_model()

        generation_model = OpenAIProvider(
            api_key=settings.GENERATION_MODEL_API_KEY,
            api_url=settings.GENERATION_MODEL_PROVIDER_URL,
            default_max_input_tokens=settings.MAX_INPUT_CHARACTERS,
            default_max_output_tokens=settings.MAX_OUTPUT_TOKENS,
            default_temperature=settings.TEMPERATURE,
        )
        generation_model.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

        vectordb_client = await VectorDBProviderFactory().create()
        stack.push_async_callback(vectordb_client.disconnect)

        cache_service = SemanticCacheService()
        stack.push_async_callback(cache_service.close)

        rag_service = RAGService(
            vectordb_client=vectordb_client,
            embedding_client=embedding_model,
            generation_client=generation_model,
        )

        app.state.settings = settings
        app.state.langfuse = langfuse
        app.state.prompt_manager = prompt_manager
        app.state.rag_service = rag_service
        app.state.cache_service = cache_service

        yield


app = FastAPI(lifespan=lifespan)

app.include_router(health_route)
app.include_router(chat_route)
