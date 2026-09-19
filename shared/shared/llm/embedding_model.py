from shared.config import get_settings
from shared.llm.openai_provider import OpenAIProvider


def get_embedding_model():
    settings = get_settings()

    embedding_model = OpenAIProvider(
        api_key=settings.EMBEDDING_MODEL_API_KEY,
        api_url=settings.EMBEDDING_MODEL_PROVIDER_URL,
    )

    embedding_model.set_embedding_model(settings.EMBEDDING_MODEL_ID)
    return embedding_model
