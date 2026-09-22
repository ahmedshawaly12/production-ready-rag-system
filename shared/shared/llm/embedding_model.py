from shared.config import get_settings
from shared.llm.openai_provider import OpenAIProvider


def get_embedding_model(
    embedding_model_id: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
):
    settings = get_settings()

    embedding_model_id = embedding_model_id or settings.EMBEDDING_MODEL_ID
    api_key = api_key or settings.EMBEDDING_MODEL_API_KEY
    api_url = api_url or settings.EMBEDDING_MODEL_PROVIDER_URL

    embedding_model = OpenAIProvider(
        api_key=api_key,
        api_url=api_url,
    )

    embedding_model.set_embedding_model(embedding_model_id)
    return embedding_model
