import asyncio
import json
import logging
from contextlib import aclosing
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from langfuse import propagate_attributes

from api.helpers.prompts import build_context
from api.helpers.utils import serialize_documents
from api.routes.schemas.chat import ChatPayloadSchema
from api.services.caching_service import SemanticCacheService
from api.services.rag_service import RAGService

chat_route = APIRouter(
    prefix="/api/v1",
    tags=["chat_api", "api_v1"],
)

logger = logging.getLogger(__name__)


NO_CONTEXT_ANSWER = (
    "The provided nutrition content does not contain enough information "
    "to answer the question."
)


# dependencies
def _get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def _get_cache_service(request: Request) -> SemanticCacheService:
    return request.app.state.cache_service


# helpers
def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _filter_by_score(documents, min_score: float | None):
    """Drop weak matches. Qdrant cosine/dot: higher score = more similar."""
    if min_score is None:
        return documents
    return [doc for doc in documents if doc.score >= min_score]


async def _cache_get(service: SemanticCacheService, **kwargs):
    """A cache failure must behave like a cache miss, not a failed request."""
    try:
        return await service.get(**kwargs)
    except Exception:
        logger.warning("Semantic cache lookup failed", exc_info=True)
        return None


async def _cache_set(service: SemanticCacheService, **kwargs) -> None:
    try:
        await service.set(**kwargs)
    except Exception:
        logger.warning("Semantic cache write failed", exc_info=True)


@chat_route.post("/chat")
async def chat(
    request: Request,
    chat_payload: ChatPayloadSchema,
    rag_service: RAGService = Depends(_get_rag_service),
    semantic_cache_service: SemanticCacheService = Depends(_get_cache_service),
):
    settings = request.app.state.settings
    langfuse = request.app.state.langfuse
    prompt_manager = request.app.state.prompt_manager
    guardrails_service = request.app.state.guardrails_service

    question = guardrails_service.mask_input(chat_payload.question)

    # Later these will come from the authenticated user request.
    user_id = "dummy_user_123"
    session_id = "dummy_session_123"
    conversation_id = "dummy_conversation_123"

    async def event_generator():
        response = ""

        try:
            with (
                langfuse.start_as_current_observation(
                    as_type="span", name="rag-chat", input={"question": question}
                ) as root_span,
                propagate_attributes(  # Propagate user/session/conversation information
                    user_id=user_id,
                    session_id=session_id,
                    metadata={"conversation_id": conversation_id},
                ),
            ):
                # 1. Embed question
                with langfuse.start_as_current_observation(
                    name="question-embedding",
                    as_type="embedding",
                    input={"text": question},
                ) as embedding_observation:
                    embedding = await rag_service.embed(question)
                    question_vector = embedding.vector

                    if question_vector:
                        embedding_observation.update(
                            model=embedding.model,  # real model
                            usage_details={"input": embedding.prompt_tokens},
                            output={"dimensions": len(question_vector)},
                        )

                    else:
                        embedding_observation.update(
                            level="ERROR", status_message="Failed to embed question"
                        )

                if not question_vector:
                    root_span.update(
                        level="ERROR", status_message="Failed to embed question"
                    )
                    yield sse("error", {"detail": "Failed to embed the question"})
                    return

                # 2. Semantic cache lookup
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="semantic-cache-lookup",
                    input={
                        "user_id": user_id,
                        "session_id": session_id,
                        "conversation_id": conversation_id,
                    },
                ) as cache_span:
                    cached = await _cache_get(
                        semantic_cache_service,
                        vector=question_vector,
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                    )
                    cache_span.update(output={"cache_hit": cached is not None})

                # 3. Return cached response
                if cached:
                    response = cached["response"]
                    root_span.update(output={"response": response, "cached": True})

                    yield sse("documents", cached["documents"])
                    yield sse("token", response)
                    yield sse("done", {"cached": True})

                    return

                # 4. Vector retrieval
                with langfuse.start_as_current_observation(
                    as_type="span",
                    name="vector-retrieval",
                    input={
                        "query": question,
                        "top_k": settings.TOP_K,
                        "collection": (settings.USED_COLLECTION_NAME),
                    },
                ) as retrieval_span:
                    documents = await rag_service.retrieve(question_vector) or []

                    documents = _filter_by_score(
                        documents, getattr(settings, "MIN_RETRIEVAL_SCORE", None)
                    )

                    retrieval_span.update(output={"documents_count": len(documents)})

                if not documents:
                    root_span.update(
                        output={"response": NO_CONTEXT_ANSWER, "cached": False}
                    )
                    yield sse("documents", [])
                    yield sse("token", NO_CONTEXT_ANSWER)
                    yield sse("done", {"cached": False})
                    return

                yield sse("documents", serialize_documents(documents))

                # 5. Build context and compile prompt
                with langfuse.start_as_current_observation(
                    as_type="span", name="prompt-compilation"
                ) as prompt_span:
                    chat_history = []

                    context = build_context(documents=documents)

                    # Sync SDK call: keep it off the event loop.
                    prompt, messages = await asyncio.to_thread(
                        prompt_manager.compile_prompt,
                        name=settings.PROMPT_NAME,
                        question=question,
                        context=context,
                        chat_history=chat_history,
                    )

                    prompt_span.update(
                        output={
                            "prompt_name": prompt.name,
                            "prompt_version": prompt.version,
                            "messages_count": len(messages),
                        }
                    )

                # 7. Stream generation
                parts: list[str] = []
                first_token_seen = False

                yield "event: token\n"
                yield "data: "

                with langfuse.start_as_current_observation(
                    as_type="generation",
                    name="rag-generation",
                    model_parameters={
                        "temperature": settings.TEMPERATURE,
                        "max_tokens": settings.MAX_OUTPUT_TOKENS,
                    },
                    input=messages,
                    prompt=prompt,
                ) as generation:
                    async with aclosing(
                        rag_service.stream_generate(messages=messages)
                    ) as stream:
                        async for chunk in stream:
                            # StreamChunk
                            if chunk.content:
                                if not first_token_seen:
                                    first_token_seen = True
                                    generation.update(
                                        completion_start_time=datetime.now(UTC),
                                        model=chunk.model,
                                    )
                                parts.append(chunk.content)
                                yield chunk.content

                            # Usage arrives in the final chunk.
                            if chunk.usage:
                                generation.update(
                                    usage_details={
                                        "input": chunk.usage.get("prompt_tokens", 0),
                                        "output": chunk.usage.get(
                                            "completion_tokens", 0
                                        ),
                                        "total": chunk.usage.get("total_tokens", 0),
                                    }
                                )

                    response = "".join(parts)

                    generation.update(
                        output=response,
                    )

                yield "\n\n"

                if not response:
                    root_span.update(level="ERROR", status_message="Empty generation")
                    yield sse("error", {"detail": "Generation returned no content"})
                    return

                root_span.update(output={"response": response, "cached": False})
                yield sse("done", {"cached": False})

                # 8. Store response in semantic cache
                with langfuse.start_as_current_observation(
                    as_type="span", name="semantic-cache-write"
                ):
                    await _cache_set(
                        semantic_cache_service,
                        prompt=question,
                        response=response,
                        documents=documents,
                        vector=question_vector,
                        user_id=user_id,
                        session_id=session_id,
                        conversation_id=conversation_id,
                    )

        except Exception:
            logger.exception("Streaming failed")

            yield sse("error", {"detail": "Generation failed"})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
