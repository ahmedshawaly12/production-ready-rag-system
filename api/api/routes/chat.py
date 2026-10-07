import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from api.helpers.prompts import build_user_prompt, system_prompt
from api.routes.schemas.chat import ChatPayloadSchema
from api.services.caching_service import SemanticCacheService
from api.services.rag_service import RAGService

chat_route = APIRouter(prefix="/api/v1", tags=["chat_api", "api_v1"])

chat_route.post("/chat")

logger = logging.getLogger(__name__)


def _get_rag_service(request: Request) -> RAGService:
    return request.app.state.rag_service


def _get_cache_service(request: Request) -> SemanticCacheService:
    return request.app.state.cache_service


def _serialize_documents(documents) -> list[dict]:
    return [
        {"score": d.score, "text": d.text, "metadata": d.metadata} for d in documents
    ]


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@chat_route.post("/chat")
async def chat(
    request: Request,
    chat_payload: ChatPayloadSchema,
    rag_service: RAGService = Depends(_get_rag_service),
    semantic_cache_service: SemanticCacheService = Depends(_get_cache_service),
):
    question = chat_payload.question
    question_vector = await rag_service.embed(question)

    if not question_vector:
        raise HTTPException(status_code=502, detail="Failed to embed the question")

    cached = await semantic_cache_service.get(question_vector)

    async def event_generator():
        try:
            if cached:
                yield sse("documents", cached["documents"])
                yield sse("token", cached["response"])
                yield sse("done", {"cached": True})
                return

            documents = await rag_service.retrieve(question_vector) or []
            yield sse("documents", _serialize_documents(documents))

            system_message = rag_service.generation_client.construct_prompt(
                prompt=system_prompt,
                role="system",
            )

            user_message = build_user_prompt(
                query=question,
                documents=documents or [],
            )

            parts: list[str] = []
            yield "event: token\n"
            yield "data: "

            async for chunk in rag_service.stream_generate(
                prompt=user_message,
                chat_history=[system_message],
            ):
                if await request.is_disconnected():
                    return

                parts.append(chunk)
                # yield sse("token", chunk)
                yield chunk

            response = "".join(parts)
            yield "\n\n"

            if response:
                await semantic_cache_service.set(
                    prompt=question,
                    response=response,
                    documents=documents or [],
                    vector=question_vector,
                )
            yield sse("done", {"cached": False})

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
