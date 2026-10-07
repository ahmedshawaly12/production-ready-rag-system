import asyncio
import logging
import random
import time
from collections import deque
from collections.abc import AsyncGenerator

from openai import AsyncOpenAI, RateLimitError

from shared.llm.llm_interface import LLMInterface


class OpenAIProvider(LLMInterface):
    def __init__(
        self,
        api_key: str,
        api_url: str | None = None,
        default_max_input_tokens: int = 1000,
        default_max_output_tokens: int = 1000,
        default_temperature: float = 0.1,
    ):
        self.api_key = api_key
        self.api_url = api_url

        self.default_max_input_tokens = default_max_input_tokens
        self.default_max_output_tokens = default_max_output_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None
        self.embedding_model_id = None

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_url if self.api_url and len(self.api_url) else None,
        )

        self.logger = logging.getLogger(__name__)

        # Token-per-minute configuration
        self.tpm_limit = 100_000
        self._token_window = deque()

        # Prevent multiple concurrent requests from modifying the token window at the same times
        self._token_lock = asyncio.Lock()

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str):
        self.embedding_model_id = model_id

    # Token Management
    def _estimate_tokens(self, texts: list[str]) -> int:
        return sum(len(t) for t in texts) // 3 + len(texts)

    async def _wait_for_budget(self, needed: int):
        limit = int(self.tpm_limit * 0.8)  # 20% safety margin
        needed = min(needed, limit)

        while True:
            async with self._token_lock:
                now = time.monotonic()
                while self._token_window and now - self._token_window[0][0] > 60:
                    self._token_window.popleft()

                used = sum(t for _, t in self._token_window)
                if used + needed <= limit:
                    self._token_window.append((now, needed))
                    return

                wait = 60 - (now - self._token_window[0][0]) + 0.1
            self.logger.info(f"Token budget reached, sleeping {wait:.1f}s")
            await asyncio.sleep(max(wait, 0.5))

    # Embed one batch asynchronously with retry handling
    async def _embed_batch(self, batch: list[str], max_retries: int = 5):
        for attempt in range(max_retries):
            await self._wait_for_budget(self._estimate_tokens(batch))
            try:
                return await self.client.embeddings.create(
                    model=self.embedding_model_id, input=batch
                )
            except RateLimitError:
                backoff = min(5 * 2**attempt, 60) + random.random()
                self.logger.warning(f"429 received, retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)

        self.logger.error(f"Embedding failed after {max_retries} retries")
        return None

    async def embed_text(self, text: str | list[str], batch_size: int = 32):
        if not self.client:
            self.logger.error("LLM provider client wasn't set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model wasn't set")
            return None

        if isinstance(text, str):
            text = [text]

        all_embeddings = []

        for i in range(0, len(text), batch_size):
            batch = text[i : i + batch_size]

            response = await self._embed_batch(batch)

            if (
                not response
                or not response.data
                or len(response.data) == 0
                or not response.data[0].embedding
            ):
                self.logger.error("Error while embedding the text batch")
                return None

            all_embeddings.extend([res.embedding for res in response.data])

        return all_embeddings

    def process_text(self, text: str):
        return text[: self.default_max_input_tokens].strip()

    async def generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ):
        if not self.client:
            self.logger.error("LLM provider client wasn't set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model wasn't set")
            return None

        if chat_history is None:
            chat_history = []

        temperature = temperature or self.default_temperature
        max_output_tokens = max_output_tokens or self.default_max_output_tokens
        chat_history.append(self.construct_prompt(self.process_text(prompt), "user"))

        try:
            response = self.client.chat.completions.create(
                model=self.generation_model_id,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=chat_history,
            )
        except RateLimitError:
            self.logger.exception("Rate limit exceeded during generation")
            return None

        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            self.logger.error("Error while generating the response")
            return None

        return response.choices[0].message.content

    async def stream_generate_text(
        self,
        prompt: str,
        chat_history: list | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[str, None]:
        if not self.client:
            self.logger.error("LLM provider client wasn't set")
            return

        if not self.generation_model_id:
            self.logger.error("Generation model wasn't set")
            return

        if chat_history is None:
            chat_history = []

        temperature = temperature or self.default_temperature
        max_output_tokens = max_output_tokens or self.default_max_output_tokens
        chat_history.append(self.construct_prompt(self.process_text(prompt), "user"))

        try:
            stream = await self.client.chat.completions.create(
                model=self.generation_model_id,
                max_tokens=max_output_tokens,
                temperature=temperature,
                messages=chat_history,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if not delta:
                    continue

                content = delta.content
                if content:
                    yield content

        except RateLimitError:
            self.logger.exception("Rate limit exceeded during streaming generation")
            return

        except asyncio.CancelledError:
            self.logger.info("Streaming generation cancelled")
            raise

        except Exception:
            self.logger.exception("Error while streaming generation")
            return

    def construct_prompt(self, prompt: str, role):
        return {"role": role, "content": prompt}
