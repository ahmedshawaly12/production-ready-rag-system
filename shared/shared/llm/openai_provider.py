import logging

from openai import OpenAI

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

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url if self.api_url and len(self.api_url) else None,
        )

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str):
        self.embedding_model_id = model_id

    def process_text(self, text: str):
        return text[: self.default_max_input_tokens].strip()

    def generate_text(
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

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=chat_history,
        )

        if (
            not response
            or not response.choices
            or len(response.choices) == 0
            or not response.choices[0].message
        ):
            self.logger.error("Error while generating the response")
            return None

        return response.choices[0].message.content

    def embed_text(self, text: str | list[str]):
        if not self.client:
            self.logger.error("LLM provider client wasn't set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model wasn't set")
            return None

        if isinstance(text, str):
            text = [text]

        response = self.client.embeddings.create(
            model=self.embedding_model_id, input=text
        )

        if (
            not response
            or not response.data
            or len(response.data) == 0
            or not response.data[0].embedding
        ):
            self.logger.error("Error while Embedding the text")
            return None

        return [res.embedding for res in response.data]

    def construct_prompt(self, prompt: str, role):
        return {"role": role, "content": prompt}
