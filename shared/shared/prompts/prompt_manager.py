from langfuse import Langfuse


class PromptManager:
    def __init__(self, langfuse_client: Langfuse):
        self.langfuse = langfuse_client
        self._prompts = {}

    def load_prompt(
        self,
        name: str,
        version: int | None = None,
        label: str | None = "production",
    ):
        if version is not None:
            prompt = self.langfuse.get_prompt(name=name, version=version, type="chat")

        else:
            prompt = self.langfuse.get_prompt(name=name, label=label, type="chat")

        self._prompts[name] = prompt
        return prompt

    def compile_prompt(
        self,
        name: str,
        question: str,
        context: str,
        chat_history: list[dict] | None = None,
    ):
        prompt = self._prompts[name]

        messages = prompt.compile(
            question=question, context=context, chat_history=chat_history or []
        )

        return prompt, messages
