import argparse
import asyncio
import json
import logging
import math
from datetime import UTC, datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langfuse import Evaluation, Langfuse
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from api.helpers.config import get_settings
from api.helpers.prompts import build_context
from api.routes.chat import NO_CONTEXT_ANSWER
from api.services.rag_service import RAGService
from shared.llm.embedding_model import get_embedding_model
from shared.llm.openai_provider import OpenAIProvider
from shared.prompts.prompt_manager import PromptManager
from shared.vectordb.vectordb_provider_facotry import VectorDBProviderFactory

log = logging.getLogger("rag_eval")

METRIC_NAMES = ("faithfulness", "context_precision", "context_recall")


# ---- 1. settings of this run ------------------------------------------------
def parse_args(settings) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model-name", required=True)
    p.add_argument("--chunk-size", type=int, required=True)
    p.add_argument("--chunk-overlap", type=int, required=True)
    p.add_argument("--judge-model", required=True, help="not the generation model")
    p.add_argument("--gen-model-name", default=settings.GENERATION_MODEL_ID)
    p.add_argument("--temperature", type=float, default=settings.TEMPERATURE)
    p.add_argument("--top-k", type=int, default=settings.TOP_K)
    p.add_argument("--dataset", type=Path, default=Path("evals/dataset.jsonl"))
    return p.parse_args()


def load_data(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [{"input": r["question"], "expected_output": r["reference"]} for r in rows]


# ---- 2. the RAG pipeline under test ----------------------------------------
class RagPipeline:
    """The answer path of /chat, without streaming."""

    def __init__(self, settings, args: argparse.Namespace, langfuse: Langfuse):
        self.settings = settings
        self.args = args
        self.prompt_manager = PromptManager(langfuse_client=langfuse)
        self.prompt_manager.load_prompt(name=settings.PROMPT_NAME, label="production")
        self._rag: RAGService | None = None
        self._lock: asyncio.Lock | None = None

    async def _service(self) -> RAGService:
        """Built on first use, inside the event loop Langfuse runs tasks in."""
        self._lock = self._lock or asyncio.Lock()
        async with self._lock:
            if self._rag is None:
                self._rag = await self._build_service()
        return self._rag

    async def _build_service(self) -> RAGService:
        s, a = self.settings, self.args
        generation = OpenAIProvider(
            api_key=s.GENERATION_MODEL_API_KEY,
            api_url=s.GENERATION_MODEL_PROVIDER_URL,
            default_max_input_tokens=s.MAX_INPUT_CHARACTERS,
            default_max_output_tokens=s.MAX_OUTPUT_TOKENS,
            default_temperature=a.temperature,
        )
        generation.set_generation_model(model_id=a.gen_model_name)
        service = RAGService(
            vectordb_client=await VectorDBProviderFactory().create(),
            embedding_client=get_embedding_model(),
            generation_client=generation,
        )
        service.app_settings = s.model_copy(update={"TOP_K": a.top_k})
        return service

    async def task(self, *, item, **_kwargs) -> dict:
        """Question in, answer + retrieved chunks out."""
        question = item["input"]
        service = await self._service()

        embedding = await service.embed(question)
        if not embedding.vector:
            raise RuntimeError("Embedding failed")

        docs = await service.retrieve(embedding.vector) or []
        min_score = getattr(self.settings, "MIN_RETRIEVAL_SCORE", None)
        if min_score is not None:
            docs = [d for d in docs if d.score >= min_score]
        if not docs:
            return {"answer": NO_CONTEXT_ANSWER, "contexts": []}

        _, messages = await asyncio.to_thread(
            self.prompt_manager.compile_prompt,
            name=self.settings.PROMPT_NAME,
            question=question,
            context=build_context(documents=docs),
            chat_history=[],
        )
        answer = await service.generate(
            messages=messages,
            max_output_tokens=self.settings.MAX_OUTPUT_TOKENS,
            temperature=self.args.temperature,
        )
        return {"answer": answer or "", "contexts": [d.text for d in docs]}


# ---- 3. RAGAS scoring (judge and embeddings go through LiteLLM) -------------
def build_metrics(settings, args: argparse.Namespace) -> dict:
    judge_llm = ChatOpenAI(
        model=args.judge_model,
        base_url=settings.GENERATION_MODEL_PROVIDER_URL,
        api_key=settings.GENERATION_MODEL_API_KEY,
        temperature=0,
        max_retries=3,
    )

    judge = LangchainLLMWrapper(judge_llm)

    return {
        "faithfulness": Faithfulness(llm=judge),
        "context_precision": LLMContextPrecisionWithReference(llm=judge),
        "context_recall": LLMContextRecall(llm=judge),
    }


def make_evaluator(metrics: dict):
    """Item evaluator: scores one answer. A metric that fails is skipped."""

    async def evaluator(*, input, output, expected_output, **_kwargs):  # noqa: A002
        if not output["contexts"]:  # nothing retrieved: retrieval scores 0
            return [
                Evaluation(name="context_precision", value=0.0),
                Evaluation(name="context_recall", value=0.0),
            ]
        sample = SingleTurnSample(
            user_input=input,
            response=output["answer"],
            retrieved_contexts=output["contexts"],
            reference=expected_output,
        )
        scores = []
        for name, metric in metrics.items():
            try:
                value = float(await metric.single_turn_ascore(sample))
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", name, exc)
                continue
            if not math.isnan(value):
                scores.append(Evaluation(name=name, value=value))
        return scores

    return evaluator


def averages(*, item_results, **_kwargs):
    """Run evaluator: one average per metric for the whole run."""
    out = []
    for name in METRIC_NAMES:
        values = [
            e.value
            for result in item_results
            for e in result.evaluations
            if e.name == name
        ]
        if values:
            out.append(Evaluation(name=f"avg_{name}", value=sum(values) / len(values)))
    return out


# ---- 4. run it and save to Langfuse -----------------------------------------
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = get_settings()
    args = parse_args(settings)

    # These six values are stored on every trace of the run in Langfuse.
    attributes = {
        "embed_model_name": args.embed_model_name,
        "gen_model_name": args.gen_model_name,
        "temperature": args.temperature,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "top_k": args.top_k,
        "judge_model": args.judge_model,
    }

    langfuse = Langfuse(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_BASE_URL,
    )
    pipeline = RagPipeline(settings, args, langfuse)
    metrics = build_metrics(settings, args)

    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M")
    run_name = f"run-{stamp}"

    result = langfuse.run_experiment(
        name="rag-eval",
        run_name=run_name,
        description=", ".join(f"{k}={v}" for k, v in attributes.items()),
        data=load_data(args.dataset),
        task=pipeline.task,
        evaluators=[make_evaluator(metrics)],
        run_evaluators=[averages],
        max_concurrency=2,
        metadata={k: str(v) for k, v in attributes.items()},
    )
    print(result.format())
    langfuse.flush()


if __name__ == "__main__":
    main()
