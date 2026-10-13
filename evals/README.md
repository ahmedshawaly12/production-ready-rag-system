# RAG Evaluation

Scores the RAG pipeline with [RAGAS](https://docs.ragas.io) and saves each run in Langfuse, so you can compare settings (models, chunking, retrieval) side by side.

## How it works

```
dataset.jsonl ──► answer each question (real RAG path) ──► score with RAGAS ──► Langfuse experiment
```

| Part | Metric | Question it answers |
|---|---|---|
| Retrieval | `context_precision` | Are the retrieved chunks relevant? |
| Retrieval | `context_recall` | Was everything needed retrieved? |
| Generation | `faithfulness` | Is the answer supported by the chunks? |
| Generation | `answer_relevancy` | Does the answer address the question? |

Scores go from 0 to 1. Higher is better.

## Folder content

| File | Purpose |
|---|---|
| `dataset.jsonl` | Questions with reference answers |
| `run_eval.py` | Runs the questions through the pipeline, scores them, and sends the run to Langfuse |

## Before you run

- The stack is running (Qdrant, LiteLLM, Redis) and the Qdrant collection contains your documents. See the [root README](../README.md#setup-from-zero-to-first-answer).
- Langfuse keys are set (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`), because results are stored there.
- The judge model exists in `docker/litellm_proxy/config.yml`.

## Setup

Install the evaluation dependencies from the repository root:

```bash
uv add --group eval ragas langchain-openai
```

### The dataset

Each line of `dataset.jsonl` has a question and the correct answer, written from your documents:

```json
{"question": "Which foods are rich in vitamin C?", "reference": "Citrus fruits, bell peppers, strawberries and kiwi."}
```

Use 30-50 questions. Write the reference answers yourself from the documents. Do not generate them with the model you are testing.

## Run

Run the script with the same environment variables as the API. Outside Docker, use `localhost` addresses (for example `http://localhost:4000` for LiteLLM).

```bash
VECTORDB_PROVDER_URL=http://localhost:6333 \
EMBEDDING_MODEL_PROVIDER_URL=http://localhost:4000 \
GENERATION_MODEL_PROVIDER_URL=http://localhost:4000 \
uv run --group eval python evals/run_eval.py \
    --embed-model-name cohere/embed-multilingual-v3.0 \
    --chunk-size 250 \
    --chunk-overlap 50 \
    --judge-model qwen/qwen3.8-27b
```

The judge model must be a `model_name` from `docker/litellm_proxy/config.yml`. Put the model you want to use there first.

| Flag | Default | Meaning |
|---|---|---|
| `--embed-model-name` | required | Embedding model of the collection |
| `--chunk-size` | required | Chunk size the collection was built with |
| `--chunk-overlap` | required | Chunk overlap the collection was built with |
| `--judge-model` | required | Model that scores the answers. Use one that is not the generation model |
| `--gen-model-name` | `GENERATION_MODEL_ID` | Generation model |
| `--temperature` | `TEMPERATURE` | Generation temperature |
| `--top-k` | `TOP_K` | Chunks retrieved per question |
| `--dataset` | `evals/dataset.jsonl` | Questions file |

`--chunk-size` and `--chunk-overlap` are labels only. Pass the values the collection was built with (the `chunk_size` and `chunk_overlap` of the data pipeline settings, default 250 and 50), so the run is recorded correctly.

## What Langfuse stores

Every trace of a run carries `embed_model_name`, `gen_model_name`, `temperature`, `chunk_size`, `chunk_overlap` and `top_k`. They are also in the run name:

```
cohere-command-r-08-2024|cohere-embed-multilingual-v3.0|t0.0|cs250|co50|k5|2026-10-12_14-30
```

Open **Experiments** in the Langfuse project to compare runs. Each run has the four scores per question and their averages (`avg_faithfulness` and so on).

## Comparing settings

Change one thing at a time, so you know what caused a difference:

| To compare | What to change |
|---|---|
| Generation models | `--gen-model-name` |
| Retrieval depth | `--top-k` |
| Chunking | Rebuild the collection with new chunk settings (reindex flow), then pass the new `--chunk-size` and `--chunk-overlap` |
| Embedding models | New collection with the new model, then pass `--embed-model-name` |
| Prompts | Promote a different prompt version in Langfuse and restart the API |

## Reading the results

| You see | Likely cause | Try |
|---|---|---|
| Low `context_recall` | Needed information is not retrieved | Higher `--top-k`, other chunk settings |
| Low `context_precision` | Too many irrelevant chunks | Lower `--top-k`, higher `MIN_RETRIEVAL_SCORE` |
| Low `faithfulness` | The model adds facts not in the chunks | Stricter prompt, lower temperature |
| Low `answer_relevancy` | The answer drifts from the question | Clearer prompt, check that the question reaches the model (see the prompt variable note in the [root README](../README.md#2-create-the-prompt)) |
