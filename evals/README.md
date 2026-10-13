# RAG Evaluation

Scores the RAG pipeline with [RAGAS](https://docs.ragas.io) and saves each run in Langfuse, so you can compare settings.

## How it works


```
questions.jsonl ──► answer each question (real RAG path) ──► score with RAGAS ──► Langfuse experiment
```

| Part | Metric | Question it answers |
|---|---|---|
| Retrieval | `context_precision` | Are the retrieved chunks relevant? |
| Retrieval | `context_recall` | Was everything needed retrieved? |
| Generation | `faithfulness` | Is the answer supported by the chunks? |

Scores go from 0 to 1. Higher is better.

## Setup

```bash
uv add --group eval ragas langchain-openai
```

Each line of the dataset has a question and the correct answer, written from your documents:

```json
{"question": "Which foods are rich in vitamin C?", "reference": "Citrus fruits, bell peppers, strawberries and kiwi."}
```

Use 30-50 questions. Run the script with the same environment variables as the API, using `localhost` addresses when you are outside Docker (for example `http://localhost:4000` for LiteLLM).

## Run

```bash
VECTORDB_PROVDER_URL=http://localhost:6333 \
EMBEDDING_MODEL_PROVIDER_URL=http://localhost:4000 \
GENERATION_MODEL_PROVIDER_URL=http://localhost:4000 \
uv run --group eval python evals/run_eval.py \
    --embed-model-name cohere/embed-multilingual-v3.0 \
    --chunk-size 512 \
    --chunk-overlap 64 \
    --judge-model qwen/qwen3.8-27b # this model should be exist in litellm config so pu the mode you need to use
```

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

`--chunk-size` and `--chunk-overlap` are labels. Pass the values the collection was built with.

## What Langfuse stores

Every trace of a run carries `embed_model_name`, `gen_model_name`, `temperature`, `chunk_size`, `chunk_overlap` and `top_k`. They are also in the run name:

```
cohere-command-r-08-2024|cohere-embed-multilingual-v3.0|t0.0|cs512|co64|k5|2026-10-12_14-30
```

Open **Experiments** in the Langfuse project to compare runs. Each run has the four scores per question and their averages (`avg_faithfulness` and so on).



## Reading the results

| You see | Likely cause | Try |
|---|---|---|
| Low `context_recall` | Needed information is not retrieved | Higher `--top-k`, other chunk settings |
| Low `context_precision` | Too many irrelevant chunks | Lower `--top-k`, higher `MIN_RETRIEVAL_SCORE` |
| Low `faithfulness` | The model adds facts not in the chunks | Stricter prompt, lower temperature |

## Good to know

- Scoring calls an LLM several times per question, and `answer_relevancy` also calls the embedding model. Make sure the `rpm` limits in `docker/litellm_proxy/config.yml` are high enough, or you will see rate-limit errors.
- If a question retrieves nothing, both retrieval scores are 0.
- A metric that fails for one question is skipped, with a warning in the output.
- If `answer_relevancy` fails with your provider, set `strictness=1` on `ResponseRelevancy` in `run_eval.py`.
