# Legixo Take-Home: Grounded Q&A API

A question-answering HTTP API over a small fictional legal corpus. Questions are
answered **only from the document set**, every answer cites the real chunks it
came from, and out-of-scope questions are refused.

**Stack:** Python 3.10+ · FastAPI · **LangGraph** (workflow orchestration) ·
**Pinecone** (serverless vector index) · Google **Gemini** (chat + embeddings,
works on the free tier).

## Architecture

The Q&A flow is a LangGraph `StateGraph` (`app/graph.py`). Full node-by-node
map in [`docs/langgraph.md`](docs/langgraph.md) (editable Excalidraw source:
[`docs/langgraph.excalidraw`](docs/langgraph.excalidraw)):

![LangGraph workflow diagram](docs/langgraph.svg)

- **Branch node:** `grade_chunks` routes between the good path (generate) and the
  bad path (rewrite & retry, or refuse).
- **Loop limit:** at most 2 query rewrites (`max_retrieval_loops`), plus a hard
  LangGraph `recursion_limit` backstop — the graph can never spin forever.
- **No fabricated citations:** `validate_citations` keeps only cited chunk IDs
  that were actually retrieved from Pinecone this run. If nothing valid remains,
  the API refuses instead of answering.

## Setup

Requires Python 3.10+.

```bash
git clone <this-repo> && cd <this-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # then edit .env with your real keys
```

### Environment variables (`.env`)

| Variable | Required | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Google AI Studio key (free tier OK): https://aistudio.google.com/apikey |
| `PINECONE_API_KEY` | yes | From https://app.pinecone.io |
| `PINECONE_INDEX_NAME` | no | Default `legixo-takehome` |
| `PINECONE_NAMESPACE` | no | Default `corpus` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no | Default `aws` / `us-east-1` (serverless free tier) |
| `GEMINI_CHAT_MODEL` / `GEMINI_EMBED_MODEL` | no | Defaults `gemini-flash-latest` / `models/gemini-embedding-001` (truncated to 768 dims to match the index) |

Keys live only in `.env` (gitignored). `.env.example` contains placeholders only.

## Ingestion

```bash
python ingest.py
```

This chunks every markdown file in `data/corpus/` by `##` section, embeds the
chunks with Gemini (`gemini-embedding-001`, truncated to 768-dim), **creates the Pinecone
serverless index automatically if it doesn't exist** (cosine, dim 768), and
upserts vectors with metadata `{chunk_id, source_file, section, text}`.

**What happens if you run ingest twice?** Nothing bad — chunk IDs are
deterministic (`<file-stem>::<section-slug>::<n>`), so a second run upserts the
same IDs in place and the vector count stays constant. To rebuild from scratch
(e.g. after deleting a corpus file), wipe the namespace first:

```bash
python ingest.py --reset
```

### Pinecone checklist

- [x] **How the index is created:** `python ingest.py` calls `ensure_index()`
  (`app/vectorstore.py`), which creates a **serverless** index automatically if
  it doesn't exist — name from `PINECONE_INDEX_NAME`, dimension **768**, metric
  **cosine**, on `PINECONE_CLOUD`/`PINECONE_REGION` (default `aws`/`us-east-1`,
  available on the Pinecone free tier) — and waits until it reports ready. No
  manual console step is required; creating an index with those exact settings
  in the Pinecone console works too.
- [x] **Env vars needed** (all listed in `.env.example`): `PINECONE_API_KEY`
  (required), plus optional `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE`,
  `PINECONE_CLOUD`, `PINECONE_REGION`. Embeddings also need `GOOGLE_API_KEY`.
- [x] **What happens if you run ingest twice:** nothing duplicates — every
  chunk has a **deterministic point ID** (`<file-stem>::<section-slug>::<n>`),
  so the second run upserts the **same IDs** in place and the namespace vector
  count stays constant (verified: 16 vectors after two consecutive runs).
  `python ingest.py --reset` deletes the namespace first for a from-scratch
  rebuild (use after removing or renaming corpus files, since orphaned IDs are
  not garbage-collected by a plain re-ingest).

## Run the server

```bash
uvicorn app.main:app --reload
```

- `GET /health` — liveness check (works even before keys are configured)
- `POST /ask` — ask a question

### Examples

Answerable question:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period in the Bluecrest employment agreement?", "include_trace": true}'
```

```json
{
  "answer": "Either party may end the agreement by giving 60 days written notice.",
  "found": true,
  "citations": [
    {
      "chunk_id": "02_employment_agreement_excerpt::notice-period::0",
      "source_file": "02_employment_agreement_excerpt.md",
      "section": "Notice period",
      "snippet": "## Notice period\n\nEither party may end this agreement by giving **60 days** written notice...",
      "score": 0.8123
    }
  ],
  "trace": [
    "retrieve: query='What is the notice period...' -> 4 chunks [...]",
    "grade_chunks: sufficient",
    "generate_answer: found=True, cited=['02_employment_agreement_excerpt::notice-period::0']",
    "validate_citations: 1 valid, 0 dropped"
  ]
}
```

Out-of-scope question (refused, no citations):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

```json
{
  "answer": "I cannot find the answer to this question in the provided documents.",
  "found": false,
  "citations": []
}
```

Error responses (exemplars — standard FastAPI `detail` envelope):

```json
// 500 — server started without keys (.env missing or placeholder values)
{
  "detail": "Missing required environment variable GOOGLE_API_KEY. Copy .env.example to .env and fill in your keys (see README)."
}
```

```json
// 502 — upstream provider failure, e.g. Gemini free-tier quota exhausted
{
  "detail": "Upstream error: Error calling model 'gemini-flash-latest' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. ... Please retry in 23s."
}
```

The free tier allows ~20 requests/minute and each question costs 2+ model calls
(grade + generate, plus rewrites on the bad path), so space manual requests
~30s apart; the self-test suite backs off and retries automatically.

Postman: import as a raw `POST http://127.0.0.1:8000/ask` request with the JSON
bodies above, header `Content-Type: application/json`. Interactive docs are also
available at http://127.0.0.1:8000/docs.

## Self-tests

With the corpus ingested and the server running:

```bash
python tests/self_test.py
```

The cases live in [`tests/self_test_cases.json`](tests/self_test_cases.json) —
13 entries with columns `question`, `expected_citation_files`,
`expected_answer_keywords`, `answerable`, and `notes` (pass/fail observations
from actual runs). `tests/self_test.py` loads that file and runs every case:
10 answerable questions (asserting expected facts in the answer **and** that
citations point at the correct source file) and 3 out-of-scope questions
(asserting refusal with zero citations). Exits non-zero on failure. Point at a
different host with `API_URL=http://host:port`.

The suite is paced for the Gemini free tier (~10 requests/min): it pauses
between cases (`SELF_TEST_PAUSE`, default 5s) and backs off and retries when it
hits a 429 quota error, so a full run takes a few minutes on a free key.

## Project layout

```
app/
  chunking.py      # markdown-section chunking, deterministic chunk IDs (pure, offline)
  config.py        # env settings, lazy-loaded
  llm.py           # Gemini chat + embeddings factories
  vectorstore.py   # Pinecone index lifecycle, upsert, query
  graph.py         # LangGraph StateGraph (retrieve -> grade -> answer/rewrite/refuse)
  main.py          # FastAPI: GET /health, POST /ask
ingest.py          # ingestion CLI (--reset to wipe namespace)
data/corpus/       # the provided fictional corpus
tests/self_test.py # 13-case self-test suite
```

## Known omissions

- No optional enhancements (LangSmith tracing, hybrid search, reranking) — the
  `trace` field in `/ask` responses covers basic observability.
- Chunk grading uses a single LLM judgment for the whole retrieved set rather
  than per-chunk grading; adequate for a 6-document corpus.
- No streaming responses; answers are returned as a single JSON object.
