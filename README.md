# Sourcebound

**Answers bound to their sources.**

A retrieval-augmented question-answering API that answers **only from the
documents you give it** and cites the exact chunks each answer came from. When
the documents don't contain the answer, it says so instead of guessing.

Built with FastAPI, LangGraph, Pinecone, and Google Gemini.

## Why this exists

Most RAG demos will happily answer from the model's world knowledge when
retrieval comes up short, and will just as happily invent a citation that looks
plausible. Both failures are invisible to the person reading the answer, which
makes the system untrustworthy exactly where grounding matters most — legal,
compliance, policy, and contract documents.

This project treats grounding as an architectural property rather than a prompt
instruction:

- **Retrieval quality is judged before answering.** A dedicated grading step
  decides whether the retrieved chunks can support an answer at all, and routes
  to a retry or an honest refusal when they can't.
- **Citations are verified, not trusted.** Every chunk ID the model cites is
  checked against the set actually returned by the vector store. Anything that
  doesn't match is dropped, and if nothing survives, the answer is replaced with
  a refusal — so a fabricated citation can never reach the caller.
- **The workflow cannot spin forever.** Query rewrites are capped and the graph
  carries a hard recursion limit.

## Architecture

The Q&A flow is a LangGraph `StateGraph` (`app/graph.py`). Full node-by-node
map in [`docs/langgraph.md`](docs/langgraph.md) (editable Excalidraw source:
[`docs/langgraph.excalidraw`](docs/langgraph.excalidraw)):

![LangGraph workflow diagram](docs/langgraph.svg)

| Node | Role |
|---|---|
| `retrieve` | Embed the current query, fetch top-k chunks from Pinecone |
| `grade_chunks` | Branch point — are these chunks good enough to answer from? |
| `rewrite_query` | Bad path: rephrase the query and retry (capped at 2 rewrites) |
| `generate_answer` | Good path: answer strictly from the chunks, declaring which it used |
| `validate_citations` | Drop unverifiable citations; refuse if none survive |
| `no_answer` | Bad path exhausted: refuse cleanly with zero citations |

## Stack

Python 3.10+ · FastAPI · LangGraph · Pinecone (serverless) · Google Gemini
(chat + embeddings, runs on the free tier).

## Quickstart

```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv
# Windows (cmd/PowerShell): .venv\Scripts\activate
# Windows (Git Bash):       source .venv/Scripts/activate
# macOS/Linux:              source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # then edit .env with your real keys
python ingest.py          # chunk, embed, and upsert the corpus
uvicorn app.main:app --reload
```

Then ask a question:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period in the employment agreement?"}'
```

### Configuration

| Variable | Required | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | yes | Google AI Studio key (free tier OK): https://aistudio.google.com/apikey |
| `PINECONE_API_KEY` | yes | From https://app.pinecone.io |
| `PINECONE_INDEX_NAME` | no | Default `sourcebound` |
| `PINECONE_NAMESPACE` | no | Default `corpus` |
| `PINECONE_CLOUD` / `PINECONE_REGION` | no | Default `aws` / `us-east-1` (serverless free tier) |
| `GEMINI_CHAT_MODEL` / `GEMINI_EMBED_MODEL` | no | Defaults `gemini-flash-latest` / `models/gemini-embedding-001` (truncated to 768 dims to match the index) |

Keys live only in `.env`, which is gitignored. `.env.example` holds placeholders
only — never commit real credentials.

## Using your own documents

Drop markdown files into `data/corpus/` and re-run `python ingest.py`. The
shipped corpus is a small set of **fictional** legal-style notes (invented
parties, courts, and statutes) used as sample data — replace it with whatever
document set you care about.

## Ingestion

```bash
python ingest.py            # incremental: re-upserts in place
python ingest.py --reset    # wipe the namespace first, then ingest fresh
```

**Index creation is automatic.** `ensure_index()` (`app/vectorstore.py`) creates
a serverless index if one doesn't exist — name from `PINECONE_INDEX_NAME`,
dimension **768**, metric **cosine**, on `PINECONE_CLOUD`/`PINECONE_REGION` — and
waits until it reports ready. Creating an index with those settings by hand in
the Pinecone console works too.

**Chunking.** Files are split on markdown `##` sections (with an
overlapping-window fallback for oversized sections), so each chunk stays
semantically self-contained. Every vector carries metadata: `chunk_id`,
`source_file`, `section`, and the chunk `text` — the last of which is what makes
citations possible without a second lookup.

**Running ingest twice is safe.** Chunk IDs are deterministic
(`<file-stem>::<section-slug>::<n>`), so a second run upserts the *same* point
IDs in place and the namespace vector count stays constant. Use `--reset` after
removing or renaming source files, since a plain re-ingest won't garbage-collect
point IDs that no longer correspond to a chunk.

## API

- `GET /health` — liveness check, makes no external calls (works before keys are configured)
- `POST /ask` — ask a question

`POST /ask` request body:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `question` | string | required | The question to answer |
| `include_trace` | bool | `false` | Include the graph execution trace in the response |

### Examples

Answerable question:

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the notice period in the employment agreement?", "include_trace": true}'
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

Question the documents don't cover — refused, with no citations:

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

Error responses (standard FastAPI `detail` envelope):

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

### Trying it interactively

- **Swagger UI:** http://127.0.0.1:8000/docs — no setup, works in the browser.
- **IDE:** open [`api.http`](api.http) in VS Code (REST Client extension) or a
  JetBrains IDE and click *Send Request* on any block.
- **Postman:** *Import → Link* → `http://127.0.0.1:8000/openapi.json` builds the
  whole collection from the OpenAPI spec; or create a `POST` to
  `http://127.0.0.1:8000/ask` with a raw JSON body.

### Free-tier rate limits

Gemini's free tier allows roughly 20 requests/minute with per-model daily caps,
and each question costs at least two model calls (grade + generate, plus one per
rewrite on the bad path). Space manual requests out, or switch
`GEMINI_CHAT_MODEL` to a model whose daily bucket is untouched. Quota errors
surface as the `502` shape above; the test suite handles them automatically.

## Tests

With the corpus ingested and the server running:

```bash
python tests/self_test.py
```

Cases live in [`tests/self_test_cases.json`](tests/self_test_cases.json) — each
entry has a `question`, the `expected_citation_files`, expected answer keywords,
an `answerable` flag, and notes from the last verified run. The suite covers 10
answerable questions (asserting both the expected facts **and** that citations
resolve to the right source file) and 3 out-of-scope questions (asserting
refusal with zero citations). It exits non-zero on failure, paces itself for
free-tier limits, and retries through quota errors. Point it at another host
with `API_URL=http://host:port`, and tune pacing with `SELF_TEST_PAUSE`.

Adding a case is just another JSON object — no Python changes needed.

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
data/corpus/       # sample fictional documents — swap in your own
docs/langgraph.md  # node-by-node graph map + diagram
tests/             # JSON-driven self-test suite
api.http           # ready-made requests for IDE HTTP clients
```

## Limitations and roadmap

- Chunk grading is a single judgment over the whole retrieved set rather than
  per-chunk scoring — fine at this corpus size, worth revisiting for larger sets.
- Dense retrieval only. Hybrid search and a reranking stage are the most
  promising accuracy upgrades.
- Responses are returned as a single JSON object; no streaming.
- Observability is limited to the optional `trace` field. Wiring in LangSmith
  would give proper per-node timings and token accounting.
- Markdown-only ingestion. PDF and DOCX loaders would broaden the input set.
