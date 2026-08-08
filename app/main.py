"""FastAPI app exposing the Q&A graph.

Endpoints:
    GET  /health  - liveness check, no external calls
    POST /ask     - {"question": "...", "include_trace": false}
                    -> {"answer", "found", "citations", "trace"?}

The QAService (and therefore the API keys) is initialized lazily on the first
/ask call, so the server starts and /health works even before .env is set up.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import ConfigError
from app.graph import QAService

app = FastAPI(
    title="Legixo Take-Home Q&A API",
    description="Grounded Q&A over a fictional legal corpus (LangGraph + Pinecone + Gemini).",
)

_service: QAService | None = None


def get_service() -> QAService:
    global _service
    if _service is None:
        _service = QAService()
    return _service


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to answer.")
    include_trace: bool = Field(
        default=False, description="Include the LangGraph execution trace in the response."
    )


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    section: str
    snippet: str
    score: float


class AskResponse(BaseModel):
    answer: str
    found: bool
    citations: list[Citation]
    trace: list[str] | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse, response_model_exclude_none=True)
def ask(request: AskRequest) -> AskResponse:
    try:
        service = get_service()
        result = service.ask(request.question)
    except ConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # surface provider/Pinecone errors cleanly
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}") from exc
    return AskResponse(
        answer=result["answer"],
        found=result["found"],
        citations=result["citations"],
        trace=result["trace"] if request.include_trace else None,
    )
