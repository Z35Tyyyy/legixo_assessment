"""LangGraph workflow for grounded Q&A over the corpus.

Graph shape (see README diagram):

    retrieve -> grade_chunks --(sufficient)--> generate_answer -> validate_citations -> END
                    |
                (insufficient, loops < MAX)--> rewrite_query -> retrieve   (loop)
                    |
                (loops exhausted)-----------> no_answer -> END

Guardrails encoded here:
- grade_chunks is the required branch node (good path vs bad path).
- `loops` counter + LangGraph recursion_limit enforce the max-loop requirement.
- validate_citations drops any cited chunk ID that was not actually retrieved,
  so the API can never return a fabricated citation.
"""

import json
import operator
import re
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from app import vectorstore
from app.config import Settings, get_settings
from app.llm import get_chat, get_embeddings

REFUSAL_TEXT = "I cannot find the answer to this question in the provided documents."
RECURSION_LIMIT = 25  # hard backstop on top of the loops counter


class GraphState(TypedDict):
    question: str
    query: str
    chunks: list[dict]
    grade: str
    loops: int
    answer: str
    found: bool
    citations: list[dict]
    trace: Annotated[list[str], operator.add]


def _message_text(message) -> str:
    """Normalize an LLM reply to plain text.

    Newer Gemini models return `content` as a list of typed blocks (text and
    thinking parts) rather than a plain string.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


def _extract_json(text: str) -> dict:
    """Parse a JSON object from an LLM reply, tolerating ``` fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in model reply: {text[:200]}")
    return json.loads(match.group(0))


class QAService:
    """Builds the graph once and answers questions through it."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.chat = get_chat(self.settings)
        self.embeddings = get_embeddings(self.settings)
        self.pc = vectorstore.get_client(self.settings)
        self.graph = self._build_graph()

    # ---- nodes -------------------------------------------------------------

    def _retrieve(self, state: GraphState) -> dict:
        query = state.get("query") or state["question"]
        vector = self.embeddings.embed_query(query)
        chunks = vectorstore.query(self.pc, self.settings, vector, self.settings.top_k)
        return {
            "chunks": chunks,
            "trace": [
                f"retrieve: query={query!r} -> {len(chunks)} chunks "
                f"{[c['chunk_id'] for c in chunks]}"
            ],
        }

    def _grade_chunks(self, state: GraphState) -> dict:
        context = "\n\n".join(
            f"[{c['chunk_id']}]\n{c['text']}" for c in state["chunks"]
        )
        prompt = (
            "You are grading retrieved document chunks for a Q&A system.\n"
            f"Question: {state['question']}\n\n"
            f"Retrieved chunks:\n{context or '(none)'}\n\n"
            "Can the question be answered using ONLY these chunks? "
            "Reply with exactly one word: sufficient or insufficient."
        )
        reply = _message_text(self.chat.invoke([HumanMessage(content=prompt)])).strip().lower()
        grade = "sufficient" if "insufficient" not in reply else "insufficient"
        if not state["chunks"]:
            grade = "insufficient"
        return {"grade": grade, "trace": [f"grade_chunks: {grade}"]}

    def _rewrite_query(self, state: GraphState) -> dict:
        prompt = (
            "Rewrite this question as a better search query for a small corpus of "
            "legal-style notes (contracts, hearing notices, statutes, settlement "
            "notes). Reply with the rewritten query only, no explanation.\n\n"
            f"Question: {state['question']}\n"
            f"Previous query: {state.get('query') or state['question']}"
        )
        new_query = _message_text(self.chat.invoke([HumanMessage(content=prompt)])).strip()
        loops = state.get("loops", 0) + 1
        return {
            "query": new_query,
            "loops": loops,
            "trace": [f"rewrite_query (loop {loops}): {new_query!r}"],
        }

    def _generate_answer(self, state: GraphState) -> dict:
        context = "\n\n".join(
            f"[{c['chunk_id']}] (from {c['source_file']})\n{c['text']}"
            for c in state["chunks"]
        )
        prompt = (
            "Answer the question using ONLY the document chunks below. Do not use "
            "any outside knowledge. If the chunks do not contain the answer, set "
            '"found" to false.\n\n'
            f"Chunks:\n{context}\n\n"
            f"Question: {state['question']}\n\n"
            "Reply with a single JSON object, no other text:\n"
            '{"found": true|false, "answer": "<concise answer>", '
            '"cited_chunk_ids": ["<chunk_id of every chunk you used>"]}'
        )
        reply = _message_text(self.chat.invoke([HumanMessage(content=prompt)]))
        try:
            data = _extract_json(reply)
        except (ValueError, json.JSONDecodeError):
            data = {"found": False, "answer": "", "cited_chunk_ids": []}
        found = bool(data.get("found"))
        return {
            "found": found,
            "answer": str(data.get("answer", "")).strip(),
            "citations": [{"chunk_id": cid} for cid in data.get("cited_chunk_ids", [])],
            "trace": [
                f"generate_answer: found={found}, "
                f"cited={[c for c in data.get('cited_chunk_ids', [])]}"
            ],
        }

    def _validate_citations(self, state: GraphState) -> dict:
        """Keep only citations that point at chunks actually retrieved."""
        retrieved = {c["chunk_id"]: c for c in state["chunks"]}
        valid = []
        for cite in state.get("citations", []):
            chunk = retrieved.get(cite.get("chunk_id"))
            if chunk:
                valid.append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "source_file": chunk["source_file"],
                        "section": chunk["section"],
                        "snippet": chunk["text"][:200],
                        "score": round(chunk["score"], 4),
                    }
                )
        dropped = len(state.get("citations", [])) - len(valid)
        if not state.get("found") or not valid:
            # No grounded citations -> refuse rather than risk a fabricated answer.
            return {
                "found": False,
                "answer": REFUSAL_TEXT,
                "citations": [],
                "trace": [
                    f"validate_citations: {len(valid)} valid, {dropped} dropped -> refusal"
                ],
            }
        return {
            "citations": valid,
            "trace": [f"validate_citations: {len(valid)} valid, {dropped} dropped"],
        }

    def _no_answer(self, state: GraphState) -> dict:
        return {
            "found": False,
            "answer": REFUSAL_TEXT,
            "citations": [],
            "trace": ["no_answer: retrieval loops exhausted -> refusal"],
        }

    # ---- routing -----------------------------------------------------------

    def _route_after_grade(self, state: GraphState) -> str:
        if state["grade"] == "sufficient":
            return "generate_answer"
        if state.get("loops", 0) < self.settings.max_retrieval_loops:
            return "rewrite_query"
        return "no_answer"

    # ---- graph -------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_chunks", self._grade_chunks)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_citations", self._validate_citations)
        graph.add_node("no_answer", self._no_answer)

        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_chunks")
        graph.add_conditional_edges(
            "grade_chunks",
            self._route_after_grade,
            {
                "generate_answer": "generate_answer",
                "rewrite_query": "rewrite_query",
                "no_answer": "no_answer",
            },
        )
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate_answer", "validate_citations")
        graph.add_edge("validate_citations", END)
        graph.add_edge("no_answer", END)
        return graph.compile()

    # ---- public API --------------------------------------------------------

    def ask(self, question: str) -> dict:
        initial: GraphState = {
            "question": question,
            "query": "",
            "chunks": [],
            "grade": "",
            "loops": 0,
            "answer": "",
            "found": False,
            "citations": [],
            "trace": [],
        }
        final = self.graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
        return {
            "answer": final["answer"],
            "found": final["found"],
            "citations": final["citations"],
            "trace": final["trace"],
        }
