"""Self-test suite: 13 questions against the running API.

Validates retrieval accuracy (answer contains expected facts), citation
accuracy (citations point at the expected source file, and are never
fabricated), and refusal behavior on out-of-scope questions.

Usage:
    1. Ingest the corpus and start the server (see README).
    2. python tests/self_test.py            # or: API_URL=... python tests/self_test.py

Exits non-zero if any case fails.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Gemini free tier allows ~10 requests/min; a refusal path uses several LLM
# calls. Pace the suite and back off on quota errors so it passes on free keys.
PAUSE_BETWEEN_CASES = float(os.getenv("SELF_TEST_PAUSE", "5"))
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF = 40  # seconds

# Cases live in tests/self_test_cases.json (question, expected citation files,
# expected answer keywords, answerable flag, and notes from the last run).
CASES_FILE = Path(__file__).resolve().parent / "self_test_cases.json"
CASES = json.loads(CASES_FILE.read_text(encoding="utf-8"))


def _post_with_backoff(question: str) -> requests.Response:
    """POST /ask, waiting out free-tier rate limits (429/RESOURCE_EXHAUSTED)."""
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        resp = requests.post(
            f"{API_URL}/ask",
            json={"question": question, "include_trace": False},
            timeout=180,
        )
        rate_limited = resp.status_code == 502 and (
            "RESOURCE_EXHAUSTED" in resp.text or "429" in resp.text
        )
        if not rate_limited or attempt == RATE_LIMIT_RETRIES:
            return resp
        print(f"        (rate limited, waiting {RATE_LIMIT_BACKOFF}s...)")
        time.sleep(RATE_LIMIT_BACKOFF)
    return resp


def run_case(case: dict) -> tuple[bool, str]:
    resp = _post_with_backoff(case["question"])
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    body = resp.json()
    answer = body.get("answer", "")
    citations = body.get("citations", [])

    if not case["answerable"]:
        if body.get("found"):
            return False, f"expected refusal, got answer: {answer[:120]!r}"
        if citations:
            return False, f"refusal must have no citations, got {len(citations)}"
        return True, "refused correctly"

    lowered = answer.lower()
    keywords = case.get("expected_answer_keywords", [])
    if case.get("keywords_any"):
        keyword_ok = any(k.lower() in lowered for k in keywords)
    else:
        keyword_ok = all(k.lower() in lowered for k in keywords)
    if not keyword_ok:
        return False, f"answer missing expected keyword(s) {keywords}: {answer[:120]!r}"
    if not citations:
        return False, "no citations returned for an answerable question"
    cited_files = {c["source_file"] for c in citations}
    expected_files = set(case.get("expected_citation_files", []))
    if expected_files and not (expected_files & cited_files):
        return False, (
            f"expected citation from {sorted(expected_files)}, got {sorted(cited_files)}"
        )
    return True, f"ok ({len(citations)} citation(s) from {sorted(cited_files)})"


def main() -> None:
    print(f"Running {len(CASES)} self-test cases against {API_URL}\n")
    failures = 0
    for i, case in enumerate(CASES, 1):
        try:
            passed, detail = run_case(case)
        except requests.RequestException as exc:
            passed, detail = False, f"request failed: {exc}"
        status = "PASS" if passed else "FAIL"
        if not passed:
            failures += 1
        print(f"[{status}] {i:2d}. {case['question'][:70]}\n        {detail}")
        if i < len(CASES):
            time.sleep(PAUSE_BETWEEN_CASES)
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
