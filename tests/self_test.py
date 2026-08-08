"""Self-test suite: 13 questions against the running API.

Validates retrieval accuracy (answer contains expected facts), citation
accuracy (citations point at the expected source file, and are never
fabricated), and refusal behavior on out-of-scope questions.

Usage:
    1. Ingest the corpus and start the server (see README).
    2. python tests/self_test.py            # or: API_URL=... python tests/self_test.py

Exits non-zero if any case fails.
"""

import os
import sys
import time

import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# Gemini free tier allows ~10 requests/min; a refusal path uses several LLM
# calls. Pace the suite and back off on quota errors so it passes on free keys.
PAUSE_BETWEEN_CASES = float(os.getenv("SELF_TEST_PAUSE", "5"))
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF = 40  # seconds

# Each case: question, keywords expected in the answer (any-casing),
# expected source file for at least one citation, and whether the system
# should answer (True) or refuse (False).
CASES = [
    # -- positive: answers must come from the corpus with correct citations --
    {
        "q": "What is the notice period in the Bluecrest Analytics employment agreement?",
        "keywords": ["60 day"],
        "source": "02_employment_agreement_excerpt.md",
        "answerable": True,
    },
    {
        "q": "How long does the non-compete last after leaving Bluecrest Analytics?",
        "keywords": ["12 month"],
        "source": "02_employment_agreement_excerpt.md",
        "answerable": True,
    },
    {
        "q": "Which court is hearing Arvind Mehta v. Northfield Logistics?",
        "keywords": ["riverside"],
        "source": "01_matter_memo_arvind_v_northfield.md",
        "answerable": True,
    },
    {
        "q": "When was the written contract signed in the Northfield Logistics matter?",
        "keywords": ["12 january 2024"],
        "source": "01_matter_memo_arvind_v_northfield.md",
        "answerable": True,
    },
    {
        "q": "How many days before the listed date must written arguments be filed?",
        "keywords": ["seven"],
        "source": "03_hearing_notice_template.md",
        "answerable": True,
    },
    {
        "q": "What interest rate applies to delayed payments when a contract fixes no rate?",
        "keywords": ["9%"],
        "source": "04_statute_style_excerpt_fictional.md",
        "answerable": True,
    },
    {
        "q": "How long is mandatory mediation under Section 14 of the Riverside Commercial Courts Act?",
        "keywords": ["30 day"],
        "source": "04_statute_style_excerpt_fictional.md",
        "answerable": True,
    },
    {
        "q": "What percentage of open invoices did Northfield offer to pay in settlement?",
        "keywords": ["70"],
        "source": "05_counsel_notes_settlement.md",
        "answerable": True,
    },
    {
        "q": "What is the monthly rent for Unit 4B at Harbor View Tower?",
        "keywords": ["45,000"],
        "source": "06_property_lease_clause.md",
        "answerable": True,
    },
    {
        "q": "Is subletting allowed under the Harbor View Tower lease?",
        "keywords": ["not allowed", "without written consent"],
        "keywords_any": True,
        "source": "06_property_lease_clause.md",
        "answerable": True,
    },
    # -- negative: out-of-scope questions must be refused, with no citations --
    {"q": "What is the capital of France?", "answerable": False},
    {"q": "What does the Indian Contract Act, 1872 say about liquidated damages?", "answerable": False},
    {"q": "Who won the 2023 Cricket World Cup?", "answerable": False},
]


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
    resp = _post_with_backoff(case["q"])
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
    keywords = case.get("keywords", [])
    if case.get("keywords_any"):
        keyword_ok = any(k.lower() in lowered for k in keywords)
    else:
        keyword_ok = all(k.lower() in lowered for k in keywords)
    if not keyword_ok:
        return False, f"answer missing expected keyword(s) {keywords}: {answer[:120]!r}"
    if not citations:
        return False, "no citations returned for an answerable question"
    cited_files = {c["source_file"] for c in citations}
    if case["source"] not in cited_files:
        return False, f"expected citation from {case['source']}, got {sorted(cited_files)}"
    return True, f"ok ({len(citations)} citation(s) incl. {case['source']})"


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
        print(f"[{status}] {i:2d}. {case['q'][:70]}\n        {detail}")
        if i < len(CASES):
            time.sleep(PAUSE_BETWEEN_CASES)
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
