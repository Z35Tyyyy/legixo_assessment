"""Markdown-aware chunking with deterministic chunk IDs.

Pure functions: no network calls, unit-testable offline.

Each corpus file is split on `##` headings. The block before the first `##`
(the `#` title plus any metadata lines) becomes its own "header" chunk. Chunk
IDs are deterministic — `<file-stem>::<section-slug>::<n>` — so re-running
ingestion upserts the same vector IDs instead of creating duplicates.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MAX_CHUNK_CHARS = 1500
OVERLAP_CHARS = 150


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    section: str
    text: str


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "section"


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown into (section_title, section_text) pairs on `##` headings."""
    lines = content.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "header"
    current_lines: list[str] = []
    for line in lines:
        if line.startswith("## "):
            if "".join(current_lines).strip():
                sections.append((current_title, current_lines))
            current_title = line[3:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)
    if "".join(current_lines).strip():
        sections.append((current_title, current_lines))
    return [(title, "\n".join(body).strip()) for title, body in sections]


def _split_oversized(text: str) -> list[str]:
    """Fallback size cap: split long sections into overlapping windows."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    parts = []
    start = 0
    while start < len(text):
        parts.append(text[start : start + MAX_CHUNK_CHARS])
        start += MAX_CHUNK_CHARS - OVERLAP_CHARS
    return parts

def chunk_file(path: Path) -> list[Chunk]:
    content = path.read_text(encoding="utf-8")
    stem = path.stem
    chunks: list[Chunk] = []
    for title, body in _split_sections(content):
        slug = _slugify(title)
        for n, part in enumerate(_split_oversized(body)):
            chunks.append(
                Chunk(
                    chunk_id=f"{stem}::{slug}::{n}",
                    source_file=path.name,
                    section=title,
                    text=part,
                )
            )
    return chunks


def chunk_corpus(corpus_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        chunks.extend(chunk_file(path))
    return chunks


def preview(chunks: Iterable[Chunk]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"{c.chunk_id}  ({len(c.text)} chars)  [{c.source_file}]")
    return "\n".join(lines)


if __name__ == "__main__":
    corpus = Path(__file__).resolve().parent.parent / "data" / "corpus"
    all_chunks = chunk_corpus(corpus)
    print(preview(all_chunks))
    print(f"\n{len(all_chunks)} chunks from {corpus}")
