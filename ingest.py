"""Ingestion pipeline: corpus files -> chunks -> embeddings -> Pinecone.

Usage:
    python ingest.py            # idempotent: same chunk IDs are upserted in place
    python ingest.py --reset    # wipe the namespace first, then ingest fresh

Running ingest twice does NOT duplicate vectors: chunk IDs are deterministic
(`<file-stem>::<section-slug>::<n>`), so Pinecone overwrites existing records.
"""

import argparse
from pathlib import Path

from app import vectorstore
from app.chunking import chunk_corpus
from app.config import get_settings
from app.llm import get_embeddings

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the corpus into Pinecone.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete everything in the namespace before ingesting.",
    )
    args = parser.parse_args()

    settings = get_settings()
    chunks = chunk_corpus(CORPUS_DIR)
    if not chunks:
        raise SystemExit(f"No markdown files found in {CORPUS_DIR}")
    print(f"Chunked {CORPUS_DIR} -> {len(chunks)} chunks")

    pc = vectorstore.get_client(settings)
    vectorstore.ensure_index(pc, settings)
    print(f"Index '{settings.index_name}' ready (namespace '{settings.namespace}')")

    if args.reset:
        vectorstore.delete_namespace(pc, settings)
        print("Namespace cleared (--reset)")

    embeddings = get_embeddings(settings)
    vectors = embeddings.embed_documents([c.text for c in chunks])
    upserted = vectorstore.upsert_chunks(pc, settings, chunks, vectors)
    print(f"Upserted {upserted} vectors")

    for chunk in chunks:
        print(f"  {chunk.chunk_id}")


if __name__ == "__main__":
    main()
