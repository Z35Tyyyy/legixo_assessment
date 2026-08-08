"""Pinecone integration: index lifecycle, upserts, and queries.

Uses the real Pinecone service (serverless index) as required by the brief.
"""

import time

from pinecone import Pinecone, ServerlessSpec

from app.chunking import Chunk
from app.config import Settings

UPSERT_BATCH_SIZE = 100


def get_client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index(pc: Pinecone, settings: Settings) -> None:
    """Create the serverless index if it does not exist yet."""
    existing = {idx["name"] for idx in pc.list_indexes()}
    if settings.index_name in existing:
        return
    pc.create_index(
        name=settings.index_name,
        dimension=settings.embed_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.cloud, region=settings.region),
    )
    # Wait until the index is ready before returning.
    while not pc.describe_index(settings.index_name).status["ready"]:
        time.sleep(1)


def delete_namespace(pc: Pinecone, settings: Settings) -> None:
    index = pc.Index(settings.index_name)
    try:
        index.delete(delete_all=True, namespace=settings.namespace)
    except Exception:
        # Namespace may not exist yet on a fresh index — nothing to delete.
        pass


def upsert_chunks(
    pc: Pinecone, settings: Settings, chunks: list[Chunk], vectors: list[list[float]]
) -> int:
    """Upsert chunks with their embeddings. Deterministic IDs make this idempotent."""
    index = pc.Index(settings.index_name)
    records = [
        {
            "id": chunk.chunk_id,
            "values": vector,
            "metadata": {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section": chunk.section,
                "text": chunk.text,
            },
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(
            vectors=records[start : start + UPSERT_BATCH_SIZE],
            namespace=settings.namespace,
        )
    return len(records)


def query(pc: Pinecone, settings: Settings, vector: list[float], top_k: int) -> list[dict]:
    """Query the index; returns matches as plain dicts with score + metadata."""
    index = pc.Index(settings.index_name)
    result = index.query(
        vector=vector,
        top_k=top_k,
        namespace=settings.namespace,
        include_metadata=True,
    )
    return [
        {
            "chunk_id": m["id"],
            "score": float(m["score"]),
            "source_file": m["metadata"].get("source_file", ""),
            "section": m["metadata"].get("section", ""),
            "text": m["metadata"].get("text", ""),
        }
        for m in result.get("matches", [])
    ]


def vector_count(pc: Pinecone, settings: Settings) -> int:
    stats = pc.Index(settings.index_name).describe_index_stats()
    ns = stats.get("namespaces", {}).get(settings.namespace)
    return int(ns["vector_count"]) if ns else 0
