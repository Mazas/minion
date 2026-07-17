"""
minion/llm/embeddings.py

Async wrapper around Ollama's /api/embed endpoint.

Returns a 768-dimensional float vector for a given text string.
Falls back gracefully if the embed model is not available — callers
receive None and should fall back to FTS5-only recall.

No new Python dependencies — uses httpx which is already installed.
"""

from __future__ import annotations

import httpx

# TODO: migrate to sqlite-vec for O(1) similarity search when memory
#       count grows large (hundreds → thousands). Pure Python cosine
#       similarity is fast enough for personal use today.

_EMBED_TIMEOUT = 10.0  # seconds


async def embed(
    text: str,
    model: str = "nomic-embed-text",
    base_url: str = "http://localhost:11434",
) -> list[float] | None:
    """
    Generate an embedding vector for the given text.

    Returns None (silently) if the model is unavailable or the request fails,
    so callers can degrade gracefully to FTS5-only recall.
    """
    try:
        async with httpx.AsyncClient(timeout=_EMBED_TIMEOUT) as client:
            resp = await client.post(
                f"{base_url}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            embeddings = data.get("embeddings")
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return None
    except Exception:
        return None


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns a value in [-1, 1]; higher is more similar.
    Pure Python — fast enough for hundreds of memories.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
