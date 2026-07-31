from typing import Any

import bm25s
import numpy as np

from ..data import MinimalSource
from .bm25_index import retrieve_top_k
from .embedding_index import retrieve_top_k_semantic

_RRF_K = 60


def _source_key(source: MinimalSource) -> tuple[str, int, int]:
    return (
        source.file_path,
        source.first_character_index,
        source.last_character_index,
    )


def retrieve_top_k_hybrid(
    query: str,
    retriever: bm25s.BM25,
    embeddings: np.ndarray,
    corpus: list[dict[str, Any]],
    k: int = 10,
) -> list[MinimalSource]:
    """Return the top-k chunks for a query, fusing BM25 and semantic rank.

    Each method contributes a candidate pool (larger than k, so a chunk
    ranked highly by only one method can still surface), scored by
    Reciprocal Rank Fusion: 1 / (_RRF_K + rank) per ranking it appears in.

    Args:
        query: The query text.
        retriever: A BM25 index loaded with its corpus.
        embeddings: The embeddings matrix from build_embedding_index,
            aligned with corpus.
        corpus: Chunk metadata dicts, aligned with embeddings.
        k: Number of top results to return.

    Returns:
        Up to k MinimalSource results, or an empty list for a blank query
        or k <= 0.
    """
    if not query or not query.strip() or k <= 0:
        return []

    pool_size = max(k * 4, 50)
    bm25_results = retrieve_top_k(query, retriever, pool_size)
    semantic_results = retrieve_top_k_semantic(
        query, embeddings, corpus, pool_size
    )

    scores: dict[tuple[str, int, int], float] = {}
    sources_by_key: dict[tuple[str, int, int], MinimalSource] = {}
    for ranking in (bm25_results, semantic_results):
        for rank, source in enumerate(ranking, start=1):
            key = _source_key(source)
            scores[key] = scores.get(key, 0.0) + 1.0 / (_RRF_K + rank)
            sources_by_key.setdefault(key, source)

    ranked_keys = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [sources_by_key[key] for key in ranked_keys[:k]]
