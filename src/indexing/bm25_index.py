"""BM25 indexing and retrieval over chunked MinimalSource corpora."""

import bm25s
from ..data import MinimalSource


def build_index(
    chunks: list[MinimalSource],
    save_dir: str = "data/processed",
) -> bm25s.BM25:
    """Tokenize, index, and persist a corpus of chunks with BM25.

    Args:
        chunks: The chunks to index (from text_chunker/code_chunker).
        save_dir: Directory to persist the index and corpus to.

    Returns:
        The built BM25 retriever.
    """
    texts = [chunk.chunk_content for chunk in chunks]
    corpus = [
        {
            "text": chunk.chunk_content,
            "file_path": chunk.file_path,
            "first_character_index": chunk.first_character_index,
            "last_character_index": chunk.last_character_index,
        }
        for chunk in chunks
    ]

    corpus_tokens = bm25s.tokenize(texts, stopwords="en", show_progress=True)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens, show_progress=True)
    retriever.save(save_dir, corpus=corpus)

    return retriever


def retrieve_top_k(
    query: str,
    retriever: bm25s.BM25,
    k: int = 10,
) -> list[MinimalSource]:
    """Return the top-k chunks for a single query, ranked by BM25 score.

    Args:
        query: The query text.
        retriever: A BM25 index loaded with its corpus.
        k: Number of top results to return.

    Returns:
        Up to k MinimalSource results, or an empty list for a blank query,
        k <= 0, or an empty corpus.
    """
    if not query or not query.strip() or k <= 0:
        return []
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    top_k = min(k, len(retriever.corpus))
    if top_k <= 0:
        return []
    results, _ = retriever.retrieve(query_tokens, k=top_k, show_progress=False)
    return [
        MinimalSource(
            chunk_content=doc["text"],
            file_path=doc["file_path"],
            first_character_index=doc["first_character_index"],
            last_character_index=doc["last_character_index"],
        )
        for doc in results[0]
    ]
