import bm25s
from ..data import MinimalSource


def build_index(
    chunks: list[MinimalSource],
    save_dir: str = "data/processed",
) -> bm25s.BM25:
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
