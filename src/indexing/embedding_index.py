from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from ..data import MinimalSource

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDINGS_FILENAME = "embeddings.npy"


@lru_cache(maxsize=1)
def _load_embedding_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    """Load and cache the sentence-transformer used for embeddings."""
    return cast(SentenceTransformer, SentenceTransformer(model_name))


def build_embedding_index(
    chunks: list[MinimalSource],
    save_dir: str = "data/processed",
) -> None:
    """Embed every chunk and persist the resulting vectors.

    Vectors are L2-normalized and saved in the same order as `chunks`, so
    row i lines up with the BM25 corpus entry built from the same chunk
    list (see build_index) - no separate metadata file is needed.

    Args:
        chunks: The chunks to embed (from text_chunker/code_chunker).
        save_dir: Directory to persist the embeddings matrix to.
    """
    model = _load_embedding_model()
    texts = [chunk.chunk_content for chunk in chunks]
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.save(Path(save_dir) / _EMBEDDINGS_FILENAME, embeddings)


def load_embedding_index(save_dir: str = "data/processed") -> np.ndarray:
    """Load a previously built embeddings matrix from disk.

    Args:
        save_dir: Directory the embeddings were saved to.

    Returns:
        The (n_chunks, dim) float32 embeddings matrix.

    Raises:
        FileNotFoundError: If no embeddings file exists at save_dir.
    """
    path = Path(save_dir) / _EMBEDDINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    return cast(np.ndarray, np.load(path))


def retrieve_top_k_semantic(
    query: str,
    embeddings: np.ndarray,
    corpus: list[dict[str, Any]],
    k: int = 10,
) -> list[MinimalSource]:
    """Return the top-k chunks for a query, ranked by cosine similarity.

    Args:
        query: The query text.
        embeddings: The (n_chunks, dim) embeddings matrix from
            build_embedding_index, aligned with corpus.
        corpus: Chunk metadata dicts, aligned with embeddings (as saved
            alongside a BM25 index built from the same chunk list).
        k: Number of top results to return.

    Returns:
        Up to k MinimalSource results, or an empty list for a blank query,
        k <= 0, or an empty corpus.
    """
    if not query or not query.strip() or k <= 0 or len(embeddings) == 0:
        return []

    model = _load_embedding_model()
    query_vector = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )[0]

    top_k = min(k, len(embeddings))
    similarities = embeddings @ query_vector
    top_indices = np.argsort(-similarities)[:top_k]

    return [
        MinimalSource(
            chunk_content=corpus[i]["text"],
            file_path=corpus[i]["file_path"],
            first_character_index=corpus[i]["first_character_index"],
            last_character_index=corpus[i]["last_character_index"],
        )
        for i in top_indices
    ]
