from functools import lru_cache
from pathlib import Path
import bm25s
import numpy as np
from tqdm import tqdm
from .chunking import code_chunker, text_chunker
from .data import (
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from .evaluation import evaluate_recall
from .generation import generate_answer
from .indexing import (
    build_embedding_index,
    build_index,
    load_embedding_index,
    retrieve_top_k,
    retrieve_top_k_hybrid,
    retrieve_top_k_semantic,
)

_RETRIEVED_SOURCES_EXCLUDE = {
    "retrieved_sources": {"__all__": {"chunk_content"}}
}
_SEARCH_RESULTS_EXCLUDE = {
    "search_results": {"__all__": _RETRIEVED_SOURCES_EXCLUDE}
}

_RETRIEVAL_METHODS = {"bm25", "semantic", "hybrid"}


@lru_cache(maxsize=None)
def _read_file_text(file_path: str) -> str:
    """Read and cache a file's full text, keyed by path.

    Args:
        file_path: Path to the file to read.

    Returns:
        The file's full text content.
    """
    return Path(file_path).read_text()


def _hydrate_source(source: MinimalSource) -> MinimalSource:
    """Re-attach chunk_content by reading the span from its original file.

    Saved search results only carry file_path/first/last character indices
    (the grader-facing schema), so answer generation needs to look the
    actual text back up before it can build context for the model.
    """
    try:
        text = _read_file_text(source.file_path)
        content = text[
            source.first_character_index:source.last_character_index
        ]
    except OSError:
        content = ""
    return source.model_copy(update={"chunk_content": content})


def _load_retrieval_backends(
    index_dir: str, method: str
) -> tuple[bm25s.BM25, np.ndarray | None] | None:
    """Load the BM25 index and, for semantic/hybrid, the embeddings matrix.

    Prints a friendly message and returns None on any failure (unknown
    method, missing BM25 index, missing embeddings), so callers can bail
    out gracefully instead of crashing.

    Args:
        index_dir: Directory the index(es) were saved to.
        method: One of "bm25", "semantic", "hybrid".

    Returns:
        (retriever, embeddings) on success - embeddings is None for
        method="bm25" - or None on failure.
    """
    if method not in _RETRIEVAL_METHODS:
        print(
            f"Unknown method '{method}'. Choose one of: "
            f"{', '.join(sorted(_RETRIEVAL_METHODS))}."
        )
        return None

    try:
        retriever = bm25s.BM25.load(index_dir, load_corpus=True)
    except FileNotFoundError:
        print(f"No index found at {index_dir}. Run `index` first.")
        return None

    if method == "bm25":
        return retriever, None

    try:
        embeddings = load_embedding_index(index_dir)
    except FileNotFoundError:
        print(f"No embedding index found at {index_dir}. Run `index` first.")
        return None

    return retriever, embeddings


def _retrieve_sources(
    method: str,
    query: str,
    retriever: bm25s.BM25,
    embeddings: np.ndarray | None,
    k: int,
) -> list[MinimalSource]:
    """Dispatch retrieval to the bm25/semantic/hybrid backend.

    Args:
        method: One of "bm25", "semantic", "hybrid".
        query: The query text.
        retriever: A BM25 index loaded with its corpus.
        embeddings: The embeddings matrix, required for semantic/hybrid.
        k: Number of top results to return.

    Returns:
        Up to k MinimalSource results.
    """
    if method == "semantic":
        if embeddings is None:
            raise ValueError("semantic retrieval requires embeddings")
        return retrieve_top_k_semantic(query, embeddings, retriever.corpus, k)
    if method == "hybrid":
        if embeddings is None:
            raise ValueError("hybrid retrieval requires embeddings")
        return retrieve_top_k_hybrid(
            query, retriever, embeddings, retriever.corpus, k
        )
    return retrieve_top_k(query, retriever, k)


class Cli:
    """Fire-exposed commands for indexing, retrieving, and answering."""

    def index(
        self,
        repo_path: str = "data/raw/",
        max_chunk_size: int = 2000,
        save_dir: str = "data/processed",
    ) -> None:
        """Chunk the repository and persist BM25 and semantic indices.

        Args:
            repo_path: Directory to ingest markdown/code files from.
            max_chunk_size: Maximum characters per chunk.
            save_dir: Directory to write the indices and corpus to.
        """
        text_chunks = text_chunker(repo_path, max_chunk_size)
        code_chunks = code_chunker(repo_path, max_chunk_size)
        print(f"Total markdown chunks created: {len(text_chunks)}")
        print(f"Total code chunks created: {len(code_chunks)}")

        chunks = text_chunks + code_chunks
        build_index(chunks, save_dir)
        print(f"BM25 index saved to {save_dir}")

        # The embedding index is a bonus feature (semantic/hybrid search).
        # Its failure - e.g. no network access to download the model -
        # must never take down the mandatory BM25 index built above.
        try:
            build_embedding_index(chunks, save_dir)
            print(f"Embedding index saved to {save_dir}")
        except Exception as e:
            print(
                "Skipping embedding index (semantic/hybrid search will be "
                f"unavailable): {e}"
            )

    def search(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        method: str = "bm25",
    ) -> None:
        """Print the top-k retrieved sources for a single query.

        Args:
            query: The question to search for.
            k: Number of top results to retrieve.
            index_dir: Directory the index(es) were saved to.
            method: Retrieval method - "bm25", "semantic", or "hybrid".
        """
        loaded = _load_retrieval_backends(index_dir, method)
        if loaded is None:
            return
        retriever, embeddings = loaded

        sources = _retrieve_sources(method, query, retriever, embeddings, k)
        result = MinimalSearchResults(
            question_id="",
            question=query,
            retrieved_sources=sources,
        )
        output = result.model_dump_json(
            indent=2, by_alias=True, exclude=_RETRIEVED_SOURCES_EXCLUDE
        )
        print(output)

    def search_dataset(
        self,
        dataset_path: str = (
            "data/datasets/UnansweredQuestions/dataset_docs_public.json"
        ),
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = "data/processed",
        method: str = "bm25",
    ) -> None:
        """Run search over every question in a dataset and save the results.

        Args:
            dataset_path: Path to a RagDataset-shaped JSON file.
            k: Number of top results to retrieve per question.
            save_directory: Directory to write the StudentSearchResults
                JSON to, under the same filename as dataset_path.
            index_dir: Directory the index(es) were saved to.
            method: Retrieval method - "bm25", "semantic", or "hybrid".
        """
        try:
            dataset_text = Path(dataset_path).read_text()
            dataset = RagDataset.model_validate_json(dataset_text)
        except (OSError, ValueError) as e:
            print(f"Could not read dataset at {dataset_path}: {e}")
            return

        loaded = _load_retrieval_backends(index_dir, method)
        if loaded is None:
            return
        retriever, embeddings = loaded

        results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=_retrieve_sources(
                    method, question.question, retriever, embeddings, k
                ),
            )
            for question in tqdm(dataset.rag_questions)
        ]

        output = StudentSearchResults(search_results=results, k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / Path(dataset_path).name
        out_path.write_text(
            output.model_dump_json(
                by_alias=True, exclude=_SEARCH_RESULTS_EXCLUDE
            )
        )
        print(f"Saved student_search_results to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report recall@k of saved search results against ground truth.

        This is for local iteration only; the official recall@k used for
        grading is computed by the moulinette, not by this command.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file, as written by search_dataset.
            dataset_path: Path to the matching AnsweredQuestions JSON file.
        """
        try:
            student_text = Path(student_search_results_path).read_text()
            student_results = StudentSearchResults.model_validate_json(
                student_text
            )
        except (OSError, ValueError) as e:
            print(
                "Could not read student results at "
                f"{student_search_results_path}: {e}"
            )
            return

        try:
            dataset_text = Path(dataset_path).read_text()
            dataset = RagDataset.model_validate_json(dataset_text)
        except (OSError, ValueError) as e:
            print(f"Could not read dataset at {dataset_path}: {e}")
            return

        recalls = evaluate_recall(student_results, dataset)
        print(
            " ".join(f"Recall@{k}: {recalls[k]:.3f}" for k in sorted(recalls))
        )

    def answer(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
        method: str = "bm25",
    ) -> None:
        """Retrieve context for a single query and print a generated answer.

        Args:
            query: The question to answer.
            k: Number of top results to retrieve as context.
            index_dir: Directory the index(es) were saved to.
            method: Retrieval method - "bm25", "semantic", or "hybrid".
        """
        loaded = _load_retrieval_backends(index_dir, method)
        if loaded is None:
            return
        retriever, embeddings = loaded

        sources = _retrieve_sources(method, query, retriever, embeddings, k)
        answer_text = generate_answer(query, sources)
        print(answer_text)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for every question in a saved search results file.

        Args:
            student_search_results_path: Path to a StudentSearchResults
                JSON file, as written by search_dataset.
            save_directory: Directory to write the
                StudentSearchResultsAndAnswer JSON to, under the same
                filename as student_search_results_path.
        """
        try:
            student_text = Path(student_search_results_path).read_text()
            student_results = StudentSearchResults.model_validate_json(
                student_text
            )
        except (OSError, ValueError) as e:
            print(
                "Could not read student results at "
                f"{student_search_results_path}: {e}"
            )
            return

        answers = []
        for result in tqdm(student_results.search_results):
            hydrated_sources = [
                _hydrate_source(source) for source in result.retrieved_sources
            ]
            answer_text = generate_answer(result.question, hydrated_sources)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )

        output = StudentSearchResultsAndAnswer(
            search_results=answers, k=student_results.k
        )

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / Path(student_search_results_path).name
        out_path.write_text(
            output.model_dump_json(
                by_alias=True, exclude=_SEARCH_RESULTS_EXCLUDE
            )
        )
        print(f"Saved student_search_results_and_answer to {out_path}")
