from pathlib import Path

import bm25s
from tqdm import tqdm

from .chunking import code_chunker, text_chunker
from .data import MinimalSearchResults, RagDataset, StudentSearchResults
from .evaluation import evaluate_recall
from .indexing import build_index, retrieve_top_k

# Output JSON must only carry the minimal MinimalSource fields the grader
# expects (file_path, first_character_index, last_character_index) -
# chunk_content is an internal-only field used for indexing/answer context.
_RETRIEVED_SOURCES_EXCLUDE = {
    "retrieved_sources": {"__all__": {"chunk_content"}}
}
_SEARCH_RESULTS_EXCLUDE = {
    "search_results": {"__all__": _RETRIEVED_SOURCES_EXCLUDE}
}


class Cli:
    def index(
        self,
        repo_path: str = "data/raw/",
        max_chunk_size: int = 2000,
        save_dir: str = "data/processed",
    ) -> None:
        text_chunks = text_chunker(repo_path, max_chunk_size)
        code_chunks = code_chunker(repo_path, max_chunk_size)
        print(f"Total markdown chunks created: {len(text_chunks)}")
        print(f"Total code chunks created: {len(code_chunks)}")

        build_index(text_chunks + code_chunks, save_dir)
        print(f"Index saved to {save_dir}")

    def search(
        self,
        query: str,
        k: int = 10,
        index_dir: str = "data/processed",
    ) -> None:
        try:
            retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        except FileNotFoundError:
            print(f"No index found at {index_dir}. Run `index` first.")
            return

        sources = retrieve_top_k(query, retriever, k)
        result = MinimalSearchResults(
            question_id="",
            question=query,
            retrieved_sources=sources,
        )
        output = result.model_dump_json(
            indent=2, exclude=_RETRIEVED_SOURCES_EXCLUDE
        )
        print(output)

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
        index_dir: str = "data/processed",
    ) -> None:
        try:
            dataset_text = Path(dataset_path).read_text()
            dataset = RagDataset.model_validate_json(dataset_text)
        except (OSError, ValueError) as e:
            print(f"Could not read dataset at {dataset_path}: {e}")
            return

        try:
            retriever = bm25s.BM25.load(index_dir, load_corpus=True)
        except FileNotFoundError:
            print(f"No index found at {index_dir}. Run `index` first.")
            return

        results = [
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=retrieve_top_k(
                    question.question, retriever, k
                ),
            )
            for question in tqdm(dataset.rag_questions)
        ]

        output = StudentSearchResults(search_results=results, k=k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / Path(dataset_path).name
        out_path.write_text(
            output.model_dump_json(exclude=_SEARCH_RESULTS_EXCLUDE)
        )
        print(f"Saved student_search_results to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
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
