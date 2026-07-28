from ..data import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

DEFAULT_KS = (1, 3, 5, 10)
DEFAULT_IOU_THRESHOLD = 0.05


def _iou(a: MinimalSource, b: MinimalSource) -> float:
    if a.file_path != b.file_path:
        return 0.0

    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    if intersection == 0:
        return 0.0

    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def _is_found(
    target: MinimalSource,
    retrieved: list[MinimalSource],
    threshold: float,
) -> bool:
    return any(_iou(target, source) >= threshold for source in retrieved)


def _question_recall(
    correct_sources: list[MinimalSource],
    retrieved_sources: list[MinimalSource],
    k: int,
    threshold: float,
) -> float:
    if not correct_sources:
        return 0.0

    top_k = retrieved_sources[:k]
    found = sum(
        1 for source in correct_sources if _is_found(source, top_k, threshold)
    )
    return found / len(correct_sources)


def evaluate_recall(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    ks: tuple[int, ...] = DEFAULT_KS,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> dict[int, float]:
    """Average recall@k, for each k, over every answered question."""
    ground_truth = {
        question.question_id: question.sources
        for question in dataset.rag_questions
        if isinstance(question, AnsweredQuestion)
    }
    retrieved_by_question = {
        result.question_id: result.retrieved_sources
        for result in student_results.search_results
    }

    recalls: dict[int, float] = {}
    for k in ks:
        scores = [
            _question_recall(
                sources,
                retrieved_by_question.get(question_id, []),
                k,
                iou_threshold,
            )
            for question_id, sources in ground_truth.items()
        ]
        recalls[k] = sum(scores) / len(scores) if scores else 0.0
    return recalls
