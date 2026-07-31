"""Pydantic data models exchanged between the RAG pipeline's stages."""

from pydantic import BaseModel, ConfigDict, Field
from typing import List
import uuid


class MinimalSource(BaseModel):
    """A chunk's source location within the ingested corpus.

    Attributes:
        file_path: Path to the source file, relative to the project root.
        first_character_index: Start of the chunk's character span.
        last_character_index: End of the chunk's character span.
        chunk_content: The chunk's text. Internal-only: never part of the
            grader-facing JSON, since ground truth/output only carry the
            three fields above.
    """

    file_path: str
    first_character_index: int
    last_character_index: int
    chunk_content: str = ""


class UnansweredQuestion(BaseModel):
    """A question with no ground truth, to be answered by the pipeline.

    Attributes:
        question_id: Unique identifier for the question.
        question: The question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its ground-truth answer and supporting sources.

    Attributes:
        sources: The correct source locations for this question.
        answer: The reference answer text.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not, read from a JSON file.

    Attributes:
        rag_questions: The dataset's questions.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieval results for a single question.

    Attributes:
        question_id: Identifier of the question these results answer.
        question: The question text.
        retrieved_sources: Top-k retrieved source locations, ranked.
    """

    # The moulinette's own schema expects the JSON key "question_str", even
    # though the subject's pseudocode names the field "question" - alias it
    # so our attribute name matches the subject while our output matches the
    # real grader. populate_by_name keeps both spellings valid on input.
    model_config = ConfigDict(populate_by_name=True)

    question_id: str
    question: str = Field(alias="question_str")
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieval results for a question, plus a generated answer.

    Attributes:
        answer: The generated answer text.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Batch retrieval output written by the search_dataset command.

    Attributes:
        search_results: Per-question retrieval results.
        k: Number of results requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch retrieval-plus-answer output written by answer_dataset.

    Attributes:
        search_results: Per-question retrieval results with answers.
        k: Number of results requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int
