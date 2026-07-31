*This project has been created as part of the 42 curriculum by ekarout.*

# RAG against the machine

## Description

This project builds a Retrieval-Augmented Generation (RAG) system that answers
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The
vLLM source tree (docs and Python code) is ingested and split into chunks, a
searchable index is built over those chunks, and given a natural-language
question the system retrieves the most relevant source locations and
optionally generates a grounded answer with a small local LLM
(`Qwen/Qwen3-0.6B`).

The system is judged primarily on retrieval quality (recall@k against a
held-out set of questions with known correct sources), and secondarily on
whether generated answers stay faithful to the retrieved context.

## Instructions

### Install

```bash
make install        # uv sync
```

### Build the index

```bash
uv run python -m src index --max_chunk_size 2000
```

Chunks every markdown/Python file under `data/raw/`, builds a BM25 index, and
saves everything to `data/processed/`. Also attempts to build a semantic
(embedding) index for the bonus features - see [Bonus features](#bonus-features)
below; if that step fails (e.g. no network access), the mandatory BM25 index
is unaffected.

### Search / answer

```bash
# Single query
uv run python -m src search "How to configure the OpenAI server?" --k 10
uv run python -m src answer "How to configure the OpenAI server?" --k 10

# Whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# Local recall@k check (for your own iteration; official scoring uses the moulinette)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

All input/output paths are CLI arguments with sensible defaults - none are
hardcoded. See [Example usage](#example-usage) for more.

### Other Makefile targets

```bash
make run     # uv run python -m src index
make debug   # run under pdb
make clean   # remove __pycache__, .mypy_cache, .pytest_cache
make lint    # flake8 + mypy
```

## Resources

- [bm25s](https://github.com/xhluca/bm25s) - the BM25 implementation used for
  lexical retrieval.
- [Sentence-BERT / sentence-transformers](https://www.sbert.net/) - used for
  the bonus semantic embedding index (`all-MiniLM-L6-v2`).
- [Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and
  Beyond"](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) -
  background on BM25's term-saturation (`k1`) and length-normalization (`b`)
  parameters.
- [Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
  and individual rank learning methods"](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) -
  the fusion method used for hybrid retrieval.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) - the generation
  model, including its chat template and thinking-mode behavior.
- [Python Fire documentation](https://github.com/google/python-fire) - how a
  plain class's methods become CLI subcommands.
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/) - data model
  validation, aliasing (used to reconcile the subject's `question` field name
  with the moulinette's actual `question_str` JSON key).

### How AI was used

Development used an AI coding assistant (Claude Code) throughout, in an
interactive, human-reviewed loop - not as an unattended code generator:

- **Architecture and CLI wiring**: scaffolding the `src/` package layout,
  wiring Python Fire commands to the chunking/indexing/evaluation/generation
  modules.
- **Debugging**: diagnosing concrete failures by running the code and reading
  tracebacks - e.g. a chunker returning `(MinimalSource(...),)` 1-tuples
  instead of `MinimalSource(...)`, a `NameError`-riddled early draft of
  `retrieve_top_k`, and a schema mismatch where the real moulinette binary
  expects the JSON key `question_str` while the subject's own pseudocode
  names the field `question` (found by running the actual grader binary
  locally and reading its validation error, then fixed with a pydantic
  field alias).
  - Every fix was verified by re-running the affected command (and, for the
    retrieval commands, by running the real `moulinette-ubuntu` binary
    against the output) rather than taken on faith.
- **Evaluation logic**: implementing recall@k (IoU-based source matching) by
  translating the subject's written definition into code.
- **Bonus design**: discussing tradeoffs for the semantic/hybrid retrieval
  bonuses (embedding library choice, CLI shape, fusion algorithm) before
  implementation, and being explicit when a claim couldn't be verified end-to-end
  in the working environment (a network restriction blocked downloading the
  embedding model there) rather than asserting it worked.
- **Documentation**: docstrings, this README.

All generated code was reviewed, run, and in several cases corrected by hand
(or by asking the assistant to re-diagnose after seeing real output) before
being kept.

## System architecture

```
data/raw/ (vLLM source)
      |
      v
+-------------------+     +--------------------+
| src/chunking/      |     |                    |
|  text_chunker.py   |---->|  list[MinimalSource]
|  code_chunker.py   |     |  (chunk_content +   |
+-------------------+     |   file/char offsets) |
                           +----------+---------+
                                      |
                                      v
                      +---------------------------+
                      | src/indexing/              |
                      |  bm25_index.py    (mandatory, lexical)
                      |  embedding_index.py (bonus, semantic)
                      |  hybrid.py         (bonus, RRF fusion)
                      +--------------+--------------+
                                     |
                          data/processed/ (persisted indices)
                                     |
                                     v
                      +---------------------------+
                      | src/cli.py (Python Fire)    |
                      |  index / search / search_dataset
                      |  evaluate / answer / answer_dataset
                      +--------------+--------------+
                            |                  |
                            v                  v
              +----------------------+  +---------------------------+
              | src/evaluation/       |  | src/generation/            |
              |  recall.py            |  |  qwen_generator.py         |
              |  (local recall@k)     |  |  (Qwen/Qwen3-0.6B answers) |
              +----------------------+  +---------------------------+
```

Pipeline stages, matching the mandatory RAG stages (indexing, retrieving,
augmenting, generating):

1. **Ingestion & chunking** (`src/chunking/`) - `text_chunker` walks
   `data/raw/` for `.md` files and `code_chunker` for `.py` files, using
   LangChain's `RecursiveCharacterTextSplitter` (a Python-language-aware
   variant for code) to split into `MinimalSource` chunks with exact
   `(file_path, first_character_index, last_character_index)` offsets into
   the original file.
2. **Indexing** (`src/indexing/`) - `build_index` tokenizes chunk text and
   builds a BM25 index via `bm25s`, persisted to `data/processed/`.
   `build_embedding_index` (bonus) additionally embeds every chunk with a
   small sentence-transformer.
3. **Retrieval** (`src/indexing/bm25_index.py`, `embedding_index.py`,
   `hybrid.py`) - given a query, `retrieve_top_k` (BM25),
   `retrieve_top_k_semantic` (cosine similarity), or `retrieve_top_k_hybrid`
   (RRF fusion of both) returns the top-k `MinimalSource` results.
4. **Augmenting & generating** (`src/generation/qwen_generator.py`) -
   retrieved chunks are concatenated into a character-budgeted context block
   and passed to `Qwen/Qwen3-0.6B` via a chat template, which generates a
   grounded answer.
5. **Evaluation** (`src/evaluation/recall.py`) - compares a
   `StudentSearchResults` file against ground-truth `AnsweredQuestions` and
   reports recall@{1,3,5,10} for local iteration.

Everything is orchestrated by `src/cli.py`, a single `Cli` class whose
methods Python Fire exposes as `uv run python -m src <command>` subcommands.
Pydantic models (`src/data/models.py`) are the data contract passed between
every stage and written to/read from disk.

## Chunking strategy

Two distinct chunkers, both built on LangChain's
`RecursiveCharacterTextSplitter` with `chunk_overlap=200` and a configurable
`max_chunk_size` (default 2000 characters, matching the moulinette's
`max_context_length`):

- **`text_chunker`** (`.md` files): splits on a general-purpose recursive
  separator hierarchy (headers/paragraphs down to sentences/words), so
  chunks tend to break at natural section boundaries before falling back to
  a sliding window once a section exceeds `max_chunk_size`.
- **`code_chunker`** (`.py` files): uses the splitter's `Language.PYTHON`
  mode, which prefers to break along class/function boundaries before
  falling back to a sliding window - keeping a function's logic together
  more often than a naive fixed-size split would.

Both record `first_character_index`/`last_character_index` as offsets into
the *original file content* (via `add_start_index=True`), which is what
recall@k is scored against - not offsets into the chunk itself.

`chunk_content` is intentionally kept on `MinimalSource` for internal use
(BM25/embedding indexing, and rebuilding LLM context from a saved search
result), but is stripped from every JSON file written to disk, since the
grader's schema only expects `file_path`/`first_character_index`/
`last_character_index`.

## Retrieval method

**Mandatory: BM25** (`src/indexing/bm25_index.py`, via `bm25s`). BM25 scores
each chunk by term frequency, saturating via `k1` (default) so repeated terms
don't dominate, and normalizing by chunk length via `b` (default) so short
and long chunks are compared fairly. Both query and corpus are tokenized
identically (lowercased, English stopwords removed) so retrieval and
indexing agree on vocabulary. BM25 was chosen over TF-IDF because it
generally performs better on mixed code+prose corpora, matching the
subject's own hint.

**Bonus: semantic** (`src/indexing/embedding_index.py`) embeds every chunk
with `all-MiniLM-L6-v2` (L2-normalized), so retrieval ranks by cosine
similarity - this catches paraphrases that share no vocabulary with the
source text, which BM25 by construction cannot.

**Bonus: hybrid** (`src/indexing/hybrid.py`) fuses the two rankings with
Reciprocal Rank Fusion: each method contributes a candidate pool
(`max(k*4, 50)` results), and every chunk's fused score is
`sum(1 / (60 + rank))` across whichever ranking(s) it appears in. RRF was
chosen over a weighted score sum because BM25 scores and cosine similarities
live on incomparable scales - RRF only needs rank position, sidestepping
score-normalization tuning entirely.

All three are selectable via `--method bm25|semantic|hybrid` on
`search`/`search_dataset`/`answer` (default: `bm25`, so the mandatory-path
behavior is unaffected unless a bonus method is explicitly requested).

## Performance analysis

Measured with `uv run python -m src search_dataset` on the public datasets,
scored against the real moulinette binary
(`evaluate_student_search_results`), BM25 method, `k=10`:

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Threshold (Recall@5) |
|---|---|---|---|---|---|
| Docs | 0.640 | 0.760 | **0.810** | 0.880 | ≥ 0.80 - **PASS** |
| Code | 0.330 | 0.470 | **0.540** | 0.590 | ≥ 0.50 - **PASS** |

Timing (measured via `exams/scripts/exam_retrieval.sh` on the private
datasets, 100 questions each for docs and code):

| Stage | Measured | Budget |
|---|---|---|
| Indexing (whole corpus) | ~8s | ≤ 300s |
| Retrieval (200 questions total) | ~8s | ≤ 90s |

Both recall thresholds and both timing budgets are comfortably met. Code
recall is noticeably lower than docs recall, which matches the subject's own
framing: questions "rarely use the same words" as the code they're about
(e.g. a question phrased in English prose vs. an identifier or API name in
the source), which is harder for a purely lexical method like BM25 - part of
the motivation for the semantic/hybrid bonus retrieval methods.

Answer generation (`Qwen/Qwen3-0.6B`, CPU) is not covered by the timing
budgets above, but is noticeably slower - roughly 40-50s per question on a
CPU-only machine in testing, dominated by model inference rather than
retrieval or context assembly.

## Design decisions

- **Chunk metadata as the single source of truth**: `MinimalSource` carries
  `chunk_content` alongside the character offsets, so the exact same object
  flows through chunking, indexing, retrieval, and generation without ever
  needing to re-read a file to recover chunk text - except when reloading a
  previously-saved search result (see below).
- **Stripping `chunk_content` from saved JSON**: the grader's schema for
  `StudentSearchResults`/`StudentSearchResultsAndAnswer` only expects
  `file_path`/`first_character_index`/`last_character_index` per source.
  `chunk_content` is excluded at serialization time (`model_dump_json(...,
  exclude=...)`) so output stays minimal and grader-compatible, while still
  being available in memory for the same process's own use.
- **Re-hydrating context for `answer_dataset`**: since saved search results
  don't carry `chunk_content`, `answer_dataset` reads each source's exact
  character span back from its original file on disk before building LLM
  context - avoiding either bloating the saved JSON with extra fields or
  re-running retrieval just to get answer context.
- **`question`/`question_str` field alias**: the subject's own pseudocode
  names the field `question`, but the actual moulinette binary requires the
  JSON key `question_str` (confirmed by running it directly and reading its
  validation error). Resolved with a pydantic `Field(alias="question_str",
  ...)` plus `populate_by_name=True`, so the Python attribute matches the
  subject's model while the JSON on disk matches the real grader.
- **BM25 as the default, always-available method**: `--method` defaults to
  `bm25` everywhere, and `index` never lets a bonus embedding-build failure
  (e.g. no network access) take down the mandatory BM25 index - the
  mandatory path must work unconditionally, independent of the bonus
  dependencies being available.
- **RRF over weighted score fusion**: see [Retrieval method](#retrieval-method).

## Challenges faced

- **Reconciling the subject's pseudocode with the real grader.** The subject
  describes `MinimalSearchResults.question: str`, but the actual moulinette
  binary rejects that output, requiring `question_str`. This only surfaced
  by running the real grader against real output, not by reading the
  subject alone - worth remembering that the moulinette, not the written
  spec, is the ground truth for wire formats.
- **Keeping chunk text available without bloating saved output.** Early
  drafts either dropped chunk text entirely (breaking `answer_dataset`,
  which needs it to build LLM context) or leaked it into grader-facing JSON
  (risking schema-strictness issues). Settled on: keep `chunk_content` as an
  optional, internal-only `MinimalSource` field, always excluded from saved
  output, and reconstructed from disk when a saved result needs to become
  LLM context again.
- **A sandboxed development environment without live internet access to
  Hugging Face**, encountered while testing the semantic-embedding bonus.
  `all-MiniLM-L6-v2` (unlike `Qwen/Qwen3-0.6B`, which was already cached
  from earlier in development) could not be downloaded there. Handled by
  making `index` degrade gracefully - the mandatory BM25 build always
  succeeds independent of the embedding download's success - and by being
  explicit that the bonus's actual retrieval quality needs verification on
  a machine with working network access, rather than claiming it was
  verified when it wasn't.
- **CPU-only LLM generation is slow.** ~40-50s per answer means a full
  100-question `answer_dataset` run takes over an hour. Mitigated with a
  process-lifetime cache (`@lru_cache`) on the loaded model/tokenizer, so
  repeated calls in the same run only pay the load cost once, and by keeping
  `max_new_tokens` bounded (300) so individual generations don't run away.

## Example usage

```bash
# Build the index (BM25 + semantic, if available)
uv run python -m src index --max_chunk_size 2000

# Ask a single question
uv run python -m src search "How to configure the OpenAI server?" --k 5
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# Same, but forcing hybrid retrieval (bonus)
uv run python -m src search "How to configure the OpenAI server?" --k 5 --method hybrid

# Run retrieval over a full dataset and score it locally
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
# -> Recall@1: 0.640 Recall@3: 0.760 Recall@5: 0.810 Recall@10: 0.880

# Generate answers for a whole dataset from a saved search result
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## Bonus features

Implemented (see [Retrieval method](#retrieval-method) for details):

1. **Semantic embeddings** - `src/indexing/embedding_index.py`, a
   CPU-friendly `all-MiniLM-L6-v2` vector index built alongside the lexical
   one.
2. **Hybrid retrieval** - `src/indexing/hybrid.py`, Reciprocal Rank Fusion
   of the BM25 and semantic rankings.

Not implemented: incremental indexing, caching, and a local HTTP API.
