from functools import lru_cache
from typing import cast

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ..data import MinimalSource

MODEL_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_MAX_NEW_TOKENS = 300

_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a codebase. "
    "Answer using ONLY the information in the provided context. "
    "If the context does not contain the answer, say so explicitly. "
    "Be concise, coherent, and mention relevant file paths when useful."
)


@lru_cache(maxsize=1)
def load_generator(
    model_name: str = MODEL_NAME,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load and cache the tokenizer/model used for answer generation.

    Args:
        model_name: HuggingFace Hub identifier of the causal LM to load.

    Returns:
        The (tokenizer, model) pair, cached across calls.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def _build_context(
    sources: list[MinimalSource], max_context_chars: int
) -> str:
    """Concatenate retrieved chunks into a character-budgeted context block.

    Args:
        sources: Retrieved sources, ranked, to draw context from.
        max_context_chars: Maximum total characters of context to include.

    Returns:
        Newline-separated blocks of "[file_path]\\ncontent", stopping once
        max_context_chars would be exceeded.
    """
    blocks = []
    used = 0
    for source in sources:
        block = f"[{source.file_path}]\n{source.chunk_content}"
        if used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)


def generate_answer(
    question: str,
    sources: list[MinimalSource],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Generate a grounded natural-language answer from retrieved sources.

    Args:
        question: The question to answer.
        sources: Retrieved sources to use as context.
        max_context_chars: Maximum total characters of context to include.
        max_new_tokens: Maximum number of tokens to generate.

    Returns:
        The generated answer text, or "" if question is blank.
    """
    if not question or not question.strip():
        return ""

    tokenizer, model = load_generator()
    context = _build_context(sources, max_context_chars)
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = cast(
        BatchEncoding,
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ),
    )
    output = cast(
        torch.Tensor,
        model.generate(  # type: ignore[operator]
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        ),
    )
    generated = output[0][inputs["input_ids"].shape[-1]:]
    decoded = cast(
        str, tokenizer.decode(generated, skip_special_tokens=True)
    )
    return decoded.strip()
