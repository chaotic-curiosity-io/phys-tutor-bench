"""Per-model API parameter compatibility.

The original Ollama-era backends sent the classic sampling parameters
(``temperature``) on every request. Newer frontier models reject them:

* Anthropic Opus 4.8 / 4.7 and Fable / Mythos 5 reject ``temperature`` (HTTP 400).
* OpenAI reasoning models (GPT-5.x, o-series) reject ``temperature`` and require
  ``max_completion_tokens`` instead of ``max_tokens``.

These helpers build the right request kwargs for a given model so the tutor,
student, and judge backends can stay model-agnostic. Models that still accept
the classic params (``claude-sonnet-4-6``, ``claude-haiku-4-5``, ``gpt-4o``,
Ollama-served models) are unaffected.
"""

from __future__ import annotations


# Anthropic model families that reject sampling params (temperature/top_p/top_k).
_ANTHROPIC_NO_SAMPLING = ("opus-4-8", "opus-4-7", "fable-5", "mythos-5", "mythos-preview")


def anthropic_rejects_sampling(model: str) -> bool:
    """True if the Anthropic model rejects sampling params (temperature, etc.)."""
    m = model.lower()
    return any(tag in m for tag in _ANTHROPIC_NO_SAMPLING)


def openai_is_reasoning(model: str) -> bool:
    """True for OpenAI reasoning models (GPT-5.x, o-series).

    These reject ``temperature`` and use ``max_completion_tokens`` rather than
    ``max_tokens``. ``gpt-4o`` and other classic chat models return False.
    """
    m = model.lower()
    if m.startswith(("o1", "o3", "o4", "o5")):
        return True
    if m.startswith("gpt-5") or m.startswith("gpt5"):
        return True
    return False


def anthropic_create_kwargs(model: str, temperature: float, max_tokens: int) -> dict:
    """Kwargs for ``client.messages.create`` (besides model/system/messages)."""
    kwargs: dict = {"max_tokens": max_tokens}
    if not anthropic_rejects_sampling(model):
        kwargs["temperature"] = temperature
    return kwargs


def openai_create_kwargs(model: str, temperature: float, max_tokens: int) -> dict:
    """Kwargs for ``client.chat.completions.create`` (besides model/messages)."""
    if openai_is_reasoning(model):
        # Reasoning models: no temperature, and the output cap moved to
        # max_completion_tokens (which also funds the hidden reasoning tokens).
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens, "temperature": temperature}


def first_text(content_blocks) -> str:
    """Return the text of the first text block in an Anthropic response.

    Robust to a model leading with a non-text (e.g. thinking) block, which can
    happen when adaptive thinking is on. Falls back to any block exposing
    ``.text``.
    """
    for block in content_blocks:
        if getattr(block, "type", None) == "text":
            return block.text
        text = getattr(block, "text", None)
        if text is not None:
            return text
    raise ValueError("No text block found in response content")
