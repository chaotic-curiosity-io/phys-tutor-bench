"""Manages the model under test (the tutor being evaluated)."""

from __future__ import annotations

from typing import Protocol

import anthropic
import openai
import httpx

from src.scenarios.schema import Scenario
from src.engine.model_compat import (
    anthropic_create_kwargs,
    openai_create_kwargs,
    first_text,
)

DEFAULT_TUTOR_SYSTEM_PROMPT = """\
You are a physics tutor working with an introductory physics student. Your goal is to \
help the student develop genuine understanding, not just get the right answer.

## Pedagogical Guidelines
- Use Socratic questioning to guide the student toward discovering the correct reasoning
- Build on what the student already understands correctly (anchoring conceptions)
- Help the student distinguish between their correct intuitions and incorrect conclusions
- Scaffold from concrete examples to general principles
- Do NOT simply tell the student the answer — help them construct understanding
- If the student is wrong, help them see WHY their reasoning leads to a contradiction
- Be encouraging and validate the student's reasoning process even when correcting conclusions
- Aim for the student to build a transferable mental model, not just memorize this specific case

## Context
The student is working on a physics problem and has given their initial response. \
Your job is to help them develop correct understanding through dialogue.
"""


class TutorBackend(Protocol):
    """Protocol for tutor model backends."""

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        system_prompt: str,
    ) -> tuple[str, int]:
        """Generate a tutor response.

        Args:
            conversation_history: Messages where "user" = student, "assistant" = tutor.
            system_prompt: The tutor's system prompt.

        Returns:
            Tuple of (response_text, token_count).
        """
        ...


class AnthropicTutor:
    """Anthropic API backend for the tutor model under test."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or anthropic.Anthropic()

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        system_prompt: str,
    ) -> tuple[str, int]:
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=conversation_history,
            **anthropic_create_kwargs(self.model, self.temperature, self.max_tokens),
        )
        text = first_text(response.content)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return text, tokens


class OpenAITutor:
    """OpenAI API backend for the tutor model under test."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        client: openai.OpenAI | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = client or openai.OpenAI()

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        system_prompt: str,
    ) -> tuple[str, int]:
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **openai_create_kwargs(self.model, self.temperature, self.max_tokens),
        )
        text = response.choices[0].message.content
        tokens = (response.usage.prompt_tokens + response.usage.completion_tokens) if response.usage else 0
        return text, tokens


class GenericHTTPTutor:
    """Generic HTTP backend for any OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        timeout: float = 600.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def respond(
        self,
        conversation_history: list[dict[str, str]],
        system_prompt: str,
    ) -> tuple[str, int]:
        messages = [{"role": "system", "content": system_prompt}]
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        resp = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        return text, tokens


def create_tutor_backend(
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    api_base: str | None = None,
    api_key: str | None = None,
) -> TutorBackend:
    """Factory to create the appropriate tutor backend based on model name.

    An explicit ``api_base`` means "talk to this OpenAI-compatible endpoint" (e.g. a
    local Ollama server at http://localhost:11434). It takes precedence over model-name
    prefix matching so that locally-served models whose names collide with cloud prefixes
    (e.g. "gpt-oss") are routed to the local endpoint instead of a cloud provider.
    """
    if api_base:
        return GenericHTTPTutor(
            base_url=api_base,
            model=model,
            api_key=api_key or "",
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if model.startswith("claude-"):
        return AnthropicTutor(model=model, temperature=temperature, max_tokens=max_tokens)
    elif model.startswith("gpt-") or model.startswith(("o1", "o3", "o4", "o5")):
        return OpenAITutor(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        # Default to Anthropic
        return AnthropicTutor(model=model, temperature=temperature, max_tokens=max_tokens)
