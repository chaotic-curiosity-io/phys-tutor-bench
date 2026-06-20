"""Tests for the per-model API parameter compatibility shim."""

from types import SimpleNamespace

import pytest

from src.engine.model_compat import (
    anthropic_rejects_sampling,
    openai_is_reasoning,
    anthropic_create_kwargs,
    openai_create_kwargs,
    first_text,
)


class TestAnthropicSampling:
    @pytest.mark.parametrize("model", [
        "claude-opus-4-8", "claude-opus-4-7", "claude-fable-5", "claude-mythos-5",
    ])
    def test_flagships_reject_sampling(self, model):
        assert anthropic_rejects_sampling(model)
        kwargs = anthropic_create_kwargs(model, 0.7, 2048)
        assert "temperature" not in kwargs
        assert kwargs["max_tokens"] == 2048

    @pytest.mark.parametrize("model", [
        "claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-6",
        "claude-sonnet-4-20250514",
    ])
    def test_others_keep_temperature(self, model):
        assert not anthropic_rejects_sampling(model)
        kwargs = anthropic_create_kwargs(model, 0.7, 1024)
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 1024


class TestOpenAISampling:
    @pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5", "gpt-5.5-2026-04-23", "o1", "o3-mini", "o4"])
    def test_reasoning_models(self, model):
        assert openai_is_reasoning(model)
        kwargs = openai_create_kwargs(model, 0.0, 16000)
        assert kwargs == {"max_completion_tokens": 16000}
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs

    @pytest.mark.parametrize("model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1"])
    def test_classic_chat_models(self, model):
        assert not openai_is_reasoning(model)
        kwargs = openai_create_kwargs(model, 0.7, 2048)
        assert kwargs["max_tokens"] == 2048
        assert kwargs["temperature"] == 0.7
        assert "max_completion_tokens" not in kwargs


class TestFirstText:
    def test_plain_text_block(self):
        blocks = [SimpleNamespace(type="text", text="hello")]
        assert first_text(blocks) == "hello"

    def test_skips_leading_thinking_block(self):
        blocks = [SimpleNamespace(type="thinking", thinking="reasoning..."),
                  SimpleNamespace(type="text", text="answer")]
        assert first_text(blocks) == "answer"

    def test_raises_when_no_text(self):
        with pytest.raises(ValueError):
            first_text([SimpleNamespace(type="thinking", thinking="x")])
