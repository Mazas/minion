"""
minion/llm/providers.py

LLM provider abstraction. Currently implements Ollama via its OpenAI-compatible
endpoint. To add a cloud provider later, implement the LLMProvider Protocol and
wire it up in get_provider().
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from minion.config import Config


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can hand back a PydanticAI Model is a valid provider."""

    def get_model(self, model_override: str | None = None) -> Model: ...


class OllamaProvider:
    """
    Connects to a local Ollama instance using its OpenAI-compatible REST API.

    model_override lets callers (e.g. the delegation tool) request a specific
    model without creating a new provider instance.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def get_model(self, model_override: str | None = None) -> Model:
        model_name = model_override or self._config.orchestrator_model
        provider = OpenAIProvider(
            base_url=self._config.ollama_base_url,
            api_key=self._config.ollama_api_key,
        )
        return OpenAIChatModel(model_name, provider=provider)


def get_provider(config: Config) -> OllamaProvider:
    """
    Factory — returns the configured provider.
    Extend this with an if/elif chain as cloud providers are added.
    """
    return OllamaProvider(config)
