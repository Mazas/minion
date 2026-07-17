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

    def get_model(self) -> Model: ...


class OllamaProvider:
    """
    Connects to a local Ollama instance using its OpenAI-compatible REST API.

    Ollama serves at http://localhost:11434/v1 by default. The OpenAI client
    requires an API key field even for local models, so we pass a dummy value.
    """

    def __init__(self, config: Config) -> None:
        self._config = config

    def get_model(self) -> Model:
        provider = OpenAIProvider(
            base_url=self._config.ollama_base_url,
            api_key=self._config.ollama_api_key,
        )
        return OpenAIChatModel(self._config.model, provider=provider)


def get_provider(config: Config) -> LLMProvider:
    """
    Factory — returns the configured provider.
    Extend this with an if/elif chain as cloud providers are added.
    """
    return OllamaProvider(config)
