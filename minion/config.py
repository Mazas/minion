"""
minion/config.py

App-wide configuration loaded from environment variables and ~/.minion/.env.
All fields are typed. Add new settings here — never hardcode values elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MINION_",
        env_file=Path.home() / ".minion" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM — orchestrator is the fast coordinator model
    orchestrator_model: str = Field(
        default="qwen3:4b",
        description="Fast orchestrator model (handles routing + simple tasks)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible endpoint",
    )
    ollama_api_key: str = Field(default="ollama")

    # Delegation — maps role names to Ollama model names
    # Override via: MINION_DELEGATE_MODELS='{"reasoning":"qwen3:8b","code":"qwen2.5-coder:7b"}'
    delegate_models: dict[str, str] = Field(
        default={"reasoning": "qwen3:8b", "code": "qwen2.5-coder:7b"},
        description="Specialist model map: role -> ollama model name",
    )

    # Storage
    data_dir: Path = Field(
        default=Path.home() / ".minion",
        description="Directory for database and local config",
    )

    # Feature flags
    enable_web_search: bool = True
    enable_shell: bool = False
    enable_filesystem: bool = True
    enable_git: bool = True

    # Search
    search_provider: str = Field(default="duckduckgo", description="Search provider to use")

    # Embeddings + semantic memory
    embed_model: str = Field(default="nomic-embed-text", description="Ollama embedding model")
    enable_semantic_search: bool = Field(default=True, description="Enable vector recall alongside FTS5")

    # Memory decay (days without recall before soft-deletion)
    memory_decay_context_days: int = Field(default=30, description="Context memory decay threshold (days)")
    memory_decay_project_days: int = Field(default=90, description="Project memory decay threshold (days)")

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    @field_validator("delegate_models", mode="before")
    @classmethod
    def parse_delegate_models(cls, v: object) -> dict[str, str]:
        if isinstance(v, str):
            return json.loads(v)
        return v  # type: ignore[return-value]

    @property
    def db_path(self) -> Path:
        return self.data_dir / "minion.db"

    @property
    def ollama_embed_base_url(self) -> str:
        # Strip /v1 suffix — embed endpoint is on the base URL
        return self.ollama_base_url.removesuffix("/v1")

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere.
config = Config()
