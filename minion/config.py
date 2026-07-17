"""
minion/config.py

App-wide configuration loaded from environment variables and ~/.minion/.env.
All fields are typed. Add new settings here — never hardcode values elsewhere.
"""

from __future__ import annotations

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

    # LLM
    model: str = Field(default="qwen3:8b", description="Ollama model name")
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Ollama OpenAI-compatible endpoint",
    )
    # A dummy key is required by the OpenAI client even for local models
    ollama_api_key: str = Field(default="ollama")

    # Storage
    data_dir: Path = Field(
        default=Path.home() / ".minion",
        description="Directory for database and local config",
    )

    # Feature flags — tools can be toggled without code changes
    enable_web_search: bool = True
    enable_shell: bool = False  # off by default; opt-in for safety
    enable_filesystem: bool = True
    enable_git: bool = True

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_path(cls, v: object) -> Path:
        return Path(str(v)).expanduser().resolve()

    @property
    def db_path(self) -> Path:
        return self.data_dir / "minion.db"

    def ensure_data_dir(self) -> None:
        """Create the data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Module-level singleton — import this everywhere.
config = Config()
