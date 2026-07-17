"""
minion/agent/session.py

Manages a single conversation session: holds message history and provides
a clean interface for the TUI to send messages and receive streamed chunks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from minion.agent.agent import AgentDeps
from minion.memory.manager import MemoryManager


class Session:
    """
    Wraps an Agent with conversation history.

    PydanticAI agents are stateless — they accept a message_history list on
    each call. The Session owns that list so the TUI doesn't have to.
    """

    def __init__(self, agent: Agent[AgentDeps, str], memory: MemoryManager) -> None:
        self._agent = agent
        self._memory = memory
        self._history: list[ModelMessage] = []

    async def stream(self, user_message: str) -> AsyncIterator[str]:
        """
        Send a user message and yield streamed text chunks.
        Updates internal history after the response completes.
        """
        deps = AgentDeps(memory=self._memory)
        async with self._agent.run_stream(
            user_message,
            deps=deps,
            message_history=self._history,
        ) as streamed:
            async for chunk in streamed.stream_text(delta=True):
                yield chunk

            # Persist the full exchange (request + response) into history
            self._history.extend(streamed.new_messages())

    def clear(self) -> None:
        """Reset conversation history."""
        self._history = []

    @property
    def message_count(self) -> int:
        return len(self._history)
