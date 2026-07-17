"""
minion/agent/session.py

Manages a single conversation session: holds message history and provides
a clean interface for the TUI to send messages and receive streamed chunks.

Yields typed StreamEvent objects so the TUI can render thinking and text
differently — thinking appears immediately as dimmed text, text as normal.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ThinkingPart, TextPart

from minion.agent.agent import AgentDeps
from minion.memory.manager import MemoryManager
from minion.tools.search import SearchProvider


@dataclass
class StreamEvent:
    """A single streamed chunk from the agent, tagged by type."""
    kind: Literal["thinking", "text"]
    content: str


class Session:
    """
    Wraps an Agent with conversation history.

    PydanticAI agents are stateless — they accept a message_history list on
    each call. The Session owns that list and persists it to SQLite after
    each exchange so conversations survive restarts.
    """

    def __init__(
        self,
        agent: Agent[AgentDeps, str],
        memory: MemoryManager,
        search: SearchProvider,
        session_id: int,
        initial_history: list[ModelMessage] | None = None,
    ) -> None:
        self._agent = agent
        self._memory = memory
        self._search = search
        self._session_id = session_id
        self._history: list[ModelMessage] = initial_history or []

    @classmethod
    async def create(
        cls,
        agent: Agent[AgentDeps, str],
        memory: MemoryManager,
        search: SearchProvider,
        resume: bool = True,
    ) -> "Session":
        """
        Factory: creates or resumes a session.

        If resume=True and a previous session exists, loads its history.
        Otherwise starts a fresh session.
        """
        if resume:
            session_id = await memory.get_latest_session_id()
            if session_id is not None:
                history = await memory.load_messages(session_id)
                return cls(agent, memory, search, session_id, history)

        session_id = await memory.create_session()
        return cls(agent, memory, search, session_id)

    async def stream(self, user_message: str) -> AsyncIterator[StreamEvent]:
        """
        Send a user message and yield StreamEvents (thinking + text chunks).

        stream_response() yields full ModelResponse snapshots on each token.
        Each snapshot contains ALL content accumulated so far in the current
        model cycle. When content shrinks, we've entered a new cycle (tool call
        completed) and reset our offset tracking accordingly.
        """
        deps = AgentDeps(memory=self._memory, search=self._search)
        async with self._agent.run_stream(
            user_message,
            deps=deps,
            message_history=self._history,
        ) as streamed:
            seen_thinking = 0
            seen_text = 0

            async for response in streamed.stream_response(debounce_by=0.05):
                for part in response.parts:
                    if isinstance(part, ThinkingPart):
                        current_len = len(part.content)
                        if current_len < seen_thinking:
                            seen_thinking = 0
                        delta = part.content[seen_thinking:]
                        if delta:
                            yield StreamEvent(kind="thinking", content=delta)
                        seen_thinking = current_len

                    elif isinstance(part, TextPart):
                        current_len = len(part.content)
                        if current_len < seen_text:
                            seen_text = 0
                        delta = part.content[seen_text:]
                        if delta:
                            yield StreamEvent(kind="text", content=delta)
                        seen_text = current_len

            self._history.extend(streamed.new_messages())

        # Persist after every exchange so history survives crashes
        await self._memory.save_messages(self._session_id, self._history)

    def clear(self) -> None:
        """Reset conversation history (starts a new logical thread)."""
        self._history = []

    @property
    def session_id(self) -> int:
        return self._session_id

    @property
    def message_count(self) -> int:
        return len(self._history)
