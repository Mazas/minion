"""
minion/agent/agent.py

The PydanticAI agent. Owns tool definitions and system prompt.

Agent dependencies (AgentDeps) carry runtime objects (e.g. MemoryManager,
SearchProvider) that tools need. PydanticAI injects deps via RunContext on
every tool call.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from minion.config import Config
from minion.llm.providers import get_provider
from minion.memory.manager import MemoryManager
from minion.tools.search import SearchProvider, format_results, get_search_provider

SYSTEM_PROMPT = """\
You are Minion, a personal AI assistant running locally on the user's machine.

## Personality
- Concise and direct. No unnecessary filler or sycophancy.
- Honest about uncertainty.
- Treat the user as a capable adult.

## Memory
You have access to a persistent memory system. Use it proactively:

- Call `recall_memories` at the start of EVERY conversation turn to surface
  relevant context before responding. Even a short query may have relevant history.
- Call `store_memory` whenever the user shares:
  - Personal facts (name, location, occupation, etc.)
  - Preferences ("I prefer X over Y")
  - Ongoing projects or goals
  - Any context that would make future responses more useful
- Do NOT ask the user if you should remember something. Just do it silently.
- When you store a memory, briefly acknowledge it (e.g. "Got it, I'll remember that.").

Memory types:
  fact        — objective facts about the user
  preference  — stated preferences and opinions
  project     — ongoing work or goals
  context     — situational/temporary context

## Web Search
You have access to a web search tool. Use it when:
- The user asks about current events, news, or recent information.
- The user asks a factual question you're uncertain about or that may have changed.
- The user explicitly asks you to search or look something up.

Do NOT search for things you know well (general programming concepts, history,
stable facts). Do NOT mention the search tool by name. Just use it and cite the
source URLs naturally in your response.

## Tools
Use tools when they genuinely help. Don't mention a tool by name to the user.
"""


@dataclass
class AgentDeps:
    """Runtime dependencies injected into every tool call."""
    memory: MemoryManager
    search: SearchProvider


def create_agent(config: Config, memory: MemoryManager) -> Agent[AgentDeps, str]:
    """
    Build and return the configured PydanticAI agent with all tools.
    """
    provider = get_provider(config)
    model = provider.get_model()

    agent: Agent[AgentDeps, str] = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        deps_type=AgentDeps,
        output_type=str,
    )

    # ── Memory tools ──────────────────────────────────────────────────────

    @agent.tool
    async def store_memory(
        ctx: RunContext[AgentDeps],
        content: str,
        type: str = "fact",
        tags: list[str] | None = None,
    ) -> str:
        """
        Store a piece of information about the user for future reference.

        Args:
            content: The information to remember. Be specific and self-contained
                     (should make sense when read back without conversation context).
            type: Category — one of: fact, preference, project, context.
            tags: Keywords that help retrieve this memory later (e.g. ["python", "tools"]).
        """
        memory = await ctx.deps.memory.remember(content, type=type, tags=tags)
        return f"Stored memory #{memory.id}: {memory.content}"

    @agent.tool
    async def recall_memories(
        ctx: RunContext[AgentDeps],
        query: str,
        limit: int = 5,
    ) -> str:
        """
        Search stored memories relevant to a query. Call this before responding
        to any message that might benefit from personal context.

        Args:
            query: What you're looking for (e.g. "user preferences", "current project").
            limit: Max number of memories to return (default 5).
        """
        memories = await ctx.deps.memory.recall(query, limit=limit)
        return MemoryManager.format_for_context(memories)

    @agent.tool
    async def forget_memory(
        ctx: RunContext[AgentDeps],
        memory_id: int,
    ) -> str:
        """
        Delete a stored memory by its ID. Use when the user asks to forget
        something or a memory is clearly outdated/wrong.

        Args:
            memory_id: The numeric ID of the memory to delete.
        """
        removed = await ctx.deps.memory.forget(memory_id)
        return f"Memory #{memory_id} deleted." if removed else f"Memory #{memory_id} not found."

    @agent.tool
    async def update_memory(
        ctx: RunContext[AgentDeps],
        memory_id: int,
        content: str,
    ) -> str:
        """
        Update the content of an existing memory.

        Args:
            memory_id: The numeric ID of the memory to update.
            content: The new content to replace the old entry.
        """
        updated = await ctx.deps.memory.update(memory_id, content)
        return f"Memory #{memory_id} updated." if updated else f"Memory #{memory_id} not found."

    # ── Search tool ───────────────────────────────────────────────────────

    if config.enable_web_search:

        @agent.tool
        async def web_search(
            ctx: RunContext[AgentDeps],
            query: str,
            limit: int = 5,
        ) -> str:
            """
            Search the web for current information. Use for recent events, news,
            or facts you're uncertain about. Returns titles, URLs, and snippets.

            Args:
                query: The search query. Be specific for better results.
                limit: Number of results to return (default 5, max 10).
            """
            results = await ctx.deps.search.search(query, limit=min(limit, 10))
            return format_results(results)

    return agent
