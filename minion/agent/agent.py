"""
minion/agent/agent.py

The PydanticAI agent. This is the core of minion — it receives messages,
decides what to do (including tool calls), and streams responses back.

In Milestone 1 there are no tools yet. They'll be registered in later milestones
by importing and decorating functions with @agent.tool.
"""

from __future__ import annotations

from pydantic_ai import Agent

from minion.config import Config
from minion.llm.providers import get_provider

SYSTEM_PROMPT = """\
You are Minion, a personal AI assistant running locally on the user's machine.

Your personality:
- Concise and direct. No unnecessary filler.
- Honest about what you don't know.
- Helpful without being sycophantic.

Capabilities you will gain over time:
- Remembering facts, preferences, and context about the user across sessions.
- Searching the web when current information is needed.
- Reading and writing files.
- Running shell commands.
- Working with git repositories.

For now, focus on being a sharp conversational assistant.
"""


def create_agent(config: Config) -> Agent[None, str]:
    """
    Build and return the configured PydanticAI agent.

    The agent is stateless — conversation history is managed externally
    by the session layer and passed in on each call. This keeps the agent
    pure and makes testing straightforward.
    """
    provider = get_provider(config)
    model = provider.get_model()

    agent: Agent[None, str] = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        output_type=str,
    )

    return agent
