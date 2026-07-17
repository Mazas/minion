"""
minion/agent/agent.py

The PydanticAI agent. Owns tool definitions and system prompt.

Agent dependencies (AgentDeps) carry runtime objects (e.g. MemoryManager,
SearchProvider) that tools need. PydanticAI injects deps via RunContext on
every tool call.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext

from minion.config import Config
from minion.llm.providers import get_provider
from minion.memory.manager import MemoryManager
from minion.tools.delegate import run_delegate
from minion.tools.filesystem import file_read, file_write, list_dir
from minion.tools.git import git_branches, git_commit, git_diff, git_log, git_status
from minion.tools.search import SearchProvider, format_results, get_search_provider
from minion.tools.shell import BlockedCommandError, shell_exec

SYSTEM_PROMPT = """\
You are Minion, a personal AI assistant running locally on the user's machine.

## Personality
- Concise and direct. No unnecessary filler or sycophancy.
- Honest about uncertainty.
- Treat the user as a capable adult.

## Memory
You have access to a persistent memory system. Use it proactively:

- Call `recall_memories` when the user's message is personal, references past
  context, or would benefit from knowing their preferences or projects.
  Skip recall for purely factual or technical questions with no personal angle
  (e.g. "how does X work", "write a function that does Y").
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

## Filesystem
When the user asks you to read, write, or list files:
- Use `read_file` to read file contents.
- Use `list_directory` to explore directory structure.
- Use `write_file` with confirm=False first to show a dry-run preview,
  then confirm with the user before calling again with confirm=True.
- Always show the resolved path so the user knows exactly what was accessed.

## Shell
Shell access is powerful and potentially dangerous. Follow these rules strictly:
- Only run commands the user explicitly asks for.
- For any command that writes, deletes, or modifies state, call `run_shell`
  with confirm=False first to show the user the exact command, then wait for
  explicit approval before calling with confirm=True.
- Read-only commands (ls, cat, grep, git status, etc.) may run directly.
- Never chain destructive commands with &&.

## Git
When working with git repositories:
- Use `git_status` to see the current state before anything else.
- Use `git_log` to show recent commits.
- Use `git_diff` to show unstaged or staged changes.
- Use `git_branches` to list branches.
- Use `git_commit` with confirm=False first to preview, then confirm=True only
  after the user explicitly approves the commit message and staged changes.
- Never stage files or modify git config — only read state and commit what is
  already staged.

## Delegation
You are a fast coordinator. Handle simple tasks directly. Delegate to specialists
when a task genuinely warrants deeper capability:

- role="reasoning": complex multi-step analysis, comparing tradeoffs, deep
  explanations, tasks requiring careful step-by-step thinking
- role="code": writing, reviewing, or debugging code

Rules:
- Only delegate if you are confident the task requires specialist capability.
  Do NOT delegate simple questions, memory operations, or web searches.
- Include ALL relevant context in the task description — the specialist has
  no conversation history and no tools.
- After the specialist responds, present the result directly to the user.
  Do not summarise or restate unless the user asks for it.
- Inform the user you're getting a specialist to handle it (e.g. "Let me
  get our reasoning specialist on this...") before delegating.

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
    Uses config.orchestrator_model as the main model.
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
        importance: int = 3,
    ) -> str:
        """
        Store a piece of information about the user for future reference.

        Args:
            content: The information to remember. Be specific and self-contained
                     (should make sense when read back without conversation context).
            type: Category — one of: fact, preference, project, context.
            tags: Keywords that help retrieve this memory later (e.g. ["python", "tools"]).
            importance: How important is this memory? 1=low, 3=normal, 5=critical.
                        High importance (>=4) memories are never auto-decayed.
        """
        memory = await ctx.deps.memory.remember(content, type=type, tags=tags, importance=importance)
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

    # ── Filesystem tools ──────────────────────────────────────────────────

    if config.enable_filesystem:

        @agent.tool
        async def read_file(
            ctx: RunContext[AgentDeps],
            path: str,
        ) -> str:
            """
            Read the contents of a file. Returns up to 32 KB of text.

            Args:
                path: Path to the file (absolute, relative, or ~ expanded).
            """
            return await asyncio.to_thread(file_read, path)

        @agent.tool
        async def write_file(
            ctx: RunContext[AgentDeps],
            path: str,
            content: str,
            confirm: bool = False,
        ) -> str:
            """
            Write text content to a file. Creates parent directories as needed.

            Always call with confirm=False first to show the user a preview.
            Only call with confirm=True after the user explicitly approves.

            Args:
                path: Path to the file (absolute, relative, or ~ expanded).
                content: Text content to write.
                confirm: Set True only after user confirms the write.
            """
            return await asyncio.to_thread(file_write, path, content, confirm)

        @agent.tool
        async def list_directory(
            ctx: RunContext[AgentDeps],
            path: str = ".",
        ) -> str:
            """
            List the contents of a directory.

            Args:
                path: Directory path (defaults to current working directory).
            """
            return await asyncio.to_thread(list_dir, path)

    # ── Shell tool ────────────────────────────────────────────────────────

    if config.enable_shell:

        @agent.tool
        async def run_shell(
            ctx: RunContext[AgentDeps],
            command: str,
            workdir: str | None = None,
            confirm: bool = False,
        ) -> str:
            """
            Run a shell command and return its output.

            For commands that modify state (write, delete, move files, etc.):
            call with confirm=False first to show the user the exact command,
            then call again with confirm=True only after explicit approval.

            Read-only commands (ls, cat, grep, git status, etc.) can use
            confirm=True directly.

            Args:
                command: The shell command to execute.
                workdir: Working directory (defaults to cwd).
                confirm: Must be True to actually run the command.
            """
            if not confirm:
                return (
                    f"Dry run — would execute:\n\n  $ {command}\n\n"
                    "Call again with confirm=True to actually run this command."
                )
            try:
                return await shell_exec(command, workdir=workdir)
            except BlockedCommandError as e:
                return f"Blocked: {e}"

    # ── Git tools ─────────────────────────────────────────────────────────

    if config.enable_git:

        @agent.tool
        async def git_status_tool(
            ctx: RunContext[AgentDeps],
            cwd: str | None = None,
        ) -> str:
            """
            Show the working tree status of a git repository.

            Args:
                cwd: Path to the repo (defaults to current directory).
            """
            return await git_status(cwd=cwd)

        @agent.tool
        async def git_log_tool(
            ctx: RunContext[AgentDeps],
            cwd: str | None = None,
            limit: int = 10,
        ) -> str:
            """
            Show recent commit history (one line per commit).

            Args:
                cwd: Path to the repo (defaults to current directory).
                limit: Number of commits to return (default 10).
            """
            return await git_log(cwd=cwd, limit=limit)

        @agent.tool
        async def git_diff_tool(
            ctx: RunContext[AgentDeps],
            cwd: str | None = None,
            staged: bool = False,
            path: str | None = None,
        ) -> str:
            """
            Show changes in the working tree or staging area.

            Args:
                cwd: Path to the repo (defaults to current directory).
                staged: If True, show staged changes instead of unstaged.
                path: Limit diff to a specific file or directory.
            """
            return await git_diff(cwd=cwd, staged=staged, path=path)

        @agent.tool
        async def git_branches_tool(
            ctx: RunContext[AgentDeps],
            cwd: str | None = None,
        ) -> str:
            """
            List local branches and show the current branch.

            Args:
                cwd: Path to the repo (defaults to current directory).
            """
            return await git_branches(cwd=cwd)

        @agent.tool
        async def git_commit_tool(
            ctx: RunContext[AgentDeps],
            message: str,
            cwd: str | None = None,
            confirm: bool = False,
        ) -> str:
            """
            Commit staged changes. Always call with confirm=False first to
            preview what will be committed, then confirm=True after approval.

            Args:
                message: Commit message.
                cwd: Path to the repo (defaults to current directory).
                confirm: Must be True to actually commit.
            """
            return await git_commit(message=message, cwd=cwd, confirm=confirm)

    # ── Delegation tool ───────────────────────────────────────────────────

    if config.delegate_models:

        @agent.tool
        async def delegate_to_specialist(
            ctx: RunContext[AgentDeps],
            role: str,
            task: str,
            context: str = "",
        ) -> str:
            """
            Delegate a task to a specialist model with deeper capability.

            Use for tasks that genuinely require specialist expertise:
            - role="reasoning": complex analysis, multi-step reasoning, tradeoffs
            - role="code": writing, reviewing, or debugging code

            The specialist has NO conversation history and NO tools. Include all
            relevant context in the task description.

            Args:
                role: Specialist role — one of the configured delegate roles.
                task: Complete self-contained task description.
                context: Any additional context that helps the specialist.
            """
            return await run_delegate(role=role, task=task, context=context, config=config)

    return agent
