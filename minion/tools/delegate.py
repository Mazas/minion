"""
minion/tools/delegate.py

Model delegation tool — spins up a specialist agent for a focused task.

The orchestrator (fast model) calls delegate_to_specialist() when a task
warrants deeper capability. The specialist runs with a role-specific system
prompt, no tools, and no conversation history — it's a pure task executor.

Each role has a focused system prompt tuned for that task type:
  reasoning — deep analysis, multi-step thinking, tradeoff comparison
  code      — writing, reviewing, and debugging code

Adding a new role: add an entry to ROLE_PROMPTS and configure the model
in ~/.minion/.env under MINION_DELEGATE_MODELS.
"""

from __future__ import annotations

from pydantic_ai import Agent

from minion.config import Config
from minion.llm.providers import get_provider

ROLE_PROMPTS: dict[str, str] = {
    "reasoning": """\
You are a deep reasoning specialist. You are given a specific task that requires
careful, thorough analysis. Think step by step. Be precise and complete.
Do not hedge unnecessarily — give your best answer with clear reasoning.
""",
    "code": """\
You are a code specialist. You write clean, correct, well-commented code.
When asked to write code: produce working code with brief explanations.
When asked to review code: identify real issues, not style nitpicks.
When asked to debug: diagnose the root cause and provide a fix.
Use the language/framework appropriate to the task unless specified.
""",
}

_DEFAULT_PROMPT = """\
You are a specialist assistant. Complete the given task thoroughly and accurately.
"""


async def run_delegate(
    role: str,
    task: str,
    context: str,
    config: Config,
) -> str:
    """
    Run a task on a specialist model and return the result as a string.

    Args:
        role: The specialist role (must be a key in config.delegate_models).
        task: Full self-contained task description.
        context: Additional context to help the specialist.
        config: App config (provides model name and Ollama endpoint).

    Returns:
        The specialist's response as a plain string, or an error message.
    """
    model_name = config.delegate_models.get(role)
    if not model_name:
        available = ", ".join(config.delegate_models.keys()) or "none configured"
        return (
            f"No specialist model configured for role {role!r}. "
            f"Available roles: {available}. "
            f"Add it to MINION_DELEGATE_MODELS in ~/.minion/.env."
        )

    system_prompt = ROLE_PROMPTS.get(role, _DEFAULT_PROMPT)
    provider = get_provider(config)
    model = provider.get_model(model_override=model_name)

    specialist: Agent[None, str] = Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=str,
    )

    full_task = task
    if context.strip():
        full_task = f"{task}\n\nAdditional context:\n{context}"

    result = await specialist.run(full_task)
    return result.output
