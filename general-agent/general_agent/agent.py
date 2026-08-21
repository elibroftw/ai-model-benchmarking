"""Core agent loop.

Minimal tool-calling loop over LiteLLM. Each step:
1. Send the running message history to the model with the tool schemas.
2. If the model calls tools, execute them and append their results.
3. If the model returns a plain assistant message, treat it as the final
   answer and stop.

This is intentionally small — enrich it (streaming, retries, planning steps,
token accounting, per-step callbacks) as real tasks demand.
"""
from __future__ import annotations

import json
from typing import Any

import litellm

from .skills import SkillRegistry
from .tools import Tool, ToolRegistry


DEFAULT_SYSTEM_PROMPT = """You are a focused agent. You solve the task the user gives you and nothing more. Do not refactor, redesign, or improve things you were not asked about.

You have access to *skills* — focused guidance documents on how to handle specific classes of sub-task without wasting effort on incidentals. When your current sub-task matches a skill's trigger, call the `read_skill` tool to load its full guidance BEFORE acting.

Available skills:
{skills_index}
"""


class Agent:
    def __init__(
        self,
        model: str,
        skills: SkillRegistry | None = None,
        tools: ToolRegistry | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
    ):
        self.model = model
        self.skills = skills or SkillRegistry()
        self.tools = tools or ToolRegistry()
        self.system_prompt_template = system_prompt
        self.max_steps = max_steps

        # Every agent gets `read_skill` for free.
        if self.tools.get("read_skill") is None:
            self.tools.register(
                Tool(
                    name="read_skill",
                    description=(
                        "Load the full text of a skill by name. Call this "
                        "BEFORE acting when your current sub-task matches a "
                        "skill's trigger."
                    ),
                    params_schema={
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The skill's kebab-case name.",
                            },
                        },
                        "required": ["name"],
                    },
                    handler=lambda name: self.skills.read(name),
                )
            )

    def _system_prompt(self) -> str:
        return self.system_prompt_template.format(
            skills_index=self.skills.index_text()
        )

    def run(self, task: str, extra_messages: list[dict] | None = None) -> dict:
        """Run the agent to completion on a single task.

        Returns a dict with `final_answer` (str|None), `messages` (full
        trace), and `stop_reason` (`"final"` | `"max_steps"` | `"error"`).
        """
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": task},
        ]
        if extra_messages:
            messages.extend(extra_messages)

        for step in range(self.max_steps):
            response = litellm.completion(
                model=self.model,
                messages=messages,
                tools=self.tools.to_openai_schemas() or None,
                tool_choice="auto" if self.tools.names() else None,
            )
            msg = response.choices[0].message

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                messages.append({"role": "assistant", "content": msg.content or ""})
                return {
                    "final_answer": msg.content,
                    "messages": messages,
                    "stop_reason": "final",
                    "steps": step + 1,
                }

            messages.append(_assistant_tool_call_message(msg, tool_calls))
            for call in tool_calls:
                tool = self.tools.get(call.function.name)
                if tool is None:
                    result = f"Error: unknown tool '{call.function.name}'."
                else:
                    try:
                        result = tool.call(call.function.arguments)
                    except Exception as e:
                        result = f"Error running {tool.name}: {type(e).__name__}: {e}"
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )

        return {
            "final_answer": None,
            "messages": messages,
            "stop_reason": "max_steps",
            "steps": self.max_steps,
        }


def _assistant_tool_call_message(msg: Any, tool_calls: list) -> dict:
    """Serialize the model's tool-call turn into a chat message dict."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {
                    "name": c.function.name,
                    "arguments": c.function.arguments,
                },
            }
            for c in tool_calls
        ],
    }
