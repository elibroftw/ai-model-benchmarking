"""Tool registry.

A Tool is a name, a description, a JSON-schema for its parameters, and a
handler callable. Handlers receive keyword args matching the schema and
return either a string (shown to the model) or any JSON-serializable value
(stringified before insertion).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    params_schema: dict
    handler: Callable[..., Any]

    def to_openai_schema(self) -> dict:
        """Emit the OpenAI-style function-calling schema LiteLLM expects."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.params_schema,
            },
        }

    def call(self, arguments: dict | str) -> str:
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments else {}
        result = self.handler(**arguments)
        if isinstance(result, str):
            return result
        try:
            return json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(result)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def to_openai_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)
