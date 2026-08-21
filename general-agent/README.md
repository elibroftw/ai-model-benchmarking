# general-agent

A model-agnostic agent framework built around one observation: **AI agents
waste time on the wrong details.** When asked to write text into an image,
they'll spend turns trying to identify the exact font instead of just picking
a readable one. When asked to rename a variable, they'll refactor surrounding
code that nobody asked them to touch.

The remedy here is **skills**: small, focused guidance documents the agent
loads on demand. A skill says "for this class of sub-task, here's what
actually matters — and here's what you should NOT spend cycles on." Skills
live in the [`skills/`](skills/) directory as plain Markdown with a small
frontmatter header.

## Design

- **Model layer**: [LiteLLM](https://github.com/BerriAI/litellm), so any
  provider works (OpenRouter, Anthropic, OpenAI, local llama.cpp, etc.).
- **Skills**: markdown files with `name` + `description` frontmatter. The
  agent sees the index (name + one-line trigger) in its system prompt; it
  loads the full text on demand via a `read_skill` tool.
- **Tools**: plain Python callables registered with a `ToolRegistry`. Every
  agent gets `read_skill` for free.
- **No sandboxing baked in.** Callers decide how much of the host to expose;
  the framework is a loop + skill loader, not a security boundary.

## Skill format

```markdown
---
name: scribbling-on-images
description: When writing text onto an image, prioritize readability. Do NOT
  spend effort matching the original font unless font-adherence is explicit
  in the task.
---

# Full skill body in markdown here…
```

The `description` is the trigger. Make it a plain-language *when-to-use* line,
not a summary — the agent decides whether to load the skill by matching its
current sub-task against this line, so a good description names the situation
where the skill applies.

## Usage sketch

```python
from general_agent import Agent, SkillRegistry

agent = Agent(
    model="openrouter/anthropic/claude-3.5-sonnet",
    skills=SkillRegistry.from_dir("skills"),
)
result = agent.run("Take input.png and write 'HELLO' centered on it, save as out.png.")
```

## Status

Early scaffolding. The skill loader and tool registry work; the main agent
loop is a minimal LiteLLM tool-calling loop that will grow as we run this
against real tasks. Existing skills are in [`skills/`](skills/); add one
whenever you catch the agent burning cycles on an incidental detail.

Planned work is in [`TODO.md`](TODO.md) — notably a `writing-skills` skill,
so the agent can author its own when it hits a task nothing covers.

## Related repos

- [`sudoku-vision-benchmark`](../sudoku-vision-benchmark/) — the benchmarker
  that exercises harnesses like this one.
- [`sudoku-agent-harness`](../sudoku-agent-harness/) — smolagents-based
  harness. Once `general-agent` is mature enough, we may wrap it as a
  drop-in replacement matching the same subprocess CLI contract.
