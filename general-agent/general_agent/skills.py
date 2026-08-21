"""Skill loading.

Skills are markdown files with a small YAML-ish frontmatter block:

    ---
    name: kebab-case-name
    description: One line explaining when this skill applies. This is what the
      agent matches against its current sub-task to decide whether to load the
      full skill body.
    ---

    # Full skill content in markdown

Only `name` and `description` are required; additional fields are preserved as
metadata for future use.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<meta>.*?)\n---\s*\n(?P<body>.*)\Z", re.DOTALL
)


@dataclass
class Skill:
    name: str
    description: str
    body: str
    source_path: Path | None = None
    metadata: dict = field(default_factory=dict)

    def full_text(self) -> str:
        """Return the skill body plus its metadata header, for injection into
        the agent's context when `read_skill` is called."""
        header = f"# Skill: {self.name}\n\n{self.description}\n\n---\n\n"
        return header + self.body.strip() + "\n"


class SkillRegistry:
    def __init__(self, skills: list[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for s in skills or []:
            self.register(s)

    @classmethod
    def from_dir(cls, path: str | Path) -> "SkillRegistry":
        """Load every *.md file under `path` as a Skill."""
        registry = cls()
        for md_path in sorted(Path(path).glob("*.md")):
            registry.register(load_skill(md_path))
        return registry

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"Duplicate skill name: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def read(self, name: str) -> str:
        """Tool-facing accessor. Returns the full skill text, or a helpful
        error string if the name is unknown."""
        skill = self.get(name)
        if skill is None:
            known = ", ".join(sorted(self._skills)) or "(none)"
            return (
                f"No skill named '{name}'. Known skills: {known}."
            )
        return skill.full_text()

    def names(self) -> list[str]:
        return list(self._skills)

    def index_text(self) -> str:
        """Compact, model-facing listing: `- name — description`."""
        if not self._skills:
            return "(no skills loaded)"
        return "\n".join(
            f"- **{s.name}** — {s.description}"
            for s in self._skills.values()
        )


def load_skill(path: str | Path) -> Skill:
    """Parse a single markdown skill file."""
    path = Path(path)
    text = path.read_text()
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(
            f"{path}: missing --- frontmatter with `name` and `description`."
        )
    meta = _parse_frontmatter(match.group("meta"))
    if "name" not in meta or "description" not in meta:
        raise ValueError(
            f"{path}: frontmatter must include both `name` and `description`."
        )
    return Skill(
        name=meta.pop("name"),
        description=meta.pop("description"),
        body=match.group("body"),
        source_path=path,
        metadata=meta,
    )


def _parse_frontmatter(text: str) -> dict:
    """Tiny YAML-ish parser: `key: value` per line, with continuation lines
    starting with whitespace being appended to the previous value.
    Deliberately minimal — no lists, no nested dicts. Add PyYAML if the format
    outgrows this."""
    out: dict = {}
    last_key: str | None = None
    for raw in text.splitlines():
        if not raw.strip():
            continue
        if raw[:1] in (" ", "\t") and last_key is not None:
            out[last_key] = f"{out[last_key]} {raw.strip()}"
            continue
        if ":" not in raw:
            raise ValueError(f"Malformed frontmatter line: {raw!r}")
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        out[key] = value
        last_key = key
    return out
