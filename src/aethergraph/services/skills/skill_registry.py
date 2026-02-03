from pathlib import Path
from typing import Any

from aethergraph.services.skills.utils import parse_skill_markdown

from .skills import Skill


class SkillRegistry:
    """
    Registry for reusable prompt skills.

    Supports:
      - Registering inline Skill objects.
      - Loading skills from one or more skill directories (markdown files).
      - Retrieving entire skills or specific sections via dot-path keys.
      - Simple filtering by tags/domain/modes.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill, *, overwrite: bool = False) -> None:
        """
        Register a Skill object.

        If overwrite=False and the id already exists, raise ValueError.
        """
        if not overwrite and skill.id in self._skills:
            raise ValueError(f"Skill with id={skill.id!r} already registered")
        self._skills[skill.id] = skill

    def register_inline(
        self,
        *,
        id: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
        domain: str | None = None,
        modes: list[str] | None = None,
        version: str | None = None,
        config: dict[str, Any] | None = None,
        sections: dict[str, str] | None = None,
        overwrite: bool = False,
    ) -> Skill:
        """
        Convenience for defining a Skill entirely in Python.

        Example:

            registry.register_inline(
                id="coding.generic",
                title="Generic coding helper",
                description="Helps with Python code generation and review.",
                tags=["coding"],
                modes=["chat", "coding"],
                sections={
                    "chat.system": "You are a helpful coding assistant...",
                    "coding.system": "You write code as JSON ...",
                },
            )
        """
        skill = Skill(
            id=id,
            title=title,
            description=description,
            tags=list(tags or []),
            domain=domain,
            modes=list(modes or []),
            version=version,
            config=dict(config or {}),
            sections=dict(sections or {}),
            raw_markdown=None,
            path=None,
        )
        self.register(skill, overwrite=overwrite)
        return skill

    def load_file(self, path: str | Path, *, overwrite: bool = False) -> Skill:
        """
        Load a single .md skill file and register it.

        Returns the Skill instance.
        """
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        skill = parse_skill_markdown(text, path=p)
        self.register(skill, overwrite=overwrite)
        return skill

    def load_path(
        self,
        root: str | Path,
        *,
        pattern: str = "*.md",
        recursive: bool = True,
        overwrite: bool = False,
    ) -> list[Skill]:
        """
        Load all skill markdown files under a directory.

        root/
          surrogate-workflow.md
          coding-generic.md
          chat-default.md
        """
        root_path = Path(root)
        if recursive:  # noqa: SIM108
            files = list(root_path.rglob(pattern))
        else:
            files = list(root_path.glob(pattern))

        loaded: list[Skill] = []
        for f in sorted(files):
            loaded.append(self.load_file(f, overwrite=overwrite))
        return loaded

    def get(self, skill_id: str) -> Skill | None:
        """
        Get a registered Skill by id.

        Returns None if not found.
        """
        return self._skills.get(skill_id)

    def require(self, skill_id: str) -> Skill:
        """
        Get a registered Skill by id.

        Raises KeyError if not found.
        """
        skill = self.get(skill_id)
        if skill is None:
            raise KeyError(f"Skill with id={skill_id!r} not found")
        return skill

    def all(self) -> list[Skill]:
        """
        Return all registered Skills.
        """
        return list(self._skills.values())

    def ids(self) -> list[str]:
        """
        Return all registered Skill ids.
        """
        return sorted(self._skills.keys())

    # -------------- section helpers ----------------
    def section(self, skill_id: str, section_key: str, default: str = "") -> str:
        """
        Return a section for a given skill, or default.
        """
        skill = self.get(skill_id)
        if not skill:
            return default
        return skill.section(section_key, default=default)

    def compile_prompt(
        self,
        skill_id: str,
        *section_keys: str,
        separator: str = "\n\n",
        fallback_keys: list[str] | None = None,
    ):
        """
        Shortcut for Skill.compile_prompt(...) by id.
        """
        skill = self.require(skill_id)
        return skill.compile_prompt(
            *section_keys,
            separator=separator,
            fallback_keys=fallback_keys,
        )

    def find(
        self,
        *,
        tag: str | None = None,
        domain: str | None = None,
        mode: str | None = None,
        predicate: callable | None = None,
    ) -> list[Skill]:
        """
        Filter skills by tag, domain, mode, and/or a custom predicate.

        Example:
            registry.find(tag="surrogate", mode="planning")
        """
        out: list[Skill] = []
        for s in self._skills.values():
            if tag and tag not in s.tags:
                continue
            if domain and s.domain != domain:
                continue
            if mode and mode not in s.modes:
                continue
            if predicate and not predicate(s):
                continue
            out.append(s)
        return out

    def describe(self) -> list[dict[str, Any]]:
        """
        Return a compact description of all registered skills.
        Useful for debugging / UIs.
        """
        info: list[dict[str, Any]] = []
        for s in self._skills.values():
            info.append(
                {
                    "id": s.id,
                    "title": s.title,
                    "description": s.description,
                    "tags": s.tags,
                    "domain": s.domain,
                    "modes": s.modes,
                    "version": s.version,
                    "path": str(s.path) if s.path else None,
                    "sections": sorted(s.sections.keys()),
                }
            )
        return info
