from __future__ import annotations

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

        This method allows you to add a `Skill` object to the registry. If a
        skill with the same ID already exists and `overwrite` is set to `False`,
        a `ValueError` will be raised.

        Examples:
            Registering a skill object:
            ```python
            skill = Skill(
                id="example.skill",
                title="Example Skill",
                description="An example skill for demonstration purposes.",
                tags=["example", "demo"],
                domain="general",
                modes=["chat"],
            )
            registry.register(skill)
            ```

            Overwriting an existing skill:
            ```python
            registry.register(skill, overwrite=True)
            ```

        Args:
            skill: The `Skill` object to register. (Required)
            overwrite: Whether to overwrite an existing skill with the same ID. (Optional)
            Defaults to `False`.
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

        This method allows you to define and register a Skill inline, without
        needing to create a separate markdown file. It is useful for quick
        prototyping or defining simple skills directly in code.

        Examples:
            Registering a basic coding helper skill:
            ```python
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
            ```

        Args:
            id: The unique identifier for the skill. (Required)
            title: A short, descriptive title for the skill. (Required)
            description: A longer description of the skill's purpose. (Optional)
            tags: A list of tags for categorization. (Optional)
            domain: The domain or category the skill belongs to. (Optional)
            modes: A list of modes the skill supports (e.g., "chat", "coding"). (Optional)
            version: The version of the skill. (Optional)
            config: A dictionary of additional configuration options. (Optional)
            sections: A dictionary mapping section keys to their content. (Optional)
            overwrite: Whether to overwrite an existing skill with the same ID. (Optional)

        Returns:
            Skill: The registered `Skill` object.
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

        This method reads the content of a markdown file, parses it into a
        `Skill` object, and registers it in the skill registry.

        Examples:
            Loading a skill from a file:
            ```python
            skill = registry.load_file("path/to/skill.md")
            ```

        Args:
            path: The file path to the markdown skill file. (Required)
            overwrite: Whether to overwrite an existing skill with the same ID. (Optional)

        Returns:
            Skill: The registered `Skill` object.
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

        This method scans the specified directory for markdown files matching
        the given pattern, parses them into `Skill` objects, and registers them
        in the skill registry.

        Examples:
            Loading all skills from a directory recursively:
            ```python
            skills = registry.load_path("path/to/skills", pattern="*.md", recursive=True)
            ```

            Loading skills from a directory without recursion:
            ```python
            skills = registry.load_path("path/to/skills", recursive=False)
            ```

        Args:
            root: The root directory to scan for markdown files. (Required)
            pattern: The glob pattern to match files (e.g., `"*.md"`). (Optional)
            Defaults to `"*.md"`.
            recursive: Whether to scan directories recursively. (Optional)
            Defaults to `True`.
            overwrite: Whether to overwrite existing skills with the same ID. (Optional)
            Defaults to `False`.

        Returns:
            list[Skill]: A list of all successfully loaded and registered `Skill` objects.
        """
        root_path = Path(root)
        if recursive:  # noqa: SIM108
            files = list(root_path.rglob(pattern))
        else:
            files = list(root_path.glob(pattern))

        loaded: list[Skill] = []
        for f in sorted(files):
            print(f"🍎 Loading skill from {f}")
            loaded.append(self.load_file(f, overwrite=overwrite))
        return loaded

    def get(self, skill_id: str) -> Skill | None:
        """
        Get a registered Skill by id.

        This method retrieves a `Skill` object from the registry using its unique
        identifier. If the skill is not found, it returns `None`.

        Examples:
            Retrieving a skill by its ID:
            ```python
            skill = registry.get("coding.generic")
            if skill:
            print(skill.title)
            ```

        Args:
            skill_id: The unique identifier of the skill to retrieve. (Required)

        Returns:
            Skill | None: The `Skill` object if found, otherwise `None`.
        """
        return self._skills.get(skill_id)

    def require(self, skill_id: str) -> Skill:
        """
        Retrieve a registered Skill by its unique identifier.

        This method ensures that the requested Skill exists in the registry.
        If the Skill is not found, it raises a KeyError.

        Examples:
            Retrieving a skill by its ID:
            ```python
            skill = registry.require("coding.generic")
            print(skill.title)
            ```

        Args:
            skill_id: The unique identifier of the skill to retrieve. (Required)

        Returns:
            Skill: The `Skill` object corresponding to the given ID.

        Raises:
            KeyError: If the skill with the specified ID is not found.
        """
        skill = self.get(skill_id)
        if skill is None:
            raise KeyError(f"Skill with id={skill_id!r} not found")
        return skill

    def all(self) -> list[Skill]:
        """
        Return all registered Skills.

        This method retrieves all `Skill` objects currently registered in the
        skill registry and returns them as a list.

        Examples:
            Retrieving all registered skills:
            ```python
            skills = registry.all()
            for skill in skills:
            print(skill.id, skill.title)
            ```

        Returns:
            list[Skill]: A list of all registered `Skill` objects.
        """
        return list(self._skills.values())

    def ids(self) -> list[str]:
        """
        Return all registered Skill ids.

        This method retrieves the unique identifiers of all `Skill` objects
        currently registered in the skill registry and returns them as a sorted list.

        Examples:
            Retrieving all skill IDs:
            ```python
            skill_ids = registry.ids()
            print(skill_ids)
            ```

        Returns:
            list[str]: A sorted list of all registered skill IDs.
        """
        return sorted(self._skills.keys())

    # -------------- section helpers ----------------
    def section(self, skill_id: str, section_key: str, default: str = "") -> str:
        """
        Return a section for a given skill, or default.

        This method retrieves the content of a specific section within a skill
        by its unique identifier and section key. If the skill or section is not
        found, it returns the provided default value.

        Examples:
            Retrieving a section from a skill:
            ```python
            section_content = registry.section("coding.generic", "chat.system")
            print(section_content)
            ```

            Providing a default value if the section is missing:
            ```python
            section_content = registry.section("nonexistent.skill", "missing.section", default="Default content")
            ```

        Args:
            skill_id: The unique identifier of the skill. (Required)
            section_key: The key of the section to retrieve. (Required)
            default: The value to return if the skill or section is not found. (Optional)

        Returns:
            str: The content of the section if found, otherwise the default value.
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

        This method compiles a prompt by combining multiple sections of a skill
        identified by its unique ID. It allows you to specify the sections to
        include, the separator to use between sections, and fallback keys for
        missing sections.

        Examples:
            Compiling a prompt with specific sections:
            ```python
            prompt = registry.compile_prompt(
                "coding.generic",
                "chat.system",
                "chat.user",
            )
            ```

            Using fallback keys for missing sections:
            ```python
            prompt = registry.compile_prompt(
                "coding.generic",
                "chat.system",
                "chat.user",
                fallback_keys=["default.system", "default.user"]
            )
            ```

        Args:
            skill_id: The unique identifier of the skill. (Required)
            *section_keys: The keys of the sections to include in the prompt. (Required)
            separator: The string to use as a separator between sections. (Optional)
            Defaults to `double newline`.
            fallback_keys: A list of fallback section keys to use if a section
            is missing. (Optional)

        Returns:
            str: The compiled prompt as a single string.
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

        This method allows you to filter registered skills based on specific
        criteria such as tags, domain, mode, or a custom predicate function.

        Examples:
            Finding skills with a specific tag and mode:
            ```python
            skills = registry.find(tag="surrogate", mode="planning")
            ```

            Using a custom predicate to filter skills:
            ```python
            skills = registry.find(predicate=lambda s: "example" in s.title)
            ```

        Args:
            tag: A string representing the tag to filter by. (Optional)
            domain: The domain or category to filter by. (Optional)
            mode: The mode (e.g., "chat", "coding") to filter by. (Optional)
            predicate: A callable that takes a `Skill` object and returns a
            boolean indicating whether the skill matches the criteria. (Optional)

        Returns:
            list[Skill]: A list of `Skill` objects that match the specified criteria.

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

        This method provides a summary of all skills currently registered in
        the registry, including their metadata such as ID, title, description,
        tags, domain, modes, version, and sections.

        Examples:
            Retrieving skill descriptions for debugging or UI purposes:
            ```python
            descriptions = registry.describe()
            for skill in descriptions:
            print(skill["id"], skill["title"])
            ```

        Returns:
            list[dict[str, Any]]: A list of dictionaries, each containing the
            metadata of a registered skill.
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
