from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """
    Skill represents a reusable prompt "skill" that can be loaded from markdown or defined inline.
    It includes metadata, configuration, and sections of content that can be used to generate prompts.

    Attributes:
        id (str): Unique identifier for the skill.
        title (str): Title of the skill.
        description (str): Description of the skill. Defaults to an empty string.
        tags (list[str]): Tags associated with the skill. Defaults to an empty list.
        domain (str | None): Domain or category of the skill. Defaults to None.
        modes (list[str]): Modes in which the skill can be used (e.g., 'chat', 'planning', 'coding'). Defaults to an empty list.
        version (str | None): Version of the skill. Defaults to None.
        config (dict[str, Any]): Additional configuration for the skill. Defaults to an empty dictionary.
        sections (dict[str, str]): Parsed sections of the skill, keyed by dot-paths.
        raw_markdown (str | None): Raw markdown content of the skill. Defaults to None.
        path (Path | None): File path of the skill, if loaded from a file. Defaults to None.

    Methods:
        section(key: str, default: str = "") -> str:
            Retrieve a specific section by its dot-path key. Returns the default value if the section is missing.
        has_section(key: str) -> bool:
            Check if a specific section exists in the skill.
        compile_prompt(*section_keys: str, separator: str, fallback_keys: Iterable[str] | None = None) -> str:
            Compile a prompt by concatenating specified sections. If no sections are specified, compiles the entire skill.
        from_dict(meta: Mapping[str, Any], sections: Mapping[str, str], *, raw_markdown: str | None = None, path: Path | None = None) -> Skill:
            Class method to create a Skill instance from metadata and sections. Useful for programmatically defining skills.
    """

    id: str
    title: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    domain: str | None = None
    modes: list[str] = field(default_factory=list)  # e.g. ['chat', 'planning', 'coding']
    version: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # parsed contents
    sections: dict[str, str] = field(default_factory=dict)
    raw_markdown: str | None = None
    path: Path | None = None

    # helpers
    def section(self, key: str, default: str = "") -> str:
        """
        Retrieve a specific section value by its dot-path key, or return a default value if the key is missing.
        This method allows accessing nested sections of a configuration or data structure
        using a dot-separated key path. If the specified key is not found, the provided
        default value is returned.

        Examples:
            Accessing a specific section:
            ```python
            value = obj.section("chat.system")
            ```

            Providing a default value if the key is missing:
            ```python
            value = obj.section("nonexistent.key", default="Default Value")
            ```

        Args:
            key: A dot-separated string representing the path to the desired section.
            default: The value to return if the key is not found (default: an empty string).

        Returns:
            The value of the specified section if found, otherwise the default value.

        Notes:
            This method assumes that the `sections` attribute is a dictionary-like object
            that supports the `get` method for key-value retrieval.
        """
        return self.sections.get(key, default)

    def has_section(self, key: str) -> bool:
        """
        Check if a specific section exists in the skill.

        This method determines whether a given dot-path key corresponds to an
        existing section in the `sections` attribute.

        Examples:
            Checking for the existence of a section:
            ```python
            exists = skill.has_section("chat.system")
            ```

            Using the method to conditionally access a section:
            ```python
            if skill.has_section("chat.example"):
                example = skill.section("chat.example")
            ```

        Args:
            key: A dot-separated string representing the path to the desired section.

        Returns:
            bool: True if the section exists, False otherwise.
        """
        return key in self.sections

    def compile_prompt(
        self,
        *section_keys: str,
        separator: str = "\n\n",
        fallback_keys: Iterable[str] | None = None,
    ) -> str:
        """
        Compile a prompt by concatenating specified sections.

        Examples
        --------
        1) Only specific sections:

            prompt = skill.compile_prompt(
                "chat.system",
                "chat.example",
            )

        2) Full skill (lazy mode):

            # No section keys -> entire skill:
            prompt = skill.compile_prompt()

        Behavior
        --------
        - If `section_keys` are provided:
            - Concatenate those sections in order, skipping any that are missing.
        - If `section_keys` is empty and `fallback_keys` is provided:
            - Use `fallback_keys` as the section list.
        - If both `section_keys` and `fallback_keys` are empty / None:
            - Return the *entire* skill by concatenating:
              1) The optional "body" preface (if present), then
              2) All other sections in lexicographic order of their keys.

        Returns
        -------
        str
            The compiled prompt string (may be empty if no sections are found).
        """
        keys: list[str] = list(section_keys)

        if not keys:
            if fallback_keys:
                # Use caller-provided fallback ordering.
                keys = list(fallback_keys)
            else:
                # "Full skill" mode: include everything.
                all_keys = list(self.sections.keys())
                ordered: list[str] = []

                # Put "body" first if present.
                if "body" in self.sections:
                    ordered.append("body")
                    all_keys.remove("body")

                # Then all other sections in a stable order.
                ordered.extend(sorted(all_keys))
                keys = ordered

        chunks: list[str] = []
        for key in keys:
            text = self.sections.get(key)
            if text:
                chunks.append(text)

        return separator.join(chunks).strip()

    @classmethod
    def from_dict(
        cls,
        meta: Mapping[str, Any],
        sections: Mapping[str, str],
        *,
        raw_markdown: str | None = None,
        path: Path | None = None,
    ) -> Skill:
        """
        Create a Skill from Python metadata + sections.
        Useful for inline / programmatic skills.
        """
        skill_id = str(meta.get("id") or meta.get("name") or "").strip()
        if not skill_id:
            raise ValueError("Skill metadata must include a non-empty 'id' field.")

        title = str(meta.get("title") or skill_id).strip()
        description = str(meta.get("description") or "").strip()

        tags = list(meta.get("tags") or [])
        domain = meta.get("domain")
        modes = list(meta.get("modes") or [])
        version = meta.get("version")

        # Any extra fields in meta go into config
        know_keys = {"id", "name", "title", "description", "tags", "domain", "modes", "version"}
        config = {k: v for k, v in meta.items() if k not in know_keys}

        return cls(
            id=skill_id,
            title=title,
            description=description,
            tags=tags,
            domain=domain,
            modes=modes,
            version=version,
            config=config,
            sections=dict(sections),
            raw_markdown=raw_markdown,
            path=path,
        )
