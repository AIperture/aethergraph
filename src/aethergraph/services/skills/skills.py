from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Skill:
    """
    A reusable prompt "skill" loaded from markdown or defined inline.

    Typical usage:
      - Header (front matter) holds structured metadata and config.
      - Body (markdown) is split into sections keyed by 'dot paths',
        e.g. 'chat.system', 'planning.header', 'coding.system'.
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
        Return a specific section (by dot-path), or default if missing.

        Example keys:
          - "chat.system"
          - "planning.header"
          - "coding.system"
          - "body" (for pre-heading text)
        """
        return self.sections.get(key, default)

    def has_section(self, key: str) -> bool:
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
