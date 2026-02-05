from pathlib import Path
import re
from typing import Any

from .skills import Skill

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # TODO: enforce PyYAML as a dependency?

_FRONT_MATTER_DELIM = re.compile(r"^---\s*$")
# Only treat H2 (##) as section delimiters, per spec.
_SECTION_HEADING_RE = re.compile(r"^(##)\s+(.*)$")


def _split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """
    Split YAML front matter from the rest of the markdown body.

    Expects:
        ---
        yaml: here
        ---
        # Markdown starts here

    Returns: (meta_dict, body_markdown)
    """
    lines = text.splitlines()
    if not lines or not _FRONT_MATTER_DELIM.match(lines[0].strip()):
        # No front matter block
        return {}, text

    # Find closing '---'
    end_idx = None
    for i in range(1, len(lines)):
        if _FRONT_MATTER_DELIM.match(lines[i].strip()):
            end_idx = i
            break

    if end_idx is None:
        # Malformed front matter; treat entire file as body
        return {}, text

    fm_lines = lines[1:end_idx]
    body_lines = lines[end_idx + 1 :]

    fm_str = "\n".join(fm_lines)
    body = "\n".join(body_lines)

    if yaml is None:
        # If PyYAML is not installed, return empty meta.
        return {}, body

    try:
        meta = yaml.safe_load(fm_str) or {}
    except Exception as exc:
        # Surface a clear error – this is almost always a YAML indentation / syntax issue
        raise ValueError(
            f"Failed to parse YAML front matter: {exc!r}\n" f"Front matter was:\n{fm_str}"
        ) from exc

    if not isinstance(meta, dict):
        raise ValueError(
            f"YAML front matter must be a mapping (dict), got {type(meta)} instead.\n"
            f"Front matter was:\n{fm_str}"
        )

    return meta, body


def _normalize_section_key(heading: str) -> str:
    """
    Normalize a heading text into a section key.

    Rules:
      - If the heading already contains a dot (e.g. "chat.system"), keep as-is.
      - Else, lowercase and replace spaces with underscores, e.g. "Chat System" -> "chat_system".
    """
    raw = heading.strip()
    if "." in raw:
        return raw.strip()
    return raw.lower().replace(" ", "_")


def _split_sections_from_body(body: str) -> dict[str, str]:
    """
    Split markdown body into sections keyed by normalized heading.

    - Intro text before any heading -> section "body"
    - Only H2 headings (`## something`) start new sections.
    - H3+ (`### ...`) are treated as content.
    """
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    buffer: list[str] = []
    preface: list[str] = []

    lines = body.splitlines()

    for line in lines:
        m = _SECTION_HEADING_RE.match(line)
        if m:
            # Flush previous section or preface
            if current_key is None:
                if buffer:
                    preface.extend(buffer)
            else:
                sections[current_key] = sections.get(current_key, []) + buffer

            buffer = []
            heading = m.group(2).strip()
            current_key = _normalize_section_key(heading)
        else:
            buffer.append(line)

    # Flush last buffer
    if current_key is None:
        if buffer:
            preface.extend(buffer)
    else:
        sections[current_key] = sections.get(current_key, []) + buffer

    out: dict[str, str] = {}
    if preface:
        out["body"] = "\n".join(preface).strip()

    for k, lines_ in sections.items():
        text = "\n".join(lines_).strip()
        if text:
            out[k] = text

    return out


def parse_skill_markdown(text: str, path: Path | None = None) -> Skill:
    """
    Parse a single markdown file into a Skill.

    The file must have YAML front matter with at least:
      - id: string
      - title: string

    Sections are defined by `## section.key` headings.
    """
    meta, body = _split_front_matter(text)
    sections = _split_sections_from_body(body)

    location = str(path) if path is not None else "<string>"

    # ---- Basic validation of YAML meta ----
    if not meta:
        raise ValueError(
            f"Skill file {location} has no YAML front matter. "
            "Expected at least:\n"
            "---\n"
            "id: some.id\n"
            "title: Some title\n"
            "---"
        )

    skill_id = meta.get("id")
    title = meta.get("title")

    if not isinstance(skill_id, str) or not skill_id.strip():
        raise ValueError(
            f"Skill file {location} is missing a valid 'id' in front matter. "
            f"Got id={skill_id!r} in:\n{meta}"
        )

    if not isinstance(title, str) or not title.strip():
        raise ValueError(
            f"Skill file {location} is missing a valid 'title' in front matter. "
            f"Got title={title!r} in:\n{meta}"
        )

    try:
        skill = Skill.from_dict(
            meta=meta,
            sections=sections,
            raw_markdown=text,
            path=path,
        )
    except Exception as e:
        raise ValueError(f"Failed to construct Skill from {location}: {e}") from e

    # Extra guard – if Skill.from_dict ever returns None, fail loudly.
    if skill is None:
        raise ValueError(
            f"Skill.from_dict returned None for skill file {location}. "
            f"Front matter: {meta}, sections: {list(sections.keys())}"
        )

    return skill
