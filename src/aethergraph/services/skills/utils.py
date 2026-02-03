from pathlib import Path
import re
from typing import Any

from .skills import Skill

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None  # TODO: enforce PyYAML as a dependency?

_FRONT_MATTER_DELIM = re.compile(r"^---\s*$")


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
        return {}, text

    # Find closing '---'
    end_idx = None
    for i in range(1, len(lines)):
        if _FRONT_MATTER_DELIM.match(lines[i].strip()):
            end_idx = i
            break

    if end_idx is None:
        # Malformed front matter; treat as no front matter
        return {}, text

    fm_lines = lines[1:end_idx]
    body_lines = lines[end_idx + 1 :]

    fm_str = "\n".join(fm_lines)
    body = "\n".join(body_lines)

    if yaml is None:
        # If PyYAML is not installed, return empty meta.
        return {}, body

    meta = yaml.safe_load(fm_str) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


_SECTION_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


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

    Example body:

        Intro text before any heading.

        ## chat.system
        System prompt here...

        ## chat.style
        Style instructions here...

        ## planning.header
        Planning header...

    Produces sections:
        {
          "body": "Intro text before any heading.",
          "chat.system": "System prompt here...",
          "chat.style": "Style instructions here...",
          "planning.header": "Planning header...",
        }
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

    The file should look like:

        ---
        id: surrogate-workflow
        title: Surrogate Workflow Orchestration
        description: Skill for planning surrogate training workflows.
        tags: [surrogate, planning]
        domain: ml/surrogate
        modes: [planning, chat]
        version: "0.1.0"
        ---
        ## chat.system
        You are a helpful assistant...

        ## planning.header
        You are a planning assistant...

        ## planning.binding_hints
        - Use ${user.dataset_path} instead of inventing paths.
        ...

    """
    meta, body = _split_front_matter(text)
    sections = _split_sections_from_body(body)

    return Skill.from_dict(
        meta=meta,
        sections=sections,
        raw_markdown=text,
        path=path,
    )
