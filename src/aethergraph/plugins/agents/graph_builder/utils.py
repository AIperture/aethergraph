from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from aethergraph.core.runtime.node_context import NodeContext


@dataclass
class BuilderFileRef:
    name: str | None
    mimetype: str | None
    source: Literal["upload", "artifact"]
    artifact_id: str | None = None  # required if source == "artifact"
    uri: str | None = (
        None  # optional pre-signed URI for direct access (can be used if source == "artifact" or "upload")
    )
    kind: Literal["code", "notebook", "text", "other"] = (
        "other"  # optional hint for the LLM about how to interpret the file
    )


def _classify_builder_file(
    name: str | None, mimetype: str | None
) -> Literal["code", "notebook", "text", "other"]:
    n = (name or "").lower()
    m = (mimetype or "").lower()

    # Code-like
    if n.endswith(".py") or "python" in m:
        return "code"

    # Notebooks
    if n.endswith(".ipynb") or "notebook" in m or "jupyter" in m or "ipynb" in m:
        return "notebook"

    # Text / docs
    if n.endswith(".md") or n.endswith(".markdown") or m in ("text/markdown", "text/x-markdown"):
        return "text"
    if n.endswith(".txt") or m == "text/plain":
        return "text"
    if m in ("application/json",):
        # JSON could be config/spec; treat as text-ish for the builder
        return "text"

    return "other"


async def _process_builder_attachments(
    attachments: list[dict[str, Any]] | None,
    context: NodeContext,
) -> tuple[list[BuilderFileRef], list[BuilderFileRef], list[BuilderFileRef], str]:
    """
    Process normalized attachments into (code, text, other, notes).

    Current supported attachment kind is:
      - artifact (with artifact_id)
    """
    artifacts = context.artifacts()
    notes = ""

    code_files: list[BuilderFileRef] = []
    text_files: list[BuilderFileRef] = []
    other_files: list[BuilderFileRef] = []

    for ref in attachments or []:
        if not isinstance(ref, dict):
            notes += f"- Ignoring malformed attachment entry: {json.dumps(ref)}\n"
            continue

        kind_name = str(ref.get("kind") or "").strip()
        if kind_name != "artifact":
            notes += f"- Ignoring unsupported attachment kind '{kind_name or 'unknown'}'.\n"
            continue

        art_id = ref.get("artifact_id")
        if not art_id:
            notes += f"- Ignoring artifact attachment without artifact_id: {json.dumps(ref)}\n"
            continue
        try:
            art = await artifacts.get_by_id(art_id)
        except Exception as e:
            notes += (
                f"- Failed to retrieve artifact for attachment artifact_id '{art_id}': {str(e)}\n"
            )
            continue

        if not art:
            notes += f"- No artifact found for attachment artifact_id '{art_id}'.\n"
            continue

        source = str(ref.get("source") or "context_ref").strip() or "context_ref"
        if source not in {"upload", "context_ref", "system"}:
            source = "context_ref"

        mimetype = (
            ref.get("mimetype") or getattr(art, "mimetype", None) or getattr(art, "mime", None)
        )
        name = (
            ref.get("name")
            or getattr(art, "name", None)
            or getattr(art, "filename", None)
            or art_id
        )
        uri = ref.get("uri") or getattr(art, "uri", None)

        kind = _classify_builder_file(name, mimetype)
        bref = BuilderFileRef(
            name=name,
            mimetype=mimetype,
            source="upload" if source == "upload" else "artifact",
            artifact_id=art_id,
            uri=uri,
            kind=kind,
        )

        if kind in ("code", "notebook"):
            code_files.append(bref)
        elif kind == "text":
            text_files.append(bref)
        else:
            if mimetype:
                notes += f"- Ignoring artifact '{name}' (id: {art_id}) with unsupported mimetype '{mimetype}'.\n"
            else:
                notes += f"- Ignoring artifact '{name}' (id: {art_id}) with unknown mimetype.\n"

    return code_files, text_files, other_files, notes


def _infer_builder_mode(message: str, attachments: list[Any] | None = None) -> str:
    """
    Heuristic: decide whether we're creating from intent or wrapping an existing script.

    Returns:
        "create_from_intent" | "wrap_existing_script"
    """
    msg = (message or "").lower()

    has_code_fence = "```" in msg
    has_py_hint = ".py" in msg or "python" in msg

    has_py_file = False
    for f in attachments or []:
        # Adjust this depending on your file structure; this is intentionally loose.
        name = ""
        if isinstance(f, dict):
            name = f.get("filename") or f.get("name") or ""
        elif hasattr(f, "filename") or hasattr(f, "name"):
            name = getattr(f, "filename", "") or getattr(f, "name", "") or ""
        if name.endswith(".py"):
            has_py_file = True
            break

    if has_py_file or (has_code_fence and has_py_hint):
        return "wrap_existing_script"

    return "create_from_intent"


def _summarize_builder_files_for_llm(
    code_files: list[BuilderFileRef],
    text_files: list[BuilderFileRef],
    other_files: list[BuilderFileRef],
    notes: str,
) -> str:
    """Build a human-readable summary of the attached files for the LLM."""
    lines: list[str] = []
    if code_files:
        lines.append("Code / notebook candidates:")
        for f in code_files:
            lines.append(
                f"  - {f.name or '<unnamed>'} "
                f"(kind={f.kind}, source={f.source}, mimetype={f.mimetype or 'unknown'}, "
                f"artifact_id={f.artifact_id or 'none'})"
            )
        lines.append("")

    if text_files:
        lines.append("Text / doc files:")
        for f in text_files:
            lines.append(
                f"  - {f.name or '<unnamed>'} "
                f"(source={f.source}, mimetype={f.mimetype or 'unknown'}, "
                f"artifact_id={f.artifact_id or 'none'})"
            )
        lines.append("")

    if other_files:
        lines.append("Other files (not treated as code or text for now):")
        for f in other_files:
            lines.append(
                f"  - {f.name or '<unnamed>'} "
                f"(source={f.source}, mimetype={f.mimetype or 'unknown'}, "
                f"artifact_id={f.artifact_id or 'none'})"
            )
        lines.append("")

    if notes.strip():
        lines.append("Notes from file/context processing:")
        for line in notes.strip().splitlines():
            lines.append(f"  {line}")

    if not lines:
        return "No usable files or artifacts were found."

    return "\n".join(lines)
