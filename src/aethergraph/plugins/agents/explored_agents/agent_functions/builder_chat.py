from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Literal

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.agent_functions.basic_chat import (
    _compute_retrieval_plan,
    gather_chat_context,
)
from aethergraph.plugins.agents.types import BUILTIN_AGENT_SKILL_ID

AGENT_BUILDER_SKILL_ID = "aethergraph-graph-agent-app-creator"


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


async def _process_builder_files(
    files: list[Any] | None,
    context_refs: list[dict[str, Any]] | None,
    context: NodeContext,
) -> tuple[list[BuilderFileRef], list[BuilderFileRef], list[BuilderFileRef], str]:
    """
    Process uploaded files and context_refs into (code, text, other, notes).

    - "files" are direct uploads passed via chat_v1.
    - "context_refs" may reference artifacts in the artifact store by artifact_id.

    We:
      - classify files into code/notebook/text/other,
      - resolve artifact refs via context.artifacts().get_by_id(),
      - produce human-readable notes about anything we ignore or can't resolve.
    """
    artifacts = context.artifacts()
    notes = ""

    code_files: list[BuilderFileRef] = []
    text_files: list[BuilderFileRef] = []
    other_files: list[BuilderFileRef] = []

    # --- 1) uploaded files ---
    for f in files or []:
        # we support both dict-like and object-like file representations for flexibility
        if isinstance(f, dict):
            name = f.get("filename") or f.get("name")
            mimetype = f.get("mimetype") or f.get("type") or f.get("content_type")
            uri = f.get("uri")  # optional pre-signed URI for direct access
        else:
            name = getattr(f, "filename", None) or getattr(f, "name", None)
            mimetype = (
                getattr(f, "mimetype", None)
                or getattr(f, "type", None)
                or getattr(f, "content_type", None)
            )
            uri = getattr(f, "uri", None)
        kind = _classify_builder_file(name, mimetype)
        ref = BuilderFileRef(
            name=name,
            mimetype=mimetype,
            source="upload",
            uri=uri,
            kind=kind,
        )

        if kind in ("code", "notebook"):
            code_files.append(ref)
        elif kind == "text":
            text_files.append(ref)
        else:
            if mimetype:
                notes += (
                    f"- Ignoring uploaded file '{name}' with unsupported mimetype '{mimetype}'.\n"
                )
            else:
                notes += f"- Ignoring uploaded file '{name}' with unknown mimetype.\n"

    # --- 2) context_refs that point to artifacts ---
    for ref in context_refs or []:
        art_id = ref.get("artifact_id")
        if not art_id:
            # Could be other ref kinds; we ignore for now but note it.
            notes += f"- Ignoring context_ref without artifact_id: {json.dumps(ref)}\n"
            continue
        try:
            art = await artifacts.get_by_id(art_id)
        except Exception as e:
            notes += f"- Failed to retrieve artifact for context_ref with artifact_id '{art_id}': {str(e)}\n"
            continue

        if not art:
            notes += f"- No artifact found for context_ref with artifact_id '{art_id}'.\n"
            continue

        mimetype = getattr(art, "mimetype", None) or getattr(art, "mime", None)
        name = getattr(art, "name", None) or getattr(art, "filename", None) or art_id
        uri = getattr(art, "uri", None)  # optional pre-signed URI for direct access

        kind = _classify_builder_file(name, mimetype)
        bref = BuilderFileRef(
            name=name,
            mimetype=mimetype,
            source="artifact",
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


def _infer_builder_mode(message: str, files: list[Any] | None = None) -> str:
    """
    Heuristic: decide whether we're creating from intent or wrapping an existing script.

    Returns:
        "create_from_intent" | "wrap_existing_script"
    """
    msg = (message or "").lower()

    has_code_fence = "```" in msg
    has_py_hint = ".py" in msg or "python" in msg

    has_py_file = False
    for f in files or []:
        # Adjust this depending on your file structure; this is intentionally loose.
        name = ""
        if isinstance(f, dict):
            name = f.get("filename") or f.get("name") or ""
        elif hasattr(f, "filename"):
            name = getattr(f, "filename", "") or ""
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


async def builder_chat_handler(
    *,
    message: str,
    files: list[Any] | None,
    context_refs: list[dict[str, Any]] | None,
    session_id: str | None,
    user_meta: dict[str, Any] | None,
    context: NodeContext,
) -> str:
    """
    Chat handler for the Agent Builder agent.

    Responsibilities:
      - Interpret user intent about AG agents/apps/graphs.
      - Use attached scripts (files) and artifacts (context_refs) as candidates
        for wrapping.
      - Produce AG code (@graph_fn, @graphify, as_agent/as_app) as text.
    """
    logger = context.logger()
    llm = context.llm()
    chan = context.ui_session_channel()

    msg_text = (message or "").strip()

    # Consider files for mode inference (wrap vs create-from-intent)
    mode = _infer_builder_mode(msg_text, files)

    # 1) Compute retrieval plan (reuse existing config for now)
    plan = _compute_retrieval_plan(
        intent=None,  # you can pass a richer intent object later
        message=msg_text,
        context=context,
    )

    # 2) Gather retrieval context (memory, KB, etc.)
    ctx = await gather_chat_context(
        message=msg_text,
        session_id=session_id,
        context=context,
        plan=plan,
    )

    session_summary = ctx.get("session_summary", "")
    recent_chat = ctx.get("recent_chat", [])
    user_memory_snippets = ctx.get("user_memory_snippets", "")
    kb_snippets = ctx.get("kb_snippets", "")

    # 3) Build system prompt from the Agent Builder skill
    skills = context.skills()
    try:
        # Base instructions: how to behave as an AG chat agent, how to use retrieval, etc.
        base_prompt = skills.compile_prompt(
            BUILTIN_AGENT_SKILL_ID,
            "chat.system",
            "chat.retrieval",
            "chat.style",
            separator="\n\n",
            fallback_keys=["chat.system"],
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "builder_chat_handler: failed to compile base prompt for %s",
            BUILTIN_AGENT_SKILL_ID,
            extra={"error": str(e)},
        )
        base_prompt = (
            "You are the built-in Aether Agent.\n\n"
            "You help users understand and use AetherGraph, including graphs, agents, "
            "memory, artifacts, and KB. Use any provided context carefully and prefer "
            "KB snippets over your own assumptions when they talk about AetherGraph APIs."
        )

    try:
        # Specialization: the full Agent Builder skill doc (no chat.* sections needed)
        # Depending on your skills implementation, calling compile_prompt with only
        # the skill id (no section keys) should render the whole body.
        builder_specialization = skills.compile_prompt(
            AGENT_BUILDER_SKILL_ID,
            separator="\n\n",
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "builder_chat_handler: failed to compile builder specialization for %s",
            AGENT_BUILDER_SKILL_ID,
            extra={"error": str(e)},
        )
        builder_specialization = (
            "# AetherGraph Graph/Agent/App Creator\n\n"
            "You specialize in turning user intent and existing Python scripts into "
            "AetherGraph graphs, agents, and apps using @graph_fn, @graphify, @tool, "
            "and as_agent/as_app metadata. You must only use real NodeContext APIs and "
            "AetherGraph primitives."
        )

    system_prompt = (
        base_prompt + "\n\n"
        "----- AGENT BUILDER SPECIALIZATION -----\n\n"
        "You are now operating in **Agent Builder** mode. In addition to your normal "
        "Aether Agent behavior, you MUST obey the following constraints and patterns "
        "for graph/agent/app creation:\n\n" + builder_specialization
    )

    # return "dummy"  # placeholder until we implement the actual LLM call below

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    # 4) Attach retrieved context as system messages, but framed for code generation
    if session_summary:
        messages.append(
            {
                "role": "system",
                "content": "Session summary (for additional context, may be optional):\n"
                + session_summary,
            }
        )

    if user_memory_snippets:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant user memory snippets that may describe prior graph/agent design:\n\n"
                    f"{user_memory_snippets}\n\n"
                    "Use these only if they clearly help shape the requested agent/app."
                ),
            }
        )

    if kb_snippets:
        messages.append(
            {
                "role": "system",
                "content": (
                    "AetherGraph docs snippets from the KB (these are ground truth for APIs):\n\n"
                    f"{kb_snippets}\n\n"
                    "Follow these APIs precisely when generating code."
                ),
            }
        )

    # 5) Process files + context_refs for builder usage
    code_files, text_files, other_files, file_notes = await _process_builder_files(
        files=files,
        context_refs=context_refs,
        context=context,
    )
    files_summary = _summarize_builder_files_for_llm(
        code_files=code_files,
        text_files=text_files,
        other_files=other_files,
        notes=file_notes,
    )

    # 6) Replay recent chat briefly for continuity
    for item in recent_chat:
        role = (item.get("role") or "user").lower()
        text = item.get("text") or ""
        mapped_role = role if role in {"user", "assistant", "system"} else "assistant"
        if text:
            messages.append({"role": mapped_role, "content": text})

    # 7) Append current user turn with explicit builder instructions
    builder_user_content = (
        "You are being invoked by the Agent Builder agent inside AetherGraph.\n\n"
        f"Builder mode: {mode}\n\n"
        "User intent (verbatim):\n"
        f"{msg_text}\n\n"
        "Attached files and context_refs summary:\n"
        f"{files_summary}\n\n"
        "Interpret this as follows:\n"
        "- Code/notebook candidates are scripts the user likely wants to wrap as agents/apps.\n"
        "- Text/doc files may describe requirements or configs.\n"
        "- Other files can usually be ignored.\n\n"
        "Follow the output-format rules defined in your skill configuration. "
        "At minimum, you MUST:\n"
        "- Explain briefly what you are proposing.\n"
        "- Provide one or more complete Python code blocks (```python ... ```)\n"
        "  for the proposed graphs/agents/apps.\n"
        "- Use only real AetherGraph primitives and NodeContext APIs.\n"
        "- When wrapping existing scripts, do NOT rewrite them; create thin wrappers "
        "that import and call them."
    )

    messages.append(
        {
            "role": "user",
            "content": builder_user_content,
        }
    )

    # 8) Streaming LLM call to UI
    async with chan.stream() as s:

        async def on_delta(piece: str) -> None:
            await s.delta(piece)

        resp_text, _usage = await llm.chat_stream(
            messages=messages,
            on_delta=on_delta,
        )

        await s.end(
            full_text=resp_text,
            memory_log=False,  # we log via mem.record_chat_* in agent_builder_agent
        )

    return resp_text
