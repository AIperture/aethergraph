from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from aethergraph.contracts.services.channel import Button
from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.plugins.agents.graph_builder.registration_utils import (
    _ensure_graph_registered_from_code,
    _propose_app_config_via_llm,
    _resolve_registration_target,
)
from aethergraph.services.llm.generic_client import GenericLLMClient

from .types import (
    BUILDER_PLAN_WRAPPER_SCHEMA,
    GRAPH_BUILDER_SKILL_ID,
    GraphBuilderBranch,
    _detect_approval_intent,
    _hash_contract,
    _load_state,
    _save_state,
)
from .utils import BuilderFileRef, _classify_builder_file

NODE_API_SKILL_ID = "ag-graph-builder-context-node-api"
CHANNEL_API_SKILL_ID = "ag-graph-builder-channel-api"
ARTIFACT_API_SKILL_ID = "ag-graph-builder-artifact-api"
GRAPHIFY_STYLE_SKILL_ID = "ag-graph-builder-graphify-style"
CHECKPOINT_SKILL_ID = "ag-graph-builder-checkpoint-pattern"


async def _recent_chat_for_llm(*, context: NodeContext, limit: int = 20) -> list[dict[str, str]]:
    mem = context.memory()
    rows = await mem.recent_chat(limit=limit)  # type: ignore
    msgs: list[dict[str, str]] = []
    for r in rows:
        role = (r.get("role") or "user").lower()
        text = (r.get("text") or "").strip()
        if not text:
            continue
        if role not in {"user", "assistant", "system"}:
            role = "assistant"
        msgs.append({"role": role, "content": text})
    return msgs


def _compile_branch_prompt(*, context: NodeContext, branch: GraphBuilderBranch) -> str:
    skills = context.skills()
    if branch == GraphBuilderBranch.PLAN:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        chan_api = skills.compile_prompt(CHANNEL_API_SKILL_ID, separator="\n\n")
        return base + "\n\n----- API REFERENCE -----\n\n" + node_api + "\n\n" + chan_api

    if branch == GraphBuilderBranch.GENERATE:
        base = skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.plan",
            "graph_builder.codegen",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )
        node_api = skills.compile_prompt(NODE_API_SKILL_ID, separator="\n\n")
        chan_api = skills.compile_prompt(CHANNEL_API_SKILL_ID, separator="\n\n")
        art_api = skills.compile_prompt(ARTIFACT_API_SKILL_ID, separator="\n\n")
        graphify_style = skills.compile_prompt(GRAPHIFY_STYLE_SKILL_ID, separator="\n\n")
        ckpt = skills.compile_prompt(CHECKPOINT_SKILL_ID, separator="\n\n")
        return (
            base
            + "\n\n----- API REFERENCE -----\n\n"
            + node_api
            + "\n\n"
            + chan_api
            + "\n\n"
            + art_api
            + "\n\n----- PATTERNS -----\n\n"
            + graphify_style
            + "\n\n"
            + ckpt
        )

    if branch == GraphBuilderBranch.CHAT:
        return skills.compile_prompt(
            GRAPH_BUILDER_SKILL_ID,
            "graph_builder.system",
            "graph_builder.chat",
            "graph_builder.style",
            separator="\n\n",
            fallback_keys=["graph_builder.system"],
        )

    return skills.compile_prompt(
        GRAPH_BUILDER_SKILL_ID,
        "graph_builder.system",
        "graph_builder.register",
        "graph_builder.style",
        separator="\n\n",
        fallback_keys=["graph_builder.system"],
    )


def _plan_card(plan: dict[str, Any]) -> dict[str, Any]:
    graph = plan.get("graph") or {}
    tools = plan.get("tools") or []
    checkpoints = plan.get("checkpointing") or []
    needs = plan.get("needs") or {}
    return {
        "kind": "card",
        "title": f"Plan: {graph.get('name') or 'proposed_graph'}",
        "sections": [
            {"label": "Type", "value": str(graph.get("type") or "graphify")},
            {
                "label": "I/O",
                "value": f"in={graph.get('inputs') or []} | out={graph.get('outputs') or []}",
            },
            {"label": "Tools", "value": f"{len(tools)} planned"},
            {"label": "Checkpointing", "value": f"{len(checkpoints)} step(s)"},
            {"label": "Needs", "value": json.dumps(needs, ensure_ascii=False)},
            {
                "label": "Tool List",
                "value": [str(t.get("name") or "unnamed_tool") for t in tools] or ["(none)"],
            },
            {
                "label": "Checkpoint List",
                "value": [str(c.get("tool") or "unknown_tool") for c in checkpoints] or ["(none)"],
            },
        ],
    }


def _extract_python_code(markdown_text: str) -> str | None:
    text = markdown_text or ""

    # Preferred: explicit Python fenced blocks.
    for m in re.finditer(r"```(python|py)\s*(.*?)(?:```|$)", text, flags=re.DOTALL | re.IGNORECASE):
        code = (m.group(2) or "").strip()
        if code:
            return code

    # Fallback: unlabeled fenced block that looks like Python source.
    for m in re.finditer(r"```[^\n]*\n(.*?)(?:```|$)", text, flags=re.DOTALL):
        candidate = (m.group(1) or "").strip()
        if not candidate:
            continue
        if re.search(
            r"(^from\s+\S+\s+import\s+|^import\s+\S+|^@(?:tool|graphify|graph_fn)\b|^\s*def\s+\w+\s*\(|^\s*async\s+def\s+\w+\s*\()",
            candidate,
            flags=re.MULTILINE,
        ):
            return candidate

    # Final fallback: harvest from a Python implementation section even if fencing is malformed/missing.
    m_section = re.search(
        r"(?:^|\n)\s*python implementation\s*:?\s*\n(.*)$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if m_section:
        tail = (m_section.group(1) or "").strip()
        m_manifest = re.search(
            r"(?:^|\n)\s*(?:file manifest|manifest)\s*:?\s*\n",
            tail,
            flags=re.IGNORECASE,
        )
        if m_manifest:
            tail = tail[: m_manifest.start()].rstrip()
        if tail:
            return tail

    return None


def _extract_plan_json(markdown_text: str) -> dict[str, Any] | None:
    text = markdown_text or ""

    # Preferred: explicit JSON fenced block.
    for m in re.finditer(r"```json\s*(.*?)(?:```|$)", text, flags=re.DOTALL | re.IGNORECASE):
        block = (m.group(1) or "").strip()
        if not block:
            continue
        try:
            obj = json.loads(block)
        except Exception:
            continue
        if isinstance(obj, dict):
            if isinstance(obj.get("plan"), dict):
                return obj["plan"]
            if isinstance(obj.get("graph"), dict):
                return obj

    # Fallback: scan arbitrary text for a decodable JSON object.
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except Exception:
            idx = start + 1
            continue
        idx = start + max(end, 1)
        if isinstance(obj, dict):
            if isinstance(obj.get("plan"), dict):
                return obj["plan"]
            if isinstance(obj.get("graph"), dict):
                return obj
    return None


async def _resolve_codegen_llm(*, context: NodeContext) -> GenericLLMClient:
    try:
        return context.llm("coding")
    except Exception:
        context.logger().warning(
            "graph_builder_v2: context.llm('coding') unavailable; fallback to default"
        )
        return context.llm()


async def _handle_plan_v2(
    *,
    message: str,
    files_summary: str,
    context: NodeContext,
    show_approval_buttons: bool = True,
) -> tuple[str, dict[str, Any] | None]:
    llm = context.llm()
    chan = context.ui_session_channel()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.PLAN)
    history = await _recent_chat_for_llm(context=context, limit=20)

    user_prompt = (
        "Produce plan only. No implementation code.\n"
        'Return strict JSON with shape: {"explanation":"...","plan":{...}}.\n\n'
        f"User request:\n{message}\n\n"
        f"Files summary:\n{files_summary}\n"
    )

    await chan.send_phase(phase="thinking", status="active", label="Building plan...")
    try:
        resp, _usage = await llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                *history,
                {"role": "user", "content": user_prompt},
            ],
            output_format="json",
            json_schema=BUILDER_PLAN_WRAPPER_SCHEMA,
            schema_name="BuilderPlanWrapperV2",
            strict_schema=True,
            validate_json=True,
            max_output_tokens=2048,
        )
    except Exception:
        await chan.send_phase(phase="thinking", status="failed", label="Planning failed.")
        raise

    obj = json.loads(resp) if isinstance(resp, str) else resp
    explanation = str(obj.get("explanation") or "").strip()
    plan = obj.get("plan") or {}

    await chan.send_rich(
        text=explanation or "I drafted a build plan. Review and choose the next action.",
        rich=_plan_card(plan),
        memory_log=False,
    )
    if show_approval_buttons:
        await chan.send_buttons(
            text="Approve this plan to generate code, or replan with more details.",
            buttons=[
                Button(label="Proceed", value="proceed"),
                Button(label="Replan", value="replan"),
            ],
            memory_log=False,
        )
    else:
        await chan.send_text(
            "I drafted a plan from current context and will continue directly to generation.",
            memory_log=False,
        )
    await chan.send_phase(phase="thinking", status="done", label="Plan is ready.")

    if plan:
        state = await _load_state(context=context, level="user")
        state.plan_ver += 1
        state.pending_plan_json = plan
        state.pending_action = "awaiting_plan_approval"
        state.last_graph_name = (plan.get("graph") or {}).get("name")
        state.last_contract_hash = _hash_contract(plan)
        await _save_state(context=context, state=state)

    return (
        "Plan prepared. Click Proceed to generate code, or Replan and share additional requirements.",
        plan,
    )


async def _handle_generate_v2(
    *,
    message: str,
    files_summary: str,
    files: list[Any],
    context: NodeContext,
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    chan = context.ui_session_channel()
    state = await _load_state(context=context, level="user")
    plan = state.pending_plan_json or state.last_plan_json
    if not plan:
        await chan.send_phase(
            phase="thinking",
            status="active",
            label="No plan found. Drafting a plan first...",
        )
        plan_message = (message or "").strip()
        if plan_message.lower().startswith("/gen"):
            remainder = plan_message[4:].strip()
            plan_message = remainder or (
                "Draft a practical build plan from prior chat context and available files."
            )
        _plan_reply, auto_plan = await _handle_plan_v2(
            message=plan_message,
            files_summary=files_summary,
            context=context,
            show_approval_buttons=False,
        )
        state = await _load_state(context=context, level="user")
        plan = state.pending_plan_json or state.last_plan_json or auto_plan
    if not plan:
        return (
            "I do not have an approved plan yet. Please run /plan first or provide requirements for planning.",
            None,
            None,
            None,
        )

    llm = await _resolve_codegen_llm(context=context)
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.GENERATE)
    history = await _recent_chat_for_llm(context=context, limit=16)

    user_prompt = (
        "Generate implementation code from the approved plan.\n"
        "Do not stream output instructions.\n"
        "Return Markdown with: brief explanation, plan JSON block, python code block, file manifest.\n\n"
        f"User message:\n{message}\n\n"
        f"Approved plan:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"Files summary:\n{files_summary}\n"
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": user_prompt},
    ]
    for f in files:
        try:
            text = await context.artifacts().load_text(uri=f.uri)
        except Exception:
            context.logger().warning(
                "graph_builder_v2: skipping non-text artifact in codegen prompt: %s", f
            )
            continue
        messages.append({"role": "system", "content": f"\n--- Relevant file: {f.name} ---\n{text}"})

    await chan.send_phase(phase="thinking", status="active", label="Generating code artifact...")
    try:
        resp_text, _usage = await llm.chat(
            messages=messages,
            max_output_tokens=4096,
            request_timeout_s=240,
        )
    except Exception:
        await chan.send_phase(phase="thinking", status="failed", label="Code generation failed.")
        raise
    print("🍎 Graph Builder: Code generation LLM response:\n", resp_text)

    new_plan = _extract_plan_json(resp_text)

    code = _extract_python_code(resp_text)
    if not code:
        await chan.send_phase(
            phase="thinking", status="failed", label="No Python code block found."
        )
        await chan.send_text(
            "I expected a Python code block in the generation output but couldn't find one. "
            "Please ensure the LLM response includes a code block like ```python ... ```."
        )
        return (
            "I could not extract Python code from the generation output. Please replan or retry.",
            None,
            None,
            None,
        )

    graph_name = (
        (new_plan or plan).get("graph", {}).get("name")
        if isinstance((new_plan or plan).get("graph"), dict)
        else None
    ) or "generated_graph"
    filename = f"{graph_name}_v{max(state.graph_ver + 1, 1)}.py"
    await chan.send_file(
        file_bytes=code.encode("utf-8"),
        filename=filename,
        title=f"Generated code: {filename}",
        memory_log=False,
    )
    await chan.send_buttons(
        text="Code generated. Do you want to register it as an app?",
        buttons=[
            Button(label="Register App", value="register"),
            Button(label="Skip", value="skip"),
        ],
        memory_log=False,
    )
    await chan.send_phase(phase="thinking", status="done", label="Code generation complete.")

    latest_state = await _load_state(context=context, level="user")
    merged_plan = new_plan or plan
    last_artifact = getattr(context.artifacts(), "last_artifact", None)
    generated_artifact_id = getattr(last_artifact, "artifact_id", None) if last_artifact else None
    generated_artifact_uri = getattr(last_artifact, "uri", None) if last_artifact else None
    generated_code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
    if merged_plan:
        latest_state.plan_ver = max(latest_state.plan_ver, 1)
        latest_state.graph_ver += 1
        latest_state.last_plan_json = merged_plan
        latest_state.pending_plan_json = None
        latest_state.last_graph_name = (merged_plan.get("graph") or {}).get("name")
        latest_state.last_contract_hash = _hash_contract(merged_plan)
        latest_state.pending_action = "awaiting_register_decision"
        latest_state.last_generated_code = code
        latest_state.last_generated_filename = filename
        latest_state.last_generated_artifact_id = generated_artifact_id
        latest_state.last_generated_artifact_uri = generated_artifact_uri
        latest_state.last_generated_code_sha256 = generated_code_sha256
        latest_state.last_generated_files = [{"name": filename, "kind": "python"}]
        await _save_state(context=context, state=latest_state)

    return (
        "I generated the code and sent it as a file.",
        merged_plan,
        code,
        filename,
    )


def _to_builder_file_ref(file_obj: Any) -> BuilderFileRef | None:
    if isinstance(file_obj, BuilderFileRef):
        return file_obj

    if isinstance(file_obj, dict):
        name = file_obj.get("name") or file_obj.get("filename")
        mimetype = file_obj.get("mimetype") or file_obj.get("type") or file_obj.get("content_type")
        source = file_obj.get("source")
        artifact_id = file_obj.get("artifact_id")
        uri = file_obj.get("uri")
        kind = file_obj.get("kind")
    else:
        name = getattr(file_obj, "name", None) or getattr(file_obj, "filename", None)
        mimetype = (
            getattr(file_obj, "mimetype", None)
            or getattr(file_obj, "type", None)
            or getattr(file_obj, "content_type", None)
        )
        source = getattr(file_obj, "source", None)
        artifact_id = getattr(file_obj, "artifact_id", None)
        uri = getattr(file_obj, "uri", None)
        kind = getattr(file_obj, "kind", None)

    if source not in {"upload", "artifact"}:
        source = "artifact" if artifact_id else "upload"
    if kind not in {"code", "notebook", "text", "other"}:
        kind = _classify_builder_file(name=name, mimetype=mimetype)

    return BuilderFileRef(
        name=name,
        mimetype=mimetype,
        source=source,
        artifact_id=artifact_id,
        uri=uri,
        kind=kind,
    )


def _normalize_builder_file_refs(files: list[Any] | None) -> list[BuilderFileRef]:
    out: list[BuilderFileRef] = []
    for f in files or []:
        ref = _to_builder_file_ref(f)
        if ref is not None:
            out.append(ref)
    return out


def _register_candidate_score(f: BuilderFileRef) -> tuple[int, int, int]:
    return (
        2 if f.source == "artifact" else 0,
        10 if f.kind == "code" else (5 if f.kind == "notebook" else (2 if f.kind == "text" else 0)),
        1 if (f.artifact_id or f.uri) else 0,
    )


async def _load_registration_source_text(
    *, context: NodeContext, file_ref: BuilderFileRef
) -> tuple[str | None, str | None]:
    artifacts = context.artifacts()
    if file_ref.artifact_id:
        try:
            return await artifacts.load_text_by_id(file_ref.artifact_id), None
        except Exception as e:
            return None, f"artifact_id `{file_ref.artifact_id}`: {e!r}"
    if file_ref.uri:
        try:
            return await artifacts.load_text(uri=file_ref.uri), None
        except Exception as e:
            return None, f"uri `{file_ref.uri}`: {e!r}"
    return None, "missing artifact_id and uri"


def _format_validation_issues(issues: list[Any]) -> str:
    lines: list[str] = []
    for issue in issues:
        code = getattr(issue, "code", "validation_issue")
        message = getattr(issue, "message", str(issue))
        lines.append(f"- {code}: {message}")
    return "\n".join(lines)


async def _handle_register_app_v2(
    *,
    message: str,
    files: list[BuilderFileRef] | None,
    context: NodeContext,
) -> str:
    """
    Register the selected graph as an app so the UI can discover it.

    The registration path is:
    1) Normalize all file inputs into BuilderFileRef.
    2) Validate source before registration.
    3) Prefer register_by_artifact(..., app_config=...) when artifact/uri is available.
    4) Fall back to in-memory registration if only generated source exists.
    """
    logger = context.logger()
    chan = context.ui_session_channel()
    registry = context.registry()

    state = await _load_state(context=context, level="user")
    if state.pending_action == "awaiting_register_decision":
        state.pending_action = None

    normalized_files = _normalize_builder_file_refs(files)
    if state.last_generated_artifact_id or state.last_generated_artifact_uri:
        normalized_files.append(
            BuilderFileRef(
                name=state.last_generated_filename,
                mimetype="text/x-python",
                source="artifact",
                artifact_id=state.last_generated_artifact_id,
                uri=state.last_generated_artifact_uri,
                kind="code",
            )
        )

    graph_name: str | None = None
    code_text: str | None = None
    selected_ref: BuilderFileRef | None = None
    validation_notes: list[str] = []

    prioritized_files = sorted(normalized_files, key=_register_candidate_score, reverse=True)
    for file_ref in prioritized_files:
        if not (file_ref.artifact_id or file_ref.uri):
            continue

        source_text, load_error = await _load_registration_source_text(
            context=context,
            file_ref=file_ref,
        )
        if not source_text:
            validation_notes.append(
                f"- Could not load source from `{file_ref.name or '<unnamed>'}` ({load_error})."
            )
            continue

        vr = registry.validate_graphify_source(
            source_text,
            filename=file_ref.name or "artifact.py",
            strict=False,
        )
        if not vr.ok:
            validation_notes.append(
                f"- Source in `{file_ref.name or '<unnamed>'}` is not a valid @graphify/@graph_fn module:\n"
                f"{_format_validation_issues(vr.issues)}"
            )
            continue

        probe = await registry.register_by_artifact(
            artifact_id=file_ref.artifact_id,
            uri=file_ref.uri,
            persist=False,
            strict=False,
        )
        if not probe.success or not probe.graph_name:
            validation_notes.append(
                f"- Source in `{file_ref.name or '<unnamed>'}` could not be registered: "
                f"{'; '.join(probe.errors) if probe.errors else 'unknown error'}"
            )
            continue

        graph_name = probe.graph_name
        code_text = source_text
        selected_ref = file_ref
        break

    if not graph_name:
        graph_name, code_text = await _resolve_registration_target(
            context=context,
            files=normalized_files,
        )
        if code_text:
            vr = registry.validate_graphify_source(
                code_text,
                filename=(state.last_generated_filename or "generated.py"),
                strict=False,
            )
            if not vr.ok:
                await _save_state(context=context, state=state)
                return (
                    "Registration failed because the candidate source is not a valid "
                    "@graphify/@graph_fn module.\n\n"
                    f"{_format_validation_issues(vr.issues)}"
                )

    if not graph_name:
        await _save_state(context=context, state=state)
        msg = (
            "I couldn't find a graph to register.\n\n"
            "- Provide a file containing a @graphify/@graph_fn, or\n"
            "- Re-run generation so I have a recent graph to register."
        )
        if validation_notes:
            msg += "\n\nValidation details:\n" + "\n".join(validation_notes)
        return msg

    if code_text:
        # hotload the graph to validate it before registration, and to extract metadata for app config proposal
        await _ensure_graph_registered_from_code(
            context=context,
            graph_name=graph_name,
            code_text=code_text,
        )

    plan = state.last_plan_json or state.pending_plan_json or {}
    app_config = await _propose_app_config_via_llm(
        context=context,
        graph_name=graph_name,
        plan=plan,
        user_message=message or "",
    )

    app_id = app_config.get("id") or graph_name
    app_version = "0.1.0"
    flow_id = app_config.get("flow_id") or graph_name

    if selected_ref and (selected_ref.artifact_id or selected_ref.uri):
        reg_result = await registry.register_by_artifact(
            artifact_id=selected_ref.artifact_id,
            uri=selected_ref.uri,
            app_config=app_config,
            persist=True,
            strict=False,
        )
        if not reg_result.success:
            await _save_state(context=context, state=state)
            return (
                "I validated the source but registration failed.\n\n"
                f"Errors:\n- {'; '.join(reg_result.errors) if reg_result.errors else 'unknown error'}"
            )
        app_id = reg_result.app_id or app_id
        app_version = reg_result.version or app_version
        flow_id = app_config.get("flow_id") or reg_result.graph_name or graph_name
    else:
        registry.register(
            nspace="app",
            name=app_id,
            version=app_version,
            obj=app_config,
            meta={
                "flow_id": flow_id,
                "graph_name": graph_name,
                "builder": "graph_builder_v2",
            },
        )

    try:
        registry.alias(nspace="app", name=app_id, tag="stable", to_version=app_version)
    except Exception:
        logger.debug(
            "graph_builder_v2: could not alias app %s@%s as 'stable'",
            app_id,
            app_version,
        )

    state.last_registered_app_id = app_id
    state.last_registered_app_version = app_version
    await _save_state(context=context, state=state)

    await chan.send_text(
        f"Registered app **{app_config.get('name', app_id)}** "
        f"(id=`{app_id}`, flow_id=`{flow_id}`).\n\n"
        "You can now run it from the App Gallery."
    )

    return (
        f"I've registered the app **{app_config.get('name', app_id)}** "
        f"with id `{app_id}`. You should now see it in the App Gallery, "
        f"backed by flow_id `{flow_id}`."
    )


async def _handle_chat_v2(*, message: str, context: NodeContext) -> str:
    msg = (message or "").strip().lower()
    intent = _detect_approval_intent(msg)
    state = await _load_state(context=context, level="user")
    has_plan = bool(state.pending_plan_json or state.last_plan_json)
    if intent == "approve" and not has_plan:
        return (
            "I do not have a plan yet. Use /plan or share requirements and I will draft one first, "
            "then we can proceed to generation."
        )
    if state.pending_action == "awaiting_plan_approval":
        return "If you want changes, tell me what to adjust and I will replan. If ready, click Proceed."
    if state.pending_action == "awaiting_register_decision" and intent == "decline":
        state.pending_action = None
        await _save_state(context=context, state=state)
        return "Okay, skipping app registration for now."

    llm = context.llm()
    system_prompt = _compile_branch_prompt(context=context, branch=GraphBuilderBranch.CHAT)
    history = await _recent_chat_for_llm(context=context, limit=18)
    text, _ = await llm.chat(
        messages=[
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": message},
        ],
        max_output_tokens=2048,
    )
    return text
