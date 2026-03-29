from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any, get_type_hints

from aethergraph.contracts.errors.errors import GraphValidationError, build_error_hints
from aethergraph.core.graph.action_spec import IOSlot, _map_py_type_to_json_type
from aethergraph.core.graph.graphify_validation import (
    emit_validation_warnings,
    format_validation_errors,
    resolve_validation_source_for_callable,
    validate_graph_source,
    warnings_as_errors_enabled,
)
from aethergraph.services.registry.agent_app_meta import (
    AgentConfig,
    AppConfig,
    build_agent_meta,
    build_app_meta,
)

from ..runtime.injection import pop_explicit_node_context, resolve_node_context_param
from ..runtime.runtime_env import RuntimeEnv
from ..runtime.runtime_registry import current_registry
from .node_spec import TaskNodeSpec

GRAPH_FN_ROOT_NODE_ID = "__graph_fn_root__"


class GraphFunction:
    def __init__(
        self,
        name: str,
        fn: Callable,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        version: str = "0.1.0",
        agent_id: str | None = None,
        app_id: str | None = None,
    ):
        self.graph_id = name
        self.name = name
        self.fn = fn
        self.inputs = inputs or []
        self.outputs = outputs or []
        self.version = version
        self.registry_key: str | None = None
        self.agent_id = agent_id
        self.app_id = app_id
        self._node_context_param = resolve_node_context_param(fn)

    async def run(self, *, env: RuntimeEnv | None = None, **inputs):
        """
        Execute this graph function directly as native Python with a synthetic root context.

        `graph_fn` is a runtime wrapper, not a DAG builder. It does not materialize a
        TaskGraph or register a scheduler, and it does not support wait/resume.
        """
        if env is None:
            from ..runtime.graph_runner import _build_env

            inherited_identity = None
            env_inputs = dict(inputs)
            parent_ctx = pop_explicit_node_context(env_inputs)
            if parent_ctx is not None:
                inherited_identity = getattr(parent_ctx, "identity", None)
            env, _retry, _max_concurrency = await _build_env(
                self,
                env_inputs,
                identity=inherited_identity,
                run_id=getattr(parent_ctx, "run_id", None),
                session_id=getattr(parent_ctx, "session_id", None),
                agent_id=getattr(parent_ctx, "agent_id", None),
                app_id=getattr(parent_ctx, "app_id", None),
            )

        node_spec = TaskNodeSpec(
            node_id=GRAPH_FN_ROOT_NODE_ID,
            type="graph_fn_root",
            metadata={"synthetic": True},
            tool_name=self.name,
            tool_version=self.version,
        )
        runtime_ctx = env.make_ctx(
            node=node_spec, resume_payload=getattr(env, "resume_payload", None)
        )
        node_ctx = runtime_ctx.create_node_context(node=node_spec)

        call_kwargs = dict(inputs)
        parent_ctx = pop_explicit_node_context(call_kwargs)
        if self._node_context_param is not None:
            call_kwargs.setdefault(self._node_context_param, parent_ctx or node_ctx)

        res = self.fn(**call_kwargs)
        if inspect.isawaitable(res):
            res = await res

        return _normalize_graph_fn_outputs(res, self.outputs)

    async def __call__(self, **inputs):
        """Async call to run the graph function."""
        from ..runtime.graph_runner import run_async

        return await run_async(self, inputs)

    def sync(self, **inputs):
        """Synchronous wrapper around async run(). Useful for quick tests or scripts."""
        from ..runtime.graph_runner import run

        return run(self, inputs)

    def io_signature(self) -> dict[str, list[IOSlot]]:
        """
        Infer typed IO based on decorator inputs/outputs and Python annotations.

        This metadata is used for registration and UI surfaces only. `graph_fn`
        execution itself is plain Python at runtime.
        """
        sig = inspect.signature(self.fn)
        hints = get_type_hints(self.fn)

        if self.inputs is not None:
            input_names = list(self.inputs)
        else:
            input_names = [p for p in sig.parameters if p != self._node_context_param]

        input_slots: list[IOSlot] = []
        for name in input_names:
            param = sig.parameters.get(name)
            if param is None:
                input_slots.append(IOSlot(name=name, type=None, required=True))
                continue

            anno = hints.get(name)
            j_type = _map_py_type_to_json_type(anno) if anno is not None else None
            required = param.default is inspect._empty
            default = None if required else param.default

            input_slots.append(
                IOSlot(
                    name=name,
                    type=j_type,
                    required=required,
                    default=default,
                )
            )

        output_slots: list[IOSlot] = []
        out_names = list(self.outputs or [])

        ret_anno = hints.get("return")
        if out_names:
            if len(out_names) == 1 and ret_anno is not None:
                j_type = _map_py_type_to_json_type(ret_anno)
                output_slots.append(IOSlot(name=out_names[0], type=j_type, required=True))
            else:
                for name in out_names:
                    output_slots.append(IOSlot(name=name, type=None, required=True))
        elif ret_anno is not None:
            j_type = _map_py_type_to_json_type(ret_anno)
            output_slots.append(IOSlot(name="result", type=j_type, required=True))

        return {"inputs": input_slots, "outputs": output_slots}


def _is_ref(x: object) -> bool:
    return isinstance(x, dict) and x.get("_type") == "ref" and "from" in x and "key" in x


def _is_nodehandle(x: object) -> bool:
    return hasattr(x, "node_id") and hasattr(x, "output_keys")


def _assert_plain_runtime_value(value: object, *, key: str | None = None) -> None:
    if _is_ref(value) or _is_nodehandle(value):
        label = f" for output '{key}'" if key else ""
        raise ValueError(
            "graph_fn_plain_runtime_only: graph_fn must return plain Python values"
            f"{label}. NodeHandle/ref outputs are only supported in @graphify."
        )


def _normalize_graph_fn_outputs(ret, declared_outputs: list[str] | None) -> dict:
    """
    Normalize graph_fn return values into a plain outputs dict.

    If exactly one output is declared, a single return value is allowed and mapped
    to that output key. Otherwise graph_fn must return a dict.
    """
    if isinstance(ret, dict):
        for k, v in ret.items():
            _assert_plain_runtime_value(v, key=k)
        result = dict(ret)
    else:
        _assert_plain_runtime_value(ret)
        if declared_outputs and len(declared_outputs) == 1:
            result = {declared_outputs[0]: ret}
        elif not declared_outputs:
            result = {"result": ret}
        else:
            raise ValueError(
                "graph_fn_result_shape_invalid: graph_fn must return a dict unless exactly "
                "one output is declared."
            )

    if declared_outputs:
        result = {k: result[k] for k in declared_outputs if k in result}
        missing = [k for k in declared_outputs if k not in result]
        if missing:
            raise ValueError(f"Missing declared outputs: {missing}")

    return result


def graph_fn(
    name: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    version: str = "0.1.0",
    *,
    entrypoint: bool = False,
    flow_id: str | None = None,
    tags: list[str] | None = None,
    as_agent: AgentConfig | None = None,
    as_app: AppConfig | None = None,
    description: str | None = None,
) -> Callable[[Callable], GraphFunction]:
    """
    Decorator to define a runtime graph function and optionally register it as an agent or app.

    `graph_fn` is the flexible runtime surface for native Python orchestration with
    injected `NodeContext`. Use `@graphify` for rigid DAG execution, persistence, and
    wait/resume behavior.
    """

    def decorator(fn: Callable) -> GraphFunction:
        source, source_name = resolve_validation_source_for_callable(fn)
        validation = validate_graph_source(
            source,
            filename=source_name,
            strict=True,
            warnings_as_errors=warnings_as_errors_enabled(),
        )
        if not validation.ok:
            error_result = type(validation)(
                ok=False,
                issues=[i for i in validation.issues if i.severity != "warning"],
                graph_names=validation.graph_names,
                graphfn_names=validation.graphfn_names,
            )
            if error_result.issues:
                message = format_validation_errors(error_result, filename=source_name)
                hints: list[dict[str, str]] = []
                for issue in error_result.issues:
                    hints.extend(build_error_hints(issue.code, issue.message))
                primary_code = (
                    error_result.issues[0].code
                    if len(error_result.issues) == 1
                    else "graph_validation_failed"
                )
                raise GraphValidationError(message, code=primary_code, hints=hints)
        emit_validation_warnings(validation, filename=source_name)

        agent_id = as_agent.get("id") if as_agent else None
        app_id = as_app.get("id") if as_app else None

        def _build_graph_fn() -> GraphFunction:
            return GraphFunction(
                name=name,
                fn=fn,
                inputs=inputs,
                outputs=outputs,
                version=version,
                agent_id=agent_id,
                app_id=app_id,
            )

        _build_graph_fn.__ag_builder__ = True
        gf = _build_graph_fn()
        registry = current_registry()

        if registry is None:
            return gf

        base_tags = tags or []
        doc_desc = inspect.getdoc(fn) or None
        eff_description = description or doc_desc or name

        graph_meta: dict[str, Any] = {
            "kind": "graphfn",
            "entrypoint": entrypoint,
            "flow_id": flow_id,
            "tags": base_tags,
            "description": eff_description,
            "inputs": inputs,
            "outputs": outputs,
        }

        registry.register(
            nspace="graphfn",
            name=name,
            version=version,
            obj=_build_graph_fn,
            meta=graph_meta,
        )

        agent_meta = build_agent_meta(
            graph_name=name,
            version=version,
            graph_meta=graph_meta,
            agent_cfg=as_agent,
        )
        if agent_meta is not None:
            registry.register(
                nspace="agent",
                name=agent_meta["id"],
                version=version,
                obj=_build_graph_fn,
                meta=agent_meta,
            )

        app_meta = build_app_meta(
            graph_name=name,
            version=version,
            graph_meta=graph_meta,
            app_cfg=as_app,
        )
        if app_meta is not None:
            registry.register(
                nspace="app",
                name=app_meta["id"],
                version=version,
                obj=_build_graph_fn,
                meta=app_meta,
            )

        return gf

    return decorator
