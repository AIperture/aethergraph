from __future__ import annotations

from typing import Literal


class AetherGraphError(Exception):
    """Base class for all AetherGraph errors."""


class NodeContractError(AetherGraphError):
    """Raised when a TaskNodeRuntime violates its declared input/output contract."""


class MissingInputError(NodeContractError):
    """Raised when a required input key is missing."""


class MissingOutputError(NodeContractError):
    """Raised when a required output key is missing."""


class ExecutionError(AetherGraphError):
    """Raised when a node's logic fails during execution."""


def build_error_hints(code: str, message: str | None = None) -> list[dict[str, str]]:
    """
    Map stable build-error codes to machine-readable DX hints.
    """
    msg = (message or "").lower()
    hints: list[dict[str, str]] = []

    code_map: dict[str, list[tuple[str, str]]] = {
        "graphify_async_def": [
            (
                "use_sync_graphify",
                "@graphify must decorate `def`, not `async def`. Move async work into @tool.",
            )
        ],
        "missing_decorator_kw": [
            (
                "include_required_graph_kwargs",
                "Include `name`, `inputs`, and `outputs` in @graphify/@graph_fn decorators.",
            )
        ],
        "graphify_control_flow_non_deterministic": [
            (
                "use_declarative_condition",
                "Use `_condition` expressions or move dynamic branching into @graph_fn/@tool.",
            )
        ],
        "graphify_unsupported_condition_expr": [
            (
                "condition_expr_shape",
                "Use bool/name or declarative dict expressions for `_condition`.",
            )
        ],
        "tool_nested_tool_call_disallowed": [
            (
                "orchestration_location",
                "Move orchestration to @graphify/@graph_fn; keep @tool bodies self-contained.",
            )
        ],
        "graphify_plain_call_used_as_handle": [
            (
                "wrap_callable_with_tool",
                "Values treated as node handles should come from @tool calls.",
            )
        ],
        "run_async_invalid_target": [
            (
                "target_kind",
                "Pass a TaskGraph/GraphFunction, a builder with `.build()`, or a zero-arg callable returning one.",
            )
        ],
        "run_async_target_not_task_graph": [
            (
                "taskgraph_required",
                "This path requires a TaskGraph target. Use run_async(graph_fn, ...) for GraphFunction execution.",
            )
        ],
        "graph_inputs_missing_required": [
            (
                "provide_required_inputs",
                "Provide all required graph inputs declared in the graph IO signature.",
            )
        ],
        "resume_snapshot_policy_violation": [
            (
                "json_only_resume_outputs",
                "Resume requires JSON-safe outputs; remove non-JSON values from checkpointed outputs.",
            )
        ],
        "graph_validation_failed": [
            (
                "fix_validation_issues",
                "Address graph validation issues and retry registration/execution.",
            )
        ],
    }

    for h_code, h_message in code_map.get(code, []):
        hints.append({"code": h_code, "message": h_message})

    if "missing required inputs" in msg and not any(
        h.get("code") == "provide_required_inputs" for h in hints
    ):
        hints.append(
            {
                "code": "provide_required_inputs",
                "message": "Provide all required graph inputs before execution.",
            }
        )

    if ("validation failed" in msg or "graph source validation failed" in msg) and not any(
        h.get("code") == "fix_validation_issues" for h in hints
    ):
        hints.append(
            {
                "code": "fix_validation_issues",
                "message": "Review validation diagnostics and fix the reported graph-definition issues.",
            }
        )

    return hints


class GraphBuildError(AetherGraphError, ValueError):
    """
    Raised for graph build/materialization/input-binding/resume-guard failures.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        stage: Literal["validation", "materialization", "input_bind", "resume_guard"],
        hints: list[dict[str, str]] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.hints = hints or build_error_hints(code, message)
        self.cause = cause

    def __str__(self) -> str:
        return f"[build:{self.stage}:{self.code}] {super().__str__()}"


class GraphValidationError(GraphBuildError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "graph_validation_failed",
        hints: list[dict[str, str]] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message,
            code=code,
            stage="validation",
            hints=hints,
            cause=cause,
        )


class GraphMaterializationError(GraphBuildError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "run_async_invalid_target",
        hints: list[dict[str, str]] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message,
            code=code,
            stage="materialization",
            hints=hints,
            cause=cause,
        )


class GraphInputBindError(GraphBuildError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "graph_inputs_missing_required",
        hints: list[dict[str, str]] | None = None,
        cause: Exception | None = None,
    ):
        super().__init__(
            message,
            code=code,
            stage="input_bind",
            hints=hints,
            cause=cause,
        )


class GraphHasPendingWaits(RuntimeError):
    """Raised when attempting to finalize a graph that has pending waits."""

    def __init__(
        self, message: str, waiting_nodes: list[str], continuations: list[dict] | None = None
    ):
        super().__init__(message)
        self.waiting_nodes = waiting_nodes
        self.continuations = continuations or []


class ResumeIncompatibleSnapshot(RuntimeError):
    """
    Raised when a snapshot is not allowed for resume under the current policy
    (e.g., contains non-JSON outputs or external refs like __aether_ref__).
    """

    def __init__(self, run_id: str, reasons: list[str]):
        super().__init__(f"Resume blocked for run_id={run_id}. " + " / ".join(reasons))
        self.run_id = run_id
        self.reasons = reasons
