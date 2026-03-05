from __future__ import annotations

import ast
from dataclasses import dataclass, field
import inspect
import logging
import os
from pathlib import Path
import warnings

_LOG = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    code: str
    message: str
    line: int | None = None
    col: int | None = None
    end_line: int | None = None
    end_col: int | None = None
    severity: str = "error"
    suggestion: str | None = None
    symbol: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    graph_names: list[str] = field(default_factory=list)
    graphfn_names: list[str] = field(default_factory=list)


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _call_kwarg_names(node: ast.AST) -> set[str]:
    if not isinstance(node, ast.Call):
        return set()
    return {kw.arg for kw in node.keywords if kw.arg}


def _extract_name_kw(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    for kw in node.keywords:
        if (
            kw.arg == "name"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            return kw.value.value.strip()
    return None


def _is_tool_decorator(dec: ast.AST) -> bool:
    return _decorator_name(dec) == "tool"


def _collect_decorated_function_names(tree: ast.AST, decorator_name: str) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node,
            (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),  # noqa: UP038
        ):
            continue
        if any(_decorator_name(dec) == decorator_name for dec in node.decorator_list):
            out.add(node.name)
    return out


def _collect_name_loads(node: ast.AST) -> set[str]:
    loads: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            loads.add(n.id)
    return loads


def _is_supported_if_test(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, bool)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _is_supported_if_test(node.operand)
    if isinstance(node, ast.Compare):
        allowed_ops = (ast.Eq, ast.NotEq, ast.In, ast.NotIn, ast.Is, ast.IsNot)
        if any(not isinstance(op, allowed_ops) for op in node.ops):
            return False

        def _simple_expr(x: ast.AST) -> bool:
            if isinstance(x, (ast.Name, ast.Constant, ast.Attribute)):  # noqa: UP038
                return True
            if isinstance(x, ast.Subscript):
                return _simple_expr(x.value)
            return False

        return _simple_expr(node.left) and all(_simple_expr(c) for c in node.comparators)
    return False


def validate_graph_source(
    source: str,
    *,
    filename: str | None = None,
    strict: bool = True,
    warnings_as_errors: bool = False,
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    graph_names: list[str] = []
    graphfn_names: list[str] = []
    display_name = filename or "<source>"

    try:
        tree = ast.parse(source, filename=display_name)
    except SyntaxError as e:
        issues.append(
            ValidationIssue(
                code="syntax_error",
                message=e.msg or "Syntax error",
                line=e.lineno,
                col=e.offset,
            )
        )
        return ValidationResult(ok=False, issues=issues)

    saw_decorator = False
    tool_names = _collect_decorated_function_names(tree, "tool")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # noqa: UP038
            continue

        is_tool_fn = any(_is_tool_decorator(dec) for dec in node.decorator_list)
        if is_tool_fn:
            for n in ast.walk(node):
                if not isinstance(n, ast.Call):
                    continue
                callee = _decorator_name(n.func)
                if callee and callee in tool_names:
                    issues.append(
                        ValidationIssue(
                            code="tool_nested_tool_call_disallowed",
                            message=(
                                f"Nested @tool call '{callee}(...)' inside tool '{node.name}' is not supported."
                            ),
                            line=getattr(n, "lineno", None),
                            col=getattr(n, "col_offset", None),
                            end_line=getattr(n, "end_lineno", None),
                            end_col=getattr(n, "end_col_offset", None),
                            symbol=node.name,
                            suggestion="Move orchestration into @graphify/@graph_fn.",
                        )
                    )

        for dec in node.decorator_list:
            dec_name = _decorator_name(dec)
            if dec_name not in {"graphify", "graph_fn"}:
                continue
            saw_decorator = True
            kw_names = _call_kwarg_names(dec)
            required = {"name", "inputs", "outputs"}
            missing = required - kw_names
            if missing:
                issues.append(
                    ValidationIssue(
                        code="missing_decorator_kw",
                        message=f"{dec_name} is missing required kwargs: {sorted(missing)}",
                        line=getattr(dec, "lineno", None),
                        col=getattr(dec, "col_offset", None),
                    )
                )

            if dec_name == "graphify" and isinstance(node, ast.AsyncFunctionDef):
                issues.append(
                    ValidationIssue(
                        code="graphify_async_def",
                        message="@graphify must decorate sync def, not async def",
                        line=node.lineno,
                        col=node.col_offset,
                        end_line=getattr(node, "end_lineno", None),
                        end_col=getattr(node, "end_col_offset", None),
                        symbol=node.name,
                    )
                )

            explicit_name = _extract_name_kw(dec)
            if dec_name == "graphify":
                graph_names.append(explicit_name or node.name)

                for n in ast.walk(node):
                    if isinstance(n, ast.Await):
                        issues.append(
                            ValidationIssue(
                                code="graphify_await_in_sync_body",
                                message="@graphify body cannot use await; graphify must declare DAG only.",
                                line=getattr(n, "lineno", None),
                                col=getattr(n, "col_offset", None),
                                end_line=getattr(n, "end_lineno", None),
                                end_col=getattr(n, "end_col_offset", None),
                                symbol=node.name,
                                suggestion="Move async execution into @tool or use @graph_fn.",
                            )
                        )

                    if isinstance(n, (ast.While, ast.For, ast.AsyncFor, ast.Try, ast.Match)):  # noqa: UP038
                        issues.append(
                            ValidationIssue(
                                code="graphify_control_flow_non_deterministic",
                                message=(
                                    f"Unsupported control flow '{type(n).__name__}' in @graphify '{node.name}'."
                                ),
                                line=getattr(n, "lineno", None),
                                col=getattr(n, "col_offset", None),
                                end_line=getattr(n, "end_lineno", None),
                                end_col=getattr(n, "end_col_offset", None),
                                symbol=node.name,
                                suggestion="Use declarative _condition or switch to @graph_fn.",
                            )
                        )

                    if isinstance(n, ast.If) and not _is_supported_if_test(n.test):
                        issues.append(
                            ValidationIssue(
                                code="graphify_control_flow_non_deterministic",
                                message=(
                                    f"@graphify '{node.name}' has unsupported if-condition shape."
                                ),
                                line=getattr(n, "lineno", None),
                                col=getattr(n, "col_offset", None),
                                end_line=getattr(n, "end_lineno", None),
                                end_col=getattr(n, "end_col_offset", None),
                                symbol=node.name,
                                suggestion="Use simple comparisons or _condition on tool calls.",
                            )
                        )

                    if isinstance(n, ast.Call):
                        for kw in n.keywords or []:
                            if kw.arg != "_condition":
                                continue
                            if not isinstance(
                                kw.value,
                                (ast.Dict, ast.Constant, ast.Name),  # noqa: UP038
                            ):
                                issues.append(
                                    ValidationIssue(
                                        code="graphify_unsupported_condition_expr",
                                        message=(
                                            "Tool _condition must be a bool/name or declarative dict expression."
                                        ),
                                        line=getattr(kw.value, "lineno", None),
                                        col=getattr(kw.value, "col_offset", None),
                                        end_line=getattr(kw.value, "end_lineno", None),
                                        end_col=getattr(kw.value, "end_col_offset", None),
                                        symbol=node.name,
                                    )
                                )

                # Warn for assigned locals that are never read in the graphify body.
                assigned_locals: dict[str, ast.AST] = {}
                for stmt in node.body:
                    if not isinstance(stmt, ast.Assign):
                        continue
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            assigned_locals[target.id] = stmt
                used_locals = _collect_name_loads(node)
                for var_name, stmt in assigned_locals.items():
                    if var_name in used_locals:
                        continue
                    issues.append(
                        ValidationIssue(
                            code="graphify_unused_local_assignment",
                            message=(
                                f"Local '{var_name}' is assigned but never used in @graphify '{node.name}'."
                            ),
                            line=getattr(stmt, "lineno", None),
                            col=getattr(stmt, "col_offset", None),
                            end_line=getattr(stmt, "end_lineno", None),
                            end_col=getattr(stmt, "end_col_offset", None),
                            severity="warning",
                            symbol=node.name,
                            suggestion=(
                                "Remove dead assignments or include the value in return/exposed outputs."
                            ),
                        )
                    )

                # Detect plain function calls later treated like NodeHandle accesses.
                call_assigns: dict[str, tuple[str | None, ast.Call]] = {}
                for stmt in node.body:
                    if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
                        continue
                    if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                        continue
                    var_name = stmt.targets[0].id
                    callee = _decorator_name(stmt.value.func)
                    call_assigns[var_name] = (callee, stmt.value)

                for n in ast.walk(node):
                    base_name = None
                    if (
                        isinstance(n, ast.Attribute)
                        and isinstance(n.value, ast.Name)
                        or isinstance(n, ast.Subscript)
                        and isinstance(n.value, ast.Name)
                    ):
                        base_name = n.value.id
                    if not base_name or base_name not in call_assigns:
                        continue

                    callee, _ = call_assigns[base_name]
                    if callee and callee in tool_names:
                        continue
                    issues.append(
                        ValidationIssue(
                            code="graphify_plain_call_used_as_handle",
                            message=(
                                f"Variable '{base_name}' is from a plain call but used like a node handle/ref."
                            ),
                            line=getattr(n, "lineno", None),
                            col=getattr(n, "col_offset", None),
                            end_line=getattr(n, "end_lineno", None),
                            end_col=getattr(n, "end_col_offset", None),
                            symbol=node.name,
                            suggestion=(
                                "Wrap the callable with @tool or return explicit ref/literal values."
                            ),
                        )
                    )
                    break

                # Warn for risky subscripts on values produced by tool node handles.
                for n in ast.walk(node):
                    if not isinstance(n, ast.Subscript):
                        continue
                    if not (
                        isinstance(n.value, ast.Attribute)
                        and isinstance(n.value.value, ast.Name)
                        and n.value.value.id in call_assigns
                    ):
                        continue
                    if isinstance(n.slice, ast.Slice):
                        continue

                    callee_name, _ = call_assigns[n.value.value.id]
                    if not (callee_name and callee_name in tool_names):
                        continue

                    issues.append(
                        ValidationIssue(
                            code="graphify_risky_subscript_on_tool_output",
                            message=(
                                f"Potentially unsafe subscript access on tool output '{n.value.value.id}.{n.value.attr}[...]' "
                                f"in @graphify '{node.name}'."
                            ),
                            line=getattr(n, "lineno", None),
                            col=getattr(n, "col_offset", None),
                            end_line=getattr(n, "end_lineno", None),
                            end_col=getattr(n, "end_col_offset", None),
                            severity="warning",
                            symbol=node.name,
                            suggestion=(
                                "Move deep dict-key extraction into a @tool or guard with explicit checks (e.g. .get)."
                            ),
                        )
                    )
            else:
                graphfn_names.append(explicit_name or node.name)

    if not saw_decorator:
        issues.append(
            ValidationIssue(
                code="missing_graph_decorator",
                message="No @graphify(...) or @graph_fn(...) decorator found.",
            )
        )

    try:
        compile(source, display_name, "exec")
    except Exception as e:
        issues.append(ValidationIssue(code="compile_error", message=repr(e)))

    has_errors = any(i.severity != "warning" for i in issues)
    has_warnings = any(i.severity == "warning" for i in issues)
    ok = (not has_errors) and (not warnings_as_errors or not has_warnings)
    result = ValidationResult(
        ok=ok, issues=issues, graph_names=graph_names, graphfn_names=graphfn_names
    )
    if strict and not ok:
        return result
    return result


def format_validation_errors(result: ValidationResult, *, filename: str | None = None) -> str:
    title = f"Graph source validation failed for {filename or '<source>'}"
    lines = [title]
    for issue in result.issues:
        loc = ""
        if issue.line is not None:
            loc = f" (line {issue.line}"
            if issue.col is not None:
                loc += f", col {issue.col}"
            loc += ")"
        detail = f"- [{issue.severity}:{issue.code}] {issue.message}{loc}"
        lines.append(detail)
        if issue.suggestion:
            lines.append(f"  suggestion: {issue.suggestion}")
    return "\n".join(lines)


def resolve_validation_source_for_callable(fn) -> tuple[str, str]:
    module = inspect.getmodule(fn)
    module_source = getattr(module, "__aethergraph_source__", None) if module is not None else None
    module_source_name = (
        getattr(module, "__aethergraph_source_name__", None) if module is not None else None
    )
    if isinstance(module_source, str):
        return module_source, str(module_source_name or "<module_source>")

    module_file = getattr(module, "__file__", None)
    if module_file:
        path = Path(module_file)
        if path.exists() and path.is_file():
            try:
                return path.read_text(encoding="utf-8"), str(path)
            except Exception:
                pass

    try:
        return inspect.getsource(fn), inspect.getsourcefile(fn) or "<callable>"
    except Exception as exc:
        fn_name = getattr(fn, "__name__", "<unknown>")
        raise ValueError(
            f"graph_validation_source_unavailable: unable to resolve source for '{fn_name}': {exc!r}"
        ) from exc


def log_validation_issues(result: ValidationResult, *, filename: str | None = None) -> None:
    warning_result = ValidationResult(
        ok=result.ok,
        issues=[i for i in result.issues if i.severity == "warning"],
        graph_names=result.graph_names,
        graphfn_names=result.graphfn_names,
    )
    if warning_result.issues:
        _LOG.warning(format_validation_errors(warning_result, filename=filename))

    error_result = ValidationResult(
        ok=result.ok,
        issues=[i for i in result.issues if i.severity != "warning"],
        graph_names=result.graph_names,
        graphfn_names=result.graphfn_names,
    )
    if error_result.issues:
        _LOG.error(format_validation_errors(error_result, filename=filename))


def emit_validation_warnings(result: ValidationResult, *, filename: str | None = None) -> None:
    warning_result = ValidationResult(
        ok=result.ok,
        issues=[i for i in result.issues if i.severity == "warning"],
        graph_names=result.graph_names,
        graphfn_names=result.graphfn_names,
    )
    if not warning_result.issues:
        return
    warnings.warn(
        format_validation_errors(warning_result, filename=filename),
        category=UserWarning,
        stacklevel=3,
    )


def warnings_as_errors_enabled() -> bool:
    raw = os.getenv("AETHERGRAPH_GRAPHIFY_WARNINGS_AS_ERRORS", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}
