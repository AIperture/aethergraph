"""Exact text projection for named semantic outputs on messaging providers."""

from __future__ import annotations

from typing import Any

from aethergraph.contracts.services.channel import OutEvent


def project_messaging_text(event: OutEvent) -> str:
    """Project one supported named output to provider-visible text.

    Examples:
        Project a workflow status update:
        ```python
        text = project_messaging_text(event)
        ```

        Reject an output without a messaging projection:
        ```python
        project_messaging_text(unsupported_event)
        ```

    Args:
        event: Structured Channel output to project.

    Returns:
        str: Explicit provider-visible representation.

    Notes:
        Unknown output names fail closed so providers never silently drop or
        generically stringify authored structured data.
    """
    rich = event.rich or {}
    output_name = str(rich.get("output_name") or "")
    value = rich.get("value")
    if not isinstance(value, dict):
        raise ValueError(f"Structured output {output_name!r} requires an object value.")
    if output_name == "workflow.status":
        return _work_status_text(value)
    if output_name == "workflow.dashboard":
        return _dashboard_text(value)
    raise ValueError(f"No messaging projection exists for structured output {output_name!r}.")


def _work_status_text(value: dict[str, Any]) -> str:
    operation = value.get("operation")
    if operation == "clear":
        return "Workflow status cleared."
    status = value.get("work_status") if operation == "replace" else value
    if not isinstance(status, dict):
        raise ValueError("workflow.status requires replace, patch, or clear data.")
    title = str(status.get("title") or status.get("workflow_id") or "Workflow")
    state = str(status.get("status") or "updated")
    summary = str(status.get("summary") or "").strip()
    return f"{title}: {state}. {summary}".strip()


def _dashboard_text(value: dict[str, Any]) -> str:
    operation = value.get("operation")
    if operation == "replace":
        dashboard = value.get("dashboard")
        if not isinstance(dashboard, dict):
            raise ValueError("workflow.dashboard replace requires dashboard data.")
        dashboard_id = dashboard.get("dashboard_id")
        status = dashboard.get("status") or "updated"
    elif operation == "patch":
        patch = value.get("patch")
        if not isinstance(patch, dict):
            raise ValueError("workflow.dashboard patch requires patch data.")
        dashboard_id = patch.get("dashboard_id")
        status = patch.get("status") or "updated"
    elif operation == "clear":
        dashboard_id = value.get("dashboard_id")
        status = "cleared"
    else:
        raise ValueError("workflow.dashboard requires replace, patch, or clear data.")
    if not dashboard_id:
        raise ValueError("workflow.dashboard requires dashboard_id.")
    return f"Workflow dashboard {dashboard_id}: {status}."
