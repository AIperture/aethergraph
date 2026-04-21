from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field  # type: ignore

from aethergraph.core.runtime.run_types import SessionKind

from .runs import RunSummary

SessionTitleSource = Literal["manual", "auto"]


class Session(BaseModel):
    session_id: str
    kind: SessionKind
    title: str | None = None
    title_source: SessionTitleSource | None = None
    user_id: str | None = None
    org_id: str | None = None
    source: str = "webui"
    external_ref: str | None = None
    created_at: datetime
    updated_at: datetime
    artifact_count: int = 0
    last_artifact_at: datetime | None = None


class SessionCreateRequest(BaseModel):
    kind: SessionKind
    title: str | None = None
    external_ref: str | None = None


class SessionListResponse(BaseModel):
    items: list[Session]
    next_cursor: str | None = None


class SessionRunsResponse(BaseModel):
    items: list[RunSummary]
    next_cursor: str | None = None


class SessionChatFile(BaseModel):
    artifact_id: str | None = None
    id: str | None = None
    url: str | None = None
    name: str | None = None
    mimetype: str | None = None
    size: int | None = None
    uri: str | None = None
    renderer: Literal["image", "download", "vega", "plotly"] | None = None

    class Config:
        extra = "allow"


class SessionChatEvent(BaseModel):
    id: str
    session_id: str
    type: str
    text: str | None
    buttons: list[dict[str, Any]] = Field(default_factory=list)
    file: SessionChatFile | None = None
    files: list[SessionChatFile] | None = None
    attachments: list[SessionChatFile] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)
    ts: float
    agent_id: str | None = None
    upsert_key: str | None = None
    rich: dict[str, Any] | None = None


class SessionChatEventListResponse(BaseModel):
    events: list[SessionChatEvent]
    next_cursor: str | None = None


class SessionWorkStatusRunRef(BaseModel):
    run_id: str | None = None
    graph_id: str | None = None


class SessionWorkStatusArtifactRef(BaseModel):
    artifact_id: str | None = None
    name: str | None = None
    kind: str | None = None


class SessionWorkStatusItem(BaseModel):
    id: str
    label: str
    kind: str
    status: str
    detail: str = ""
    order: int = 0
    run_ref: SessionWorkStatusRunRef | None = None
    artifact_ref: SessionWorkStatusArtifactRef | None = None


class SessionWorkStatus(BaseModel):
    workflow_id: str
    title: str
    kind: str
    status: str
    summary: str = ""
    active_item_id: str | None = None
    updated_at: str
    items: list[SessionWorkStatusItem] = Field(default_factory=list)


class SessionWorkStatusResponse(BaseModel):
    work_status: SessionWorkStatus | None = None


class SessionDashboardPatchOp(BaseModel):
    op: Literal["replace", "add", "remove", "append"]
    path: str
    value: Any | None = None


class SessionDashboardState(BaseModel):
    dashboard_id: str
    dashboard_type: str
    workflow_id: str
    revision: int
    status: str
    updated_at: str
    data: dict[str, Any] = Field(default_factory=dict)


class SessionDashboardEnvelope(BaseModel):
    dashboard: SessionDashboardState | None = None
    patch: dict[str, Any] | None = None


class SessionDashboardStateResponse(BaseModel):
    dashboards: list[SessionDashboardState] = Field(default_factory=list)


class SessionUpdateRequest(BaseModel):
    title: str | None = None
    external_ref: str | None = None


class SessionInferTitleRequest(BaseModel):
    force: bool = False
    mode: Literal["initial", "refresh"] = "initial"


class SessionInferTitleResponse(BaseModel):
    session_id: str
    title: str | None = None
    updated: bool = False
    reason: (
        Literal[
            "generated",
            "skipped_has_title",
            "skipped_manual",
            "skipped_no_context",
            "skipped_disabled_llm",
        ]
        | None
    ) = None
