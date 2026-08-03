from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel  # type: ignore

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
