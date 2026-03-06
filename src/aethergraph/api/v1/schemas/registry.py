from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field  # type: ignore

from .input_schema import InputFieldSpec


class SlashCommandDescriptor(BaseModel):
    name: str
    description: str = ""


class AgentDescriptor(BaseModel):
    id: str
    graph_id: str
    deletable: bool = False
    slash_commands: list[SlashCommandDescriptor] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class AppDescriptor(BaseModel):
    id: str
    graph_id: str
    deletable: bool = False
    slash_commands: list[SlashCommandDescriptor] = Field(default_factory=list)
    input_schema: list[InputFieldSpec] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class RegistryRegisterRequest(BaseModel):
    source: Literal["file", "artifact"] = "file"
    path: str | None = None
    artifact_id: str | None = None
    uri: str | None = None
    app_config: dict[str, Any] | None = None
    agent_config: dict[str, Any] | None = None
    persist: bool = True
    strict: bool = True


class RegistryRegisterResponse(BaseModel):
    success: bool
    source_kind: str
    source_ref: str
    filename: str | None = None
    sha256: str | None = None
    graph_name: str | None = None
    app_id: str | None = None
    agent_id: str | None = None
    version: str | None = None
    entry_id: str | None = None
    errors: list[str] = Field(default_factory=list)
