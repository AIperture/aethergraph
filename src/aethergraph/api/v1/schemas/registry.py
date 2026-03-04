from __future__ import annotations

from typing import Any

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
