from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .specs import ContinuationResolver, HarnessAttachment, HarnessScenario, WaitResolutionRecord


@dataclass
class WaitResponse:
    kind: str | None = None
    node_id: str | None = None
    prompt_contains: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def matches(self, continuation: Any) -> bool:
        if self.kind and getattr(continuation, "kind", None) != self.kind:
            return False
        if self.node_id and getattr(continuation, "node_id", None) != self.node_id:
            return False
        prompt = getattr(continuation, "prompt", None)
        prompt_text = prompt if isinstance(prompt, str) else str(prompt or "")
        if self.prompt_contains and self.prompt_contains not in prompt_text:
            return False
        return True


@dataclass
class FailOnWaitResponder(ContinuationResolver):
    reason: str = "Scenario does not allow interactive waits"

    async def resolve(
        self,
        continuation: Any,
        *,
        scenario: HarnessScenario,
        wait_index: int,
        history: list[WaitResolutionRecord],
    ) -> dict[str, Any]:
        raise RuntimeError(f"{self.reason}: kind={getattr(continuation, 'kind', None)}")


@dataclass
class ScriptedResponder(ContinuationResolver):
    responses: list[WaitResponse] = field(default_factory=list)
    defaults_by_kind: dict[str, dict[str, Any]] = field(default_factory=dict)

    async def resolve(
        self,
        continuation: Any,
        *,
        scenario: HarnessScenario,
        wait_index: int,
        history: list[WaitResolutionRecord],
    ) -> dict[str, Any]:
        for response in self.responses:
            if response.matches(continuation):
                return dict(response.payload)
        kind = getattr(continuation, "kind", None) or ""
        if kind in self.defaults_by_kind:
            return dict(self.defaults_by_kind[kind])
        raise RuntimeError(
            f"No scripted response for wait kind={kind!r} node={continuation.node_id!r}"
        )


@dataclass
class AttachmentResponder(ScriptedResponder):
    files: list[HarnessAttachment] = field(default_factory=list)
    text: str = ""

    async def resolve(
        self,
        continuation: Any,
        *,
        scenario: HarnessScenario,
        wait_index: int,
        history: list[WaitResolutionRecord],
    ) -> dict[str, Any]:
        if getattr(continuation, "kind", None) == "user_files":
            payload_files = [
                attachment.to_resume_file() for attachment in self.files or scenario.attachments
            ]
            return {"text": self.text, "files": payload_files}
        return await super().resolve(
            continuation,
            scenario=scenario,
            wait_index=wait_index,
            history=history,
        )


def attachment_from_path(path: str, *, mimetype: str | None = None) -> HarnessAttachment:
    return HarnessAttachment(
        path=path,
        name=Path(path).name,
        mimetype=mimetype,
    )
