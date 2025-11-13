from __future__ import annotations
from typing import Any, Dict, List, Optional
from ..waitable import DualStageTool, waitable_tool 
from ...execution.wait_types import WaitSpec, WaitRequested
from aethergraph.services.channel.wait_helpers import create_and_notify_continuation


def normalize_approval_result(payload: dict) -> dict:
    """
    Normalize approval result from various adapters into a consistent format.
    
    It assumes the payload may contain:
    - "approved": bool (explicit approval flag)
    - "choice": str (the user's choice)
    It infers "approved" from "choice" if "approved" is not present.
    """
    choice = payload.get("choice")

    # infer from options (first = approved)
    options = payload.get("options") or payload.get("buttons")
    if not options:
        prompt = payload.get("prompt")
        if isinstance(prompt, dict):
            options = prompt.get("buttons")

    if not options or choice is None:
        approved = False
    else:
        choice_norm = str(choice).strip().lower()
        first_norm = str(options[0]).strip().lower()
        approved = (choice_norm == first_norm)

    return {"approved": approved, "choice": choice}


# ----- AskTextTool -----
class AskText(DualStageTool):
    outputs = ["text"]

    async def setup(
            self,
            prompt: Optional[str] = None,
            *,
            silent: bool = False,
            timeout_s: int = 3600,
            channel: Optional[str] = None,
            context,
    ) -> WaitSpec | Dict[str, Any]:
        token, inline = await create_and_notify_continuation(
            context=context,
            kind="user_input",
            payload={"prompt": prompt, "_force_push": True, "_silent": silent} if prompt else {"_force_push": True, "_silent": silent},
            timeout_s=timeout_s,
            channel=channel,
        )
        return WaitSpec(channel=channel, token=token, kind="user_input", deadline=timeout_s, notified=True, inline_payload=inline)
    
    async def on_resume(self, resume: Dict[str, Any], *, context: Any) -> Dict[str, Any]:
        text = resume.get("text", "")
        return {"text": text} 

ask_text_ds = waitable_tool(AskText) 

# ----- WaitText Tool  -----
class WaitText(DualStageTool):
    outputs = ["text"]

    async def setup(
            self,
            *,
            timeout_s: int = 3600,
            channel: Optional[str] = None,
            context,
    ) -> WaitSpec | Dict[str, Any]:
        token, inline = await create_and_notify_continuation(
            context=context,
            kind="user_input",
            payload={"prompt": None, "_force_push": True, "_silent": True},
            timeout_s=timeout_s,
            channel=channel,
        )
        return WaitSpec(channel=channel, token=token, kind="user_input", deadline=timeout_s, notified=True, inline_payload=inline)
    
    async def on_resume(self, resume: Dict[str, Any], *, context: Any) -> Dict[str, Any]:
        text = resume.get("text", "")
        return {"text": text} 
    
wait_text_ds = waitable_tool(WaitText) 

# ----- AskApprovalTool -----
class AskApproval(DualStageTool):
    outputs = ["approved", "choice"]

    async def setup(
            self,
            prompt: str, 
            options: List[str] | tuple[str, ...] = ("Approve", "Reject"),
            *,
            timeout_s: int = 3600,
            channel: Optional[str] = None,
            context: Any,
    ) -> WaitSpec | Dict[str, Any]:
        token, inline = await create_and_notify_continuation(
            context=context,
            kind="approval",
            payload={"prompt": {"title": prompt, "buttons": list(options)}, "_force_push": True},
            timeout_s=timeout_s,
            channel=channel,
        )
        return WaitSpec(channel=channel, token=token, kind="approval", deadline=timeout_s, notified=True, inline_payload=inline)

    
    async def on_resume(self, resume: Dict[str, Any], *, context: Any) -> Dict[str, Any]:
        return normalize_approval_result(resume)
    
ask_approval_ds = waitable_tool(AskApproval) 


# ----- AskFiles Tool -----
class AskFiles(DualStageTool):
    outputs = ["text", "files"]

    async def setup(
        self,
        *,
        prompt: str,
        accept: Optional[List[str]] = None,
        multiple: bool = True,
        timeout_s: int = 3600,
        channel: Optional[str] = None,
        context: Any
    ) -> WaitSpec | Dict[str, Any]:
        token, inline = await create_and_notify_continuation(
            context=context,
            kind="user_files",
            payload={"prompt": prompt, "accept": accept or [], "multiple": bool(multiple), "_force_push": True},
            timeout_s=timeout_s,
            channel=channel,
        )
        return WaitSpec(channel=channel, token=token, kind="user_files", deadline=timeout_s, notified=True, inline_payload=inline)

    async def on_resume(self, resume: Dict[str, Any], *, context: Any) -> Dict[str, Any]:
        files = resume.get("files", [])
        if not isinstance(files, list):
            files = []
        return {
            "text": str(resume.get("text", "")),
            "files": files,
        }

ask_files_ds = waitable_tool(AskFiles)