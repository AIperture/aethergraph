from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from aethergraph.services.continuations.continuation import Correlator

router = APIRouter()


@router.get("/api/continuations/latest")
async def latest(request: Request, channel: str, kind: str | None = Query(None)):
    """
    Console: resolve newest open continuation bound to this channel.
    We use a channel-wide correlator (no thread/message).
    TODO:  add a 'message' query param later if want more precise matching.
    """
    c = request.app.state.container

    # First try channel-wide correlator (message="")
    corr = Correlator(scheme="console", channel=channel, thread="", message="")
    cont = await c.cont_store.find_by_correlator(corr=corr)

    # (Optional) If we pass ?message=... from a UI client, try that first:
    # message = request.query_params.get("message")
    # if message:
    #     corr_precise = Correlator(scheme="console", channel=channel, thread="", message=message)
    #     cont = c.cont_store.find_by_correlator(corr=corr_precise) or cont

    if kind and cont and cont.kind != kind:
        # If caller asks for a specific kind, and the found one doesn't match, return None
        return None

    return cont.to_dict() if cont else None


class ConsoleResume(BaseModel):
    run_id: str
    node_id: str
    token: str
    payload: dict


@router.post("/api/console/resume")
async def console_resume(request: Request, req: ConsoleResume):
    c = request.app.state.container
    payload = dict(req.payload or {})
    cont = (
        await c.cont_store.get_by_token(req.token)
        if hasattr(c.cont_store, "get_by_token")
        else None
    )

    if cont and getattr(cont, "kind", "") == "approval":
        # If client already parsed the choice, don't override it
        if "choice" in payload or "approved" in payload:
            pass  # trust the client (console watcher)
        else:
            raw = str(payload.get("text", "")).strip()
            # Try to use mappings echoed by the client (if any)
            options_map = payload.get("options_map") or {}
            label_map = payload.get("options_label_to_value") or {}

            # Otherwise reconstruct from the Continuation.prompt
            if not options_map and isinstance(cont.prompt, dict):
                labels = cont.prompt.get("buttons") or cont.prompt.get("options") or []
                options_map = {
                    str(i + 1): str(lbl).lower() for i, lbl in enumerate(labels, start=1)
                }
                label_map = {str(lbl).lower(): str(lbl).lower() for lbl in labels}

            if raw.isdigit() and raw in options_map:
                choice = options_map[raw]
            else:
                choice = label_map.get(raw.lower(), raw.lower())

            payload = {
                "approved": choice in {"approve", "approved", "yes", "y"},
                "choice": choice,
            }

    await c.resume_router.resume(req.run_id, req.node_id, req.token, payload)

    # TODO: (optional safety) if our resume router does NOT mark it closed internally:
    # c.cont_store.mark_closed(req.token) or delete it;
    # now it seems we haven't gone through resume_router for console resumes yet. So the continuation remains open.

    return {"ok": True}
