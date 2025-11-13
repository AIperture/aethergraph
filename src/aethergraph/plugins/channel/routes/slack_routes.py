# slack_http_routes.py

from fastapi import APIRouter, Request
from starlette.responses import JSONResponse
from ..utils.slack_utils import _verify_sig, handle_slack_events_common, handle_slack_interactive_common
import json 

router = APIRouter()

@router.post("/slack/events")
async def slack_events(request: Request):
    settings = request.app.state.settings
    container = request.app.state.container

    body = await request.body()
    _verify_sig(request, body)  # HTTP-only

    payload = json.loads(body)

    # URL verification (Events API handshake)
    if payload.get("type") == "url_verification":
        # Just echo the challenge back
        return JSONResponse(payload)

    # Delegate real work to shared handler
    resp = await handle_slack_events_common(container, settings, payload)
    return JSONResponse(resp or {})


@router.post("/slack/interact")
async def slack_interact(request: Request):
    """Handle interactive components (buttons) from Slack via HTTP."""
    settings = request.app.state.settings  # not strictly needed here yet
    container = request.app.state.container

    body = await request.body()
    _verify_sig(request, body)  # HTTP-only

    form = await request.form()
    payload = json.loads(form["payload"])

    await handle_slack_interactive_common(container, payload)
    return JSONResponse({})  # ack
