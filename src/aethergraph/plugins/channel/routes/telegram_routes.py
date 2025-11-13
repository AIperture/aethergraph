import asyncio
from fastapi import APIRouter, Request, HTTPException
from starlette.responses import Response
from ..utils.telegram_utils import _verify_secret, _process_update


router = APIRouter()


@router.post("/telegram/webhook")
async def telegram_webhook(request: Request):
    c = request.app.state.container
    BOT_TOKEN = request.app.state.settings.telegram.bot_token.get_secret_value() or ""
    if not BOT_TOKEN:
        raise HTTPException(503, "telegram bot token not configured")
    try:
        _verify_secret(request)
        payload = await request.json()
    except HTTPException:
        raise
    except Exception:
        return Response(status_code=400)

    asyncio.create_task(_process_update(c, payload, BOT_TOKEN))
    return Response(status_code=200)

