# ws_mcp_server.py  (robust for websockets v15, with optional token auth)
from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse

import httpx
from websockets import exceptions as ws_exceptions, serve
from websockets.http import Headers

# -------- Config --------
DEMO_WS_TOKEN = os.getenv("DEMO_WS_TOKEN", "").strip()
REQUIRE_HEADER_BEARER = True  # require Authorization header when token set
ALLOW_FIRST_MESSAGE_AUTH = True  # also allow in-band JSON-RPC auth frame

TOOLS = [
    {
        "name": "search",
        "description": "Search Wikipedia and return top hits.",
        "input_schema": {
            "type": "object",
            "properties": {"q": {"type": "string"}, "k": {"type": "integer"}},
            "required": ["q"],
        },
    }
]


def ok(i, result):
    return {"jsonrpc": "2.0", "id": i, "result": result}


def err(i, msg, code=-32000, data=None):
    e = {"jsonrpc": "2.0", "id": i, "error": {"code": code, "message": msg}}
    if data is not None:
        e["error"]["data"] = data
    return e


async def do_search(q: str, k: int = 5):
    params = {
        "action": "query",
        "list": "search",
        "format": "json",
        "srsearch": q,
        "srlimit": max(1, min(int(k or 5), 10)),
    }
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        data = r.json()
    hits = []
    for item in data.get("query", {}).get("search") or []:
        title = item.get("title", "")
        page = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
        hits.append({"title": title, "url": page, "snippet": item.get("snippet", "")})
    return {"hits": hits}


# ---------- Handshake-time token check (recommended) ----------
async def process_request(path: str, request_headers: Headers):
    """If DEMO_WS_TOKEN is set, enforce Authorization: Bearer <token> at handshake."""
    if not DEMO_WS_TOKEN or not REQUIRE_HEADER_BEARER:
        return  # accept; continue with handshake

    auth = request_headers.get("Authorization", "")
    if auth == f"Bearer {DEMO_WS_TOKEN}":
        return  # ok

    # Reject handshake with 401
    body = b"Unauthorized"
    headers = [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", str(len(body))),
        ("WWW-Authenticate", 'Bearer realm="mcp-ws", error="invalid_token"'),
    ]
    return (401, headers, body)


# ---------- Handler ----------
async def handle(ws):
    # Optional: in-band first-message auth if header was not used
    if DEMO_WS_TOKEN and ALLOW_FIRST_MESSAGE_AUTH and (not REQUIRE_HEADER_BEARER):
        try:
            first_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            first = json.loads(first_raw)
            if first.get("method") != "auth/bearer":
                await ws.send(
                    json.dumps(err(first.get("id"), "Unauthorized: expected auth/bearer"))
                )
                await ws.close()
                return
            tok = (first.get("params") or {}).get("token", "")
            if tok != DEMO_WS_TOKEN:
                await ws.send(json.dumps(err(first.get("id"), "Unauthorized: bad token")))
                await ws.close()
                return
            # auth ok; optionally reply success
            await ws.send(json.dumps(ok(first.get("id"), {"ok": True})))
        except Exception:
            # couldn't read/parse first frame or wrong shape
            try:
                await ws.send(json.dumps(err(None, "Unauthorized")))
            finally:
                await ws.close()
            return

    try:
        async for raw in ws:
            try:
                req = json.loads(raw)
                mid = req.get("id")
                method = req.get("method")
                params = req.get("params") or {}

                if method == "tools/list":
                    await ws.send(json.dumps(ok(mid, TOOLS)))
                    continue

                if method == "tools/call":
                    name = (params.get("name") or "").strip()
                    args = params.get("arguments") or {}
                    if name in ("search", "query"):
                        res = await do_search(args.get("q", ""), int(args.get("k", 5)))
                        await ws.send(json.dumps(ok(mid, res)))
                        continue
                    await ws.send(json.dumps(err(mid, f"Unknown tool: {name}")))
                    continue

                if method == "resources/list":
                    await ws.send(json.dumps(ok(mid, [])))
                    continue

                if method == "resources/read":
                    await ws.send(json.dumps(ok(mid, {"uri": params.get("uri"), "data": None})))
                    continue

                await ws.send(json.dumps(err(mid, f"Unknown method: {method}")))
            except Exception as e:
                # Return JSON-RPC error but keep the session alive
                try:
                    rid = req.get("id") if isinstance(req, dict) else None
                except Exception:
                    rid = None
                await ws.send(json.dumps(err(rid, str(e))))
    except (ws_exceptions.ConnectionClosedOK, ws_exceptions.ConnectionClosedError):
        return


async def main(host="0.0.0.0", port=8765):
    # If REQUIRE header-based auth and DISABLE in-band auth:
    #   set REQUIRE_HEADER_BEARER=True and ALLOW_FIRST_MESSAGE_AUTH=False
    async with serve(
        handle,
        host,
        port,
        ping_interval=20,
        ping_timeout=10,
        close_timeout=2,
        max_queue=32,
        process_request=process_request,  # <— handshake auth hook
    ):
        logger = logging.getLogger("aethergraph.plugins.mcp.ws_server")
        logger.info(f"MCP WS server listening on ws://{host}:{port}")
        if DEMO_WS_TOKEN:
            mode = []
            if REQUIRE_HEADER_BEARER:
                mode.append("header")
            if ALLOW_FIRST_MESSAGE_AUTH and not REQUIRE_HEADER_BEARER:
                mode.append("first-message")
            logger.info(f"Auth enabled: token set; modes: {', '.join(mode) or 'none'}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
