from __future__ import annotations
from fastapi import FastAPI, Response, Request, HTTPException
from pydantic import BaseModel
import uvicorn, json, urllib.parse
import httpx
import os
import logging

# TODO: move it to tests/examples later
DEMO_HTTP_TOKEN = os.getenv("DEMO_HTTP_TOKEN")

app = FastAPI()

TOOLS = [{
    "name": "search",
    "description": "Search Wikipedia and return top hits.",
    "input_schema": {
        "type": "object",
        "properties": { "q": {"type":"string"}, "k": {"type":"integer"} },
        "required": ["q"]
    }
}]

class RPCReq(BaseModel):
    jsonrpc: str
    id: int | str | None = None
    method: str
    params: dict | None = None

def ok(i, result): return {"jsonrpc":"2.0","id":i,"result":result}
def err(i, msg, code=-32000, data=None):
    e={"jsonrpc":"2.0","id":i,"error":{"code":code,"message":msg}}
    if data is not None: e["error"]["data"]=data
    return e

async def do_search(q: str, k: int = 5):
    url = "https://en.wikipedia.org/w/api.php"
    params = {"action":"query","list":"search","format":"json","srsearch": q, "srlimit": max(1, min(int(k or 5), 10))}
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get(url, params=params); r.raise_for_status()
        data = r.json()
    hits = []
    for it in (data.get("query",{}).get("search") or []):
        title = it.get("title","")
        page = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ","_"))
        hits.append({"title":title, "url":page, "snippet": it.get("snippet","")})
    return {"hits": hits}


@app.post("/rpc")
async def rpc(req: RPCReq, request: Request):
    if DEMO_HTTP_TOKEN:
        auth = request.headers.get("authorization","")
        if auth != f"Bearer {DEMO_HTTP_TOKEN}":
            raise HTTPException(status_code=401, detail="Unauthorized")
    else:
        logger = logging.getLogger("aethergraph.plugins.mcp.http_server")
        logger.warning("No auth token DEMO_HTTP_TOKEN set, skipping auth check. Set up DEMO_HTTP_TOKEN in env for test.")
    try:
        p = req.params or {}
        if req.method == "tools/list":
            return ok(req.id, TOOLS)
        if req.method == "tools/call":
            name = (p.get("name") or "").strip()
            args = p.get("arguments") or {}
            if name in ("search","query"):
                res = await do_search(args.get("q",""), int(args.get("k",5)))
                return ok(req.id, res)
            return err(req.id, f"Unknown tool: {name}")
        if req.method == "resources/list":
            return ok(req.id, [])
        if req.method == "resources/read":
            return ok(req.id, {"uri": p.get("uri"), "data": None})
        return err(req.id, f"Unknown method: {req.method}")
    except Exception as e:
        return err(req.id, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8769)
