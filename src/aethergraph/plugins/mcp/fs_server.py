# Minimal MCP filesystem server over stdio JSON-RPC (cross-platform)
import sys, os, json, time, traceback

TOOLS = [
    {"name":"readFile","description":"Read a text file","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
    {"name":"writeFile","description":"Write text to a file","input_schema":{"type":"object","properties":{"path":{"type":"string"},"text":{"type":"string"}},"required":["path","text"]}},
    {"name":"listDir","description":"List directory entries","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
    {"name":"stat","description":"Stat a file or directory","input_schema":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}},
]

def _ok(id, result): return {"jsonrpc":"2.0","id":id,"result":result}
def _err(id, code=-32000, msg="Server error", data=None):
    e = {"jsonrpc":"2.0","id":id,"error":{"code":code,"message":msg}}
    if data is not None: e["error"]["data"] = data
    return e

def list_tools():
    return TOOLS

def call(name, args):
    if name == "readFile":
        p = args["path"]
        with open(p, "r", encoding="utf-8") as f: txt = f.read()
        return {"text": txt}
    if name == "writeFile":
        p, t = args["path"], args["text"]
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", encoding="utf-8") as f: f.write(t)
        return {"ok": True, "bytes": len(t)}
    if name == "listDir":
        p = args["path"]
        entries = []
        for name in os.listdir(p):
            fp = os.path.join(p, name)
            entries.append({"name": name, "is_dir": os.path.isdir(fp), "size": os.path.getsize(fp) if os.path.isfile(fp) else None})
        return {"entries": entries}
    if name == "stat":
        p = args["path"]
        st = os.stat(p)
        return {"path": p, "is_dir": os.path.isdir(p), "size": st.st_size, "mtime": st.st_mtime}
    raise ValueError(f"Unknown tool: {name}")

def main():
    stdin = sys.stdin
    stdout = sys.stdout
    while True:
        line = stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            mid = req.get("id")
            method = req.get("method")
            params = req.get("params") or {}
            if method == "tools/list":
                resp = _ok(mid, list_tools())
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments") or {}
                resp = _ok(mid, call(name, args))
            elif method == "resources/list":
                resp = _ok(mid, [])  # not used in this minimal server
            elif method == "resources/read":
                resp = _ok(mid, {"uri": params.get("uri"), "data": None})
            else:
                resp = _err(mid, msg=f"Unknown method {method}")
        except Exception as e:
            resp = _err(req.get("id"), msg=str(e), data=traceback.format_exc())
        stdout.write(json.dumps(resp) + "\n")
        stdout.flush()

if __name__ == "__main__":
    main()
