from __future__ import annotations
import re
from typing import Any, Dict, Optional, List
from aethergraph.services.memory.facade import MemoryFacade

# -------- Regexes (unchanged) --------
STEP_RE        = re.compile(r'^\s*\$step\[(?P<idx>-?\d+)\]\.refs\.(?P<key>\w+)\s*$')
FROM_RE        = re.compile(r'^\s*\$from:(\w+)\s*$')
VAR_RE         = re.compile(r'^\s*\$var:(\w+)\s*$')

REF_KIND_RE    = re.compile(r'^\s*\$resolve\s*:\s*ref\.kind\s*=\s*(\w+)\s*\|\s*last\s*$', re.I)
NAME_RE        = re.compile(r'^\s*\$resolve\s*:\s*name\s*=\s*(\w+)\s*\|\s*last\s*$', re.I)
TOPIC_NAME_RE  = re.compile(r'^\s*\$resolve\s*:\s*topic\s*=\s*([\w\.\-\/]+)\s*\|\s*name\s*=\s*(\w+)\s*$', re.I)
LEGACY_KIND_RE = re.compile(r'^\s*\$resolve\s*:\s*kind\s*=\s*(\w+)\s*\|\s*last\s*$', re.I)

class ResolverContext:
    def __init__(self, mem: MemoryFacade, seq_ctx: Optional[dict] = None, vars: Dict[str, Any] | None = None):
        self.mem = mem
        self.seq_ctx = seq_ctx or {}
        self.vars = vars or {}

def _get_step_outputs(seq_ctx: dict, j: int) -> Optional[Dict[str, Any]]:
    steps = (seq_ctx.get("steps") or [])
    if 0 <= j < len(steps):
        return steps[j].get("outputs") or {}
    return None

async def _latest_ref_by_kind(mem: MemoryFacade, kind: str) -> Optional[str]:
    arr = await mem.latest_refs_by_kind(kind, limit=1)
    if arr:
        return arr[0].get("uri")
    # Fallback scan
    events = await mem.recent(kinds=["tool_result", "checkpoint"], limit=400)
    for e in reversed(events):
        outs = e.outputs or []
        for v in outs:
            if v.get("vtype") == "ref" and isinstance(v.get("value"), dict):
                if v["value"].get("kind") == kind:
                    return v["value"].get("uri")
        # legacy
        if e.outputs_ref and f"{kind}_ref" in e.outputs_ref:
            return e.outputs_ref.get(f"{kind}_ref")
    return None

async def _latest_value_by_name(mem: MemoryFacade, name: str) -> Optional[Any]:
    ent = await mem.last_by_name(name)
    if ent:
        return ent.get("value")
    # Fallback scan
    events = await mem.recent(kinds=["tool_result", "checkpoint"], limit=400)
    for e in reversed(events):
        outs = e.outputs or []
        for v in outs:
            if v.get("name") == name:
                return v.get("value")
        if e.outputs_ref and name in e.outputs_ref:
            return e.outputs_ref.get(name)
    return None

async def _latest_value_by_topic_name(mem: MemoryFacade, topic: str, name: str) -> Optional[Any]:
    ent = await mem.last_outputs_by_topic(topic)
    if ent:
        last = ent.get("last_outputs") or {}
        if name in last:
            return last[name]
    # Fallback scan
    events = await mem.recent(kinds=["tool_result"], limit=400)
    for e in reversed(events):
        if (e.tool or "") != topic:
            continue
        outs = e.outputs or []
        for v in outs:
            if v.get("name") == name:
                return v.get("value")
        if e.outputs_ref and name in e.outputs_ref:
            return e.outputs_ref.get(name)
    return None

async def resolve_params(raw: Dict[str, Any], ctx: ResolverContext) -> Dict[str, Any]:
    out = dict(raw)

    # 1) $step[i].refs.key
    for k, v in list(out.items()):
        if not isinstance(v, str): continue
        m = STEP_RE.match(v)
        if not m: continue
        idx = int(m.group("idx"))
        key = m.group("key")
        steps = (ctx.seq_ctx.get("steps") or [])
        j = idx if idx >= 0 else len(steps) + idx
        refs = _get_step_outputs(ctx.seq_ctx, j)
        if refs and key in refs:
            out[k] = refs[key]
        else:
            if key.endswith("_ref"):
                kind = key[:-4]
                out[k] = await _latest_ref_by_kind(ctx.mem, kind)
            else:
                out[k] = None

    # 2) $from:TAG (example strategy slot)
    for k, v in list(out.items()):
        if not isinstance(v, str): continue
        m = FROM_RE.match(v)
        if m:
            tag = m.group(1)
            resolved = None
            if tag == "last_opt_top1":
                resolved = await _latest_value_by_topic_name(ctx.mem, "optimize.flow", "top1_ref")
            out[k] = resolved

    # 3) New selectors + legacy
    for k, v in list(out.items()):
        if not isinstance(v, str): continue
        if (m := REF_KIND_RE.match(v)):
            out[k] = await _latest_ref_by_kind(ctx.mem, m.group(1).lower()); continue
        if (m := NAME_RE.match(v)):
            out[k] = await _latest_value_by_name(ctx.mem, m.group(1)); continue
        if (m := TOPIC_NAME_RE.match(v)):
            out[k] = await _latest_value_by_topic_name(ctx.mem, m.group(1), m.group(2)); continue
        if (m := LEGACY_KIND_RE.match(v)):
            out[k] = await _latest_ref_by_kind(ctx.mem, m.group(1).lower()); continue

    # 4) $var:NAME
    for k, v in list(out.items()):
        if isinstance(v, str) and (m := VAR_RE.match(v)):
            out[k] = ctx.vars.get(m.group(1))

    return out
