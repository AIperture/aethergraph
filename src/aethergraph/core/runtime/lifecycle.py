# lifecycle.py
import asyncio
from typing import Dict, Any
from aethergraph.core.runtime.runtime_services import current_services

async def start_all_services() -> None:
    svc = current_services()
    tasks = []
    for name, inst in getattr(svc, "ext_services", {}).items():
        start = getattr(inst, "start", None)
        if asyncio.iscoroutinefunction(start):
            tasks.append(start())
    if tasks:
        await asyncio.gather(*tasks)

async def close_all_services() -> None:
    svc = current_services()
    tasks = []
    for name, inst in getattr(svc, "ext_services", {}).items():
        close = getattr(inst, "close", None)
        if asyncio.iscoroutinefunction(close):
            tasks.append(close())
    if tasks:
        await asyncio.gather(*tasks)
