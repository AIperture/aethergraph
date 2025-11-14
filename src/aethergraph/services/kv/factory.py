import os

from .ephemeral import EphemeralKV
from .layered import LayeredKV
from .sqlite_kv import SQLiteKV


def make_kv():
    kind = (os.getenv("KV_BACKEND", "layered")).lower()
    prefix = os.getenv("KV_PREFIX", "")  # e.g., tenant/project scoping

    if kind == "ephemeral":
        return EphemeralKV(prefix=prefix)

    if kind == "sqlite":
        path = os.getenv("KV_SQLITE_PATH", "./artifacts/kv.sqlite")
        return SQLiteKV(path, prefix=prefix)

    if kind == "layered":
        cache = EphemeralKV(prefix=prefix)
        durable = SQLiteKV(os.getenv("KV_SQLITE_PATH", "./artifacts/kv.sqlite"), prefix=prefix)
        return LayeredKV(cache, durable)

    # (future) cloud:
    # if kind == "redis": return RedisKV(...)

    raise ValueError(f"Unknown KV_BACKEND={kind}")
