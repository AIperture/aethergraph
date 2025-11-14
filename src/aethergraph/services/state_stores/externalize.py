from __future__ import annotations

import hashlib
import io
import json
from typing import Any


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _maybe_numpy_to_bytes(obj: Any) -> bytes | None:
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            buf = io.BytesIO()
            # .npy
            import numpy as _np

            _np.save(buf, obj, allow_pickle=False)
            return buf.getvalue()
    except Exception:
        pass
    return None


def _maybe_torch_to_bytes(obj: Any) -> bytes | None:
    try:
        import torch

        if torch.is_tensor(obj):
            buf = io.BytesIO()
            torch.save(obj, buf)  # binary, portable within torch
            return buf.getvalue()
    except Exception:
        pass
    return None


def _maybe_json_bytes(obj: Any) -> bytes | None:
    # Only if JSON-serializable (pure)
    try:
        payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        return payload
    except Exception:
        return None


def _pickle_fallback(obj: Any) -> bytes | None:
    try:
        import pickle

        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        return None


async def externalize_to_artifact(
    obj: Any,
    *,
    run_id: str,
    graph_id: str,
    node_id: str,
    tool_name: str | None,
    tool_version: str | None,
    artifacts,  # AsyncArtifactStore
) -> dict[str, Any]:
    """
    Try to persist obj into artifact store; return a standard __aether_ref__ dict.
    Priority: bytes | numpy | torch | json | pickle
    """
    binary: bytes | None = None
    mime = "application/octet-stream"
    planned_ext = ".bin"

    # Already bytes?
    if isinstance(obj, bytes | bytearray):
        binary = bytes(obj)

    if binary is None:
        binary = _maybe_numpy_to_bytes(obj)
        if binary is not None:
            mime = "application/x-npy"
            planned_ext = ".npy"

    if binary is None:
        binary = _maybe_torch_to_bytes(obj)
        if binary is not None:
            mime = "application/x-pytorch"
            planned_ext = ".pt"

    if binary is None:
        json_bytes = _maybe_json_bytes(obj)
        if json_bytes is not None:
            # Use .json for nicer preview; still count as binary save
            binary = json_bytes
            mime = "application/json"
            planned_ext = ".json"

    if binary is None:
        binary = _pickle_fallback(obj)
        if binary is not None:
            mime = "application/x-pickle"
            planned_ext = ".pkl"

    if binary is None:
        # Give up: write a tiny JSON marker
        a = await artifacts.save_json(
            {"note": "unexternalizable-object", "repr": repr(obj)[:200]}, suggested_uri=None
        )
        return {
            "__aether_ref__": a.uri,
            "mime": "application/json",
            "sha256": a.sha256,
            "kind": a.kind,
        }

    sha = _sha256_bytes(binary)
    # Use staged writer for atomicity FIXME: this write causes async loop errors, disable externalize for now
    async with await artifacts.open_writer(
        kind="blob",
        run_id=run_id,
        graph_id=graph_id,
        node_id=node_id,
        tool_name=tool_name or "externalize",
        tool_version=tool_version or "0.1.0",
        planned_ext=planned_ext,
        pin=True,
    ) as w:
        await w.write(binary)  # FS wrapper should support .write
        a = await w.commit(mime=mime, sha256=sha)

    return {"__aether_ref__": a.uri, "mime": mime, "sha256": sha, "kind": a.kind}
