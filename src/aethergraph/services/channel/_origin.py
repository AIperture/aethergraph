from __future__ import annotations

from typing import Literal

from aethergraph.contracts.integration import OriginBinding


def _console_origin_binding(
    *,
    session_id: str,
    source: Literal["cli", "local"],
) -> OriginBinding:
    """Build the explicit console binding used by local execution entrypoints."""
    return OriginBinding(
        integration_id=f"{source}-console",
        route_id=f"{source}-console",
        session_id=session_id,
        channel_key="console:stdin",
        external_conversation_id=session_id,
        capability_profile_id="console-v1",
    )
