from typing import Any
from uuid import uuid4


async def create_and_notify_continuation(
    *,
    context,
    kind: str,
    payload: dict[str, Any],
    timeout_s: int,
    channel: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """Create one public interaction before publishing its question.

    Dual-stage tools keep their normal scheduler resume even for inline answers.
    Delivery does not mutate the issued continuation or add transport lookup keys.

    Examples:
        Ask for text:
        ```python
        token, inline = await create_and_notify_continuation(
            context=context, kind="user_input", payload={"prompt": "Material?"}, timeout_s=60,
        )
        ```
        Select an explicit delivery channel:
        ```python
        token, inline = await create_and_notify_continuation(
            context=context, kind="user_input", payload={"prompt": "Name?"},
            timeout_s=60, channel="endpoint:conversation/one",
        )
        ```

    Args:
        context: Execution context owning continuation creation and delivery.
        kind: Runtime interaction kind.
        payload: Question and continuation setup data.
        timeout_s: Interaction lifetime in seconds.
        channel: Optional explicit channel address.

    Returns:
        tuple[str, dict[str, Any] | None]: One-time token and optional inline response.

    Notes:
        Public interaction identity is distinct from the private resume token.
    """
    bus = context.services.channels

    ch_key = context.channel(channel)._resolve_key()

    cont = await context.create_continuation(
        channel=ch_key,
        kind=kind,
        payload={**payload, "_interaction_id": f"interaction-{uuid4().hex}"},
        deadline_s=timeout_s,
    )

    res = await bus.notify(cont)
    inline = (res or {}).get("payload")
    return str(cont.token), inline
