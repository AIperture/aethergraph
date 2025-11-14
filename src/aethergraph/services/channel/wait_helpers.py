from typing import Any

from aethergraph.services.continuations.continuation import Correlator


async def create_and_notify_continuation(
    *,
    context,
    kind: str,
    payload: dict[str, Any],
    timeout_s: int,
    channel: str | None = None,
) -> tuple[str, dict[str, Any] | None]:
    """
    Returns (token, inline_payload_or_none)
    Also binds correlators into the continuation store best-effort.
    """
    bus = context.services.channels  # ChannelBus
    store = context.services.continuation_store  # ContinuationStore

    ch_key = channel or bus.get_default_channel_key() or "console:stdin"

    cont = await context.create_continuation(
        channel=ch_key, kind=kind, payload=payload, deadline_s=timeout_s
    )

    res = await bus.notify(cont)
    inline = (res or {}).get("payload")
    if inline is not None:
        # Don't short circut for DualStageTool, we will still roundtrip through resume
        # so the toll path is uniform across adapters
        pass

    corr = (res or {}).get("correlator")
    if corr:
        await store.bind_correlator(token=cont.token, corr=corr)
        # also bind a message-less thread root for loopup by thread only
        await store.bind_correlator(
            token=cont.token,
            corr=Correlator(
                scheme=corr.scheme, channel=corr.channel, thread=corr.thread, message=""
            ),
        )
    else:
        # best-effort: bind a correlator with just channel+thread if available
        # best-effort
        peek = await bus.peek_correlator(ch_key)
        if peek:
            await store.bind_correlator(
                token=cont.token, corr=Correlator(peek.scheme, peek.channel, peek.thread, "")
            )
        else:
            await store.bind_correlator(
                token=cont.token, corr=Correlator(bus._prefix(ch_key), ch_key, "", "")
            )

    return str(cont.token), inline
