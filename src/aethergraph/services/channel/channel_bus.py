from __future__ import annotations

import warnings

from aethergraph.contracts.services.channel import (
    Button,
    ChannelAdapter,
    ChannelRoutingError,
    OutEvent,
)
from aethergraph.services.channel.choices import build_choice_options, prompt_choices_from_prompt
from aethergraph.services.continuations.continuation import Correlator


class ChannelBus:
    """
    Transport layer:
      - publish(event) : deliver an OutEvent unchanged to its exact adapter
      - notify(cont)   : raise a prompt from a Continuation; inline-resume if adapter can read input
      - peek_correlator(channel_key): ask adapter for a thread hint (optional)
    Optionally aware of:
      - resume_router  : used for inline resume (console/local-web)
      - store          : used to bind transport correlators to internal continuations
    """

    def __init__(
        self,
        adapters: dict[str, ChannelAdapter],
        *,
        logger=None,
        resume_router=None,
        store=None,
    ):
        """Create an exact-delivery Channel bus.

        Examples:
            Register adapters at construction:
            ```python
            bus = ChannelBus({"ui": ui_adapter, "console": console_adapter})
            ```

            Attach continuation services:
            ```python
            bus = ChannelBus(adapters, resume_router=router, store=store)
            ```

        Args:
            adapters: Adapter mapping keyed by exact channel prefix.
            logger: Optional Channel logger.
            resume_router: Optional continuation resume router.
            store: Optional continuation store.

        Returns:
            None.

        Notes:
            The bus owns neither a default address nor an alias registry.
        """
        self.adapters = dict(adapters)
        self.logger = logger
        self.resume_router = resume_router
        self.store = store

    # ---- admin ----
    def register_adapter(self, prefix: str, adapter: ChannelAdapter) -> None:
        self.adapters[prefix] = adapter

    # ---- internals ----
    def _prefix(self, channel_key: str) -> str:
        return channel_key.split(":", 1)[0]

    def _pick(self, channel_key: str) -> ChannelAdapter:
        prefix = self._prefix(channel_key)
        if prefix not in self.adapters:
            known_prefixes = tuple(sorted(self.adapters))
            raise ChannelRoutingError(
                code="channel.adapter_not_found",
                channel_key=channel_key,
                known_prefixes=known_prefixes,
                message=(
                    f"No channel adapter is registered for prefix {prefix!r}; "
                    f"known prefixes: {list(known_prefixes)}."
                ),
            )
        return self.adapters[prefix]

    def _warn(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    async def _bind_correlator_if_any(
        self,
        send_result: dict | None,
        *,
        continuation_token: str | None,
    ):
        if not self.store or not send_result or not continuation_token:
            return
        corr = send_result.get("correlator")
        if isinstance(corr, Correlator):
            try:
                await self.store.bind_correlator(token=continuation_token, corr=corr)
            except Exception as e:
                self._warn(f"Failed to bind correlator: {e}")

    # ---- core send path ----
    async def publish(self, event: OutEvent) -> dict | None:
        """Deliver one event unchanged to its exact channel adapter.

        Examples:
            Deliver a text event:
            ```python
            event = OutEvent(
                type="agent.message",
                channel="endpoint:sessions/s1",
                text="Done",
            )
            await bus.publish(event)
            ```

            Inspect an adapter delivery receipt:
            ```python
            receipt = await bus.publish(event)
            delivery_id = (receipt or {}).get("delivery_id")
            ```

        Args:
            event: Fully formed event with a concrete channel address.

        Returns:
            Adapter-defined delivery metadata, or `None` when the adapter does
            not return metadata.

        Notes:
            Projection and capability validation belong to the configured
            adapter boundary. This method never converts or drops an event and
            never performs inline continuation resume; use `notify` for an
            interaction prompt.
        """
        adapter = self._pick(event.channel)
        res = await adapter.send(event)
        return res

    # ---- continuation-aware notify (used by ChannelSession.ask_*) ----
    async def notify(self, continuation) -> dict | None:
        """
        Present a prompt for a Continuation, returning either:
        - {"payload": {...}} for inline adapters (console/local-web), or
        - {"correlator": Correlator(...)} for push-only adapters (Slack/Telegram).
        Never calls resume_router here; ChannelSession owns the wait/inline short-circuit.
        """
        ch = continuation.channel
        kind = continuation.kind
        prompt = continuation.prompt

        continuation_payload = getattr(continuation, "payload", None)
        interaction_id = (
            continuation_payload.get("_interaction_id")
            if isinstance(continuation_payload, dict)
            else None
        )
        if not isinstance(interaction_id, str) or not interaction_id:
            raise ValueError("Continuation is missing its public interaction identity.")

        meta = {
            "interaction_id": interaction_id,
            "interaction_kind": kind,
        }
        if isinstance(continuation_payload, dict) and kind in (
            "user_files",
            "user_input_or_files",
        ):
            meta["accept"] = list(continuation_payload.get("accept") or [])
            meta["multiple"] = bool(continuation_payload.get("multiple", True))

        # Enrich continuation meta with the same context fields we attach
        # on normal channel events (if present on the continuation object).
        session_id = getattr(continuation, "session_id", None)
        if session_id is not None:
            meta.setdefault("session_id", session_id)

        run_id = getattr(continuation, "run_id", None)
        if run_id is not None:
            meta.setdefault("run_id", run_id)

        node_id = getattr(continuation, "node_id", None)
        if node_id is not None:
            meta.setdefault("node_id", node_id)

        agent_id = getattr(continuation, "agent_id", None)
        if agent_id is not None:
            meta.setdefault("agent_id", agent_id)

        app_id = getattr(continuation, "app_id", None)
        if app_id is not None:
            meta.setdefault("app_id", app_id)

        graph_id = getattr(continuation, "graph_id", None)
        if graph_id is not None:
            meta.setdefault("graph_id", graph_id)

        # Shape event
        if kind == "user_input":
            silent = False
            if hasattr(continuation, "payload") and isinstance(continuation.payload, dict):
                silent = continuation.payload.get("_silent", False)

            txt = prompt if isinstance(prompt, str) else None

            if silent and not txt:
                # Silent wait: don't emit a session.need_input event at all.
                # Just return {} so ChannelSession will rely on the normal wait/resolve path.
                meta["_prompt"] = False
                return {}

            # Normal ask_text path
            txt = txt or "Please reply."
            meta["_prompt"] = True
            event = OutEvent(type="session.need_input", channel=ch, text=txt, meta=meta)
            needed_cap = "input"

        elif kind in ("approval", "choice"):
            choices = []
            if isinstance(prompt, dict):
                txt = prompt.get("title") or prompt.get("prompt") or "Approve?"
                choices = prompt_choices_from_prompt(prompt)
            elif isinstance(prompt, str):
                txt = prompt or "Approve?"
            else:
                txt = "Approve?"
            if not choices:
                choices = build_choice_options(["Approve", "Reject"])
            btns = [Button(label=choice.label, value=choice.id) for choice in choices]
            meta["options"] = [choice.label for choice in choices]
            meta["choices"] = [
                {"id": choice.id, "label": choice.label, "aliases": list(choice.aliases)}
                for choice in choices
            ]
            meta["_prompt"] = True
            event = OutEvent(
                type="session.need_approval", channel=ch, text=txt, buttons=btns, meta=meta
            )
            needed_cap = "buttons"

        elif kind in ("user_files", "user_input_or_files"):
            # Console has no uploads; treat as text input. Other adapters may enhance later.
            txt = prompt if isinstance(prompt, str) else (prompt or "Please reply.")
            meta["_prompt"] = True
            event = OutEvent(type="session.need_input", channel=ch, text=txt, meta=meta)
            needed_cap = "input"

        else:
            txt = str(prompt) if isinstance(prompt, str) else "Waiting…"
            event = OutEvent(type="session.waiting", channel=ch, text=txt, meta=meta)
            adapter = self._pick(ch)
            res = await adapter.send(event)
            await self._bind_correlator_if_any(
                res,
                continuation_token=continuation.token,
            )
            return res

        # Inline vs push-only
        adapter = self._pick(ch)
        caps = getattr(adapter, "capabilities", set())

        force_push = False
        if isinstance(prompt, dict):
            force_push = bool(prompt.get("_force_push"))
        if (needed_cap in caps) and not force_push:
            # Inline path
            res = await adapter.send(event)
            await self._bind_correlator_if_any(
                res,
                continuation_token=continuation.token,
            )
            return res

        # Push-only path
        res = await adapter.send(event)
        await self._bind_correlator_if_any(
            res,
            continuation_token=continuation.token,
        )
        return res

    # ---- optional: ask adapter for correlator/“thread” without sending ----
    async def peek_correlator(self, channel_key: str) -> Correlator | None:
        adapter = self._pick(channel_key)
        scheme = self._prefix(channel_key)
        thread_ts = None
        if hasattr(adapter, "peek_thread"):
            try:
                thread_ts = await adapter.peek_thread(channel_key)
            except Exception:
                thread_ts = None
        return Correlator(scheme=scheme, channel=channel_key, thread=thread_ts, message=None)
