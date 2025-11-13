from __future__ import annotations
from typing import Dict, Any, Optional, Protocol, Set
import asyncio
import warnings

from aethergraph.contracts.services.channel import OutEvent, Button, ChannelAdapter
from aethergraph.services.continuations.continuation import Correlator

class ChannelBus:
    """
    Transport layer:
      - publish(event) : send any OutEvent with smart fallbacks
      - notify(cont)   : raise a prompt from a Continuation; inline-resume if adapter can read input
      - peek_correlator(channel_key): ask adapter for a thread hint (optional)
    Optionally aware of:
      - resume_router  : used for inline resume (console/local-web)
      - store          : used to bind correlator↔token and to mint short resume_key
    """

    def __init__(
        self,
        adapters: Dict[str, ChannelAdapter],
        *,
        default_channel: str = "console:stdin",
        channel_aliases: Optional[Dict[str, str]] = None,
        logger=None,
        resume_router=None,
        store=None,
    ):
        self.adapters = dict(adapters)
        self.default_channel = default_channel
        self.logger = logger
        self.resume_router = resume_router
        self.store = store
        self.channel_aliases: Dict[str, str] = dict(channel_aliases or {})

    # ---- admin ----
    def register_adapter(self, prefix: str, adapter: ChannelAdapter) -> None:
        self.adapters[prefix] = adapter

    def set_default_channel_key(self, channel_key: str) -> None:
        self.default_channel = channel_key

    def get_default_channel_key(self) -> str:
        return self.default_channel

    def register_alias(self, alias: str, target: str) -> None:
        """Register or overwrite a human-friendly alias -> canonical key."""
        if self._prefix(target) not in self.adapters:
            raise RuntimeError(f"Cannot alias to unknown channel prefix: {self._prefix(target)}")
        
        self.channel_aliases[alias] = target

    def resolve_channel_key(self, key: str) -> str:
        """
        Resolve a channel key via the alias map.
        If `key` matches an alias exactly, return the mapped canonical key;
        otherwise return `key` as-is.
        """
        return self.channel_aliases.get(key, key)

    # ---- internals ----
    def _prefix(self, channel_key: str) -> str:
        return channel_key.split(":", 1)[0]

    def _pick(self, channel_key: str) -> ChannelAdapter:
        # IMPORTANT: resolve aliases *before* looking up adapter
        resolved_key = self.resolve_channel_key(channel_key)
        prefix = self._prefix(resolved_key)
        if prefix not in self.adapters:
            raise RuntimeError(f"No adapter for prefix={prefix}; known: {list(self.adapters.keys())}; Check if you have enabled the required channel service in .env and registered the adapter.")
        return self.adapters[prefix]

    def _warn(self, msg: str) -> None:
        if self.logger:
            self.logger.warning(msg)
        else:
            warnings.warn(msg)

    async def _bind_correlator_if_any(self, event: OutEvent, send_result: dict | None):
        if not self.store or not send_result:
            return
        corr = send_result.get("correlator")
        token = (event.meta or {}).get("token")
        if isinstance(corr, Correlator) and token:
            try:
                await self.store.bind_correlator(token=token, corr=corr)
            except Exception as e:
                self._warn(f"Failed to bind correlator: {e}")

    def _smart_fallback(self, adapter: ChannelAdapter, event: OutEvent) -> OutEvent | None:
        # Determine required capability for the event type
        need = None
        if event.type in ("agent.message", "agent.message.update", "session.waiting", "session.need_input"):
            need = "text"
        elif event.type in ("agent.stream.start", "agent.stream.delta", "agent.stream.end"):
            need = "stream"
        elif event.type in ("session.need_approval", "link.buttons"):
            need = "buttons"
        elif event.type == "file.upload":
            need = "file"

        caps: Set[str] = getattr(adapter, "capabilities", set())

        # Supported as-is
        if (need is None) or (need in caps):
            return event

        # buttons → text (numbered list)
        if need == "buttons" and "text" in caps:
            opts = []
            if event.buttons:
                for b in event.buttons:
                    lbl = getattr(b, "label", None) or str(getattr(b, "value", "") or "").title() or "Option"
                    val = getattr(b, "value", None) or str(lbl).lower()
                    opts.append({"label": str(lbl), "value": str(val)})
            else:
                for o in (event.meta or {}).get("options", []):
                    s = str(o); opts.append({"label": s, "value": s.lower()})
            if not opts:
                opts = [{"label": "Approve", "value": "approve"},
                        {"label": "Reject",  "value": "reject"}]
            lines = [f"{i+1}. {o['label']}" for i, o in enumerate(opts)]
            hint  = "Reply with the number or the label."
            txt   = (event.text or "Choose an option:") + "\n" + "\n".join(lines) + f"\n{hint}"
            meta = dict(event.meta or {})
            meta["options"] = [o["label"] for o in opts]
            meta["options_map"] = {str(i+1): o["value"] for i, o in enumerate(opts)}
            meta["options_label_to_value"] = {o["label"].lower(): o["value"] for o in opts}
            return OutEvent(type="agent.message", channel=event.channel, text=txt, meta=meta)

        # stream → text
        if need == "stream" and "text" in caps:
            if event.type == "agent.stream.delta":
                return OutEvent(type="agent.message", channel=event.channel, text=event.text or "", meta=event.meta)
            return None  # drop start/end

        # file → text link if available
        if need == "file" and "text" in caps:
            if event.file and "url" in event.file:
                return OutEvent(
                    type="agent.message", channel=event.channel,
                    text=f"[file] {event.file.get('filename','file')}: {event.file['url']}",
                    meta=event.meta
                )
            self._warn("Binary file not representable on this adapter.")
            return None

        self._warn(f"Adapter lacks '{need}', dropping event type={event.type}.")
        return None

    # ---- core send path ----
    async def publish(self, event: OutEvent) -> dict | None:
        """
        Send any OutEvent; apply smart fallbacks; bind correlator if adapter returns one.
        No inline resume here (use notify for interactions).
        """
        adapter = self._pick(event.channel)
        evt = self._smart_fallback(adapter, event)
        if evt is None:
            return None
        res = await adapter.send(evt)
        await self._bind_correlator_if_any(evt, res)
        return res

    # ---- continuation-aware notify (used by ChannelSession.ask_*) ----
    async def notify(self, continuation) -> dict | None:
        """
        Present a prompt for a Continuation, returning either:
        - {"payload": {...}} for inline adapters (console/local-web), or
        - {"correlator": Correlator(...)} for push-only adapters (Slack/Telegram).
        Never calls resume_router here; ChannelSession owns the wait/inline short-circuit.
        """
        ch   = continuation.channel
        kind = continuation.kind
        prompt = continuation.prompt

        # Short token for constrained transports
        resume_key = None
        if self.store and hasattr(self.store, "alias_for"):
            try:
                resume_key = await self.store.alias_for(continuation.token)
            except Exception:
                resume_key = None
        if not resume_key:
            resume_key = str(continuation.token)[:24]

        meta = {
            "run_id":  continuation.run_id,
            "node_id": continuation.node_id,
            "token":   continuation.token,
            "resume_key": resume_key,
        }

        # Shape event
        if kind == "user_input":
            silent = False
            if hasattr(continuation, "payload"):
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


        elif kind == "approval":
            labels: list[str] = []
            if isinstance(prompt, dict):
                txt = prompt.get("title") or prompt.get("prompt") or "Approve?"
                labels = prompt.get("buttons") or prompt.get("options") or []
            elif isinstance(prompt, str):
                txt = prompt or "Approve?"
            else:
                txt = "Approve?"
            if not labels:
                labels = ["Approve", "Reject"]
            btns = [Button(label=str(l), value=str(l).lower()) for l in labels]
            meta["options"] = labels
            meta["_prompt"] = True
            event = OutEvent(type="session.need_approval", channel=ch, text=txt, buttons=btns, meta=meta)
            needed_cap = "buttons"

        elif kind in ("user_files", "user_input_or_files"):
            # Console has no uploads; treat as text input. Other adapters may enhance later.
            txt = prompt if isinstance(prompt, str) else (prompt or "Please reply.")
            meta["_prompt"] = True
            event = OutEvent(type="session.need_input", channel=ch, text=txt, meta=meta)
            needed_cap = "input"

        else:
            txt = str(prompt) if isinstance(prompt, str) else "Waiting…"
            return await self.publish(OutEvent(type="session.waiting", channel=ch, text=txt, meta=meta))


        # Inline vs push-only
        adapter = self._pick(ch)
        caps = getattr(adapter, "capabilities", set())

        force_push = False
        if isinstance(prompt, dict):
            force_push = bool(prompt.get("_force_push"))
        if (needed_cap in caps) and not force_push:
            # Inline path
            res = await adapter.send(event)
            await self._bind_correlator_if_any(event, res)
            return res
        
        # Push-only path
        return await self.publish(event)



    # ---- optional: ask adapter for correlator/“thread” without sending ----
    async def peek_correlator(self, channel_key: str) -> Optional[Correlator]:
        adapter = self._pick(channel_key)
        scheme = self._prefix(channel_key)
        thread_ts = None
        if hasattr(adapter, "peek_thread"):
            try:
                thread_ts = await adapter.peek_thread(channel_key)
            except Exception:
                thread_ts = None
        return Correlator(scheme=scheme, channel=channel_key, thread=thread_ts, message=None)
