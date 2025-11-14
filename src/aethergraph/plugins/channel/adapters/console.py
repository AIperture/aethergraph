import asyncio
import sys

from aethergraph.contracts.services.channel import ChannelAdapter, OutEvent
from aethergraph.services.continuations.continuation import Correlator


class ConsoleChannelAdapter(ChannelAdapter):
    # console can now ask for text and approvals (buttons via numeric mapping)
    capabilities: set[str] = {"text", "input", "buttons"}

    def __init__(self):
        self._seq_by_chan: dict[str, int] = {}

    async def send(self, event: OutEvent) -> dict | None:
        # non-interactive path: just print
        if event.type not in ("session.need_input", "session.need_approval"):
            line = f"[console] {event.type} :: {event.text or ''}"
            if event.image:
                line += f" [image] {event.image.get('title', '')}: {event.image.get('url', '')}"
            if event.file:
                line += f" [file]  {event.file.get('filename', '')}: {event.file.get('url', '') or '(binary)'}"
            if event.buttons:
                labels = ", ".join(b.label for b in event.buttons)
                line += f" [buttons] {labels}"
            print(line)

            seq = self._seq_by_chan.get(event.channel, 0) + 1
            self._seq_by_chan[event.channel] = seq
            return {
                "correlator": Correlator(
                    scheme="console",
                    channel=event.channel,
                    thread=None,
                    message=str(seq),
                )
            }

        # Interactive: input
        if event.type == "session.need_input":
            prompt = (event.text or "Please reply: ").rstrip() + " "
            try:
                answer = await self._readline(prompt)
                return {"payload": {"text": answer}}
            except _NoInlineInput:
                # Signal the waiter to persist a real continuation instead of inlining
                print(
                    "\n[console] (no input captured; will persist a continuation and wait for resume)"
                )
                return None

        # Interactive: approval
        if event.type == "session.need_approval":
            labels = [b.label for b in (event.buttons or [])] or (event.meta or {}).get(
                "options", []
            )
            if not labels:
                labels = ["Approve", "Reject"]

            print((event.text or "Choose an option:").strip())
            for i, label in enumerate(labels, 1):
                print(f"  {i}. {label}")

            try:
                ans = await self._readline("Reply with number or label: ")
                by_num = {str(i): label for i, label in enumerate(labels, 1)}
                choice_label = by_num.get(ans, ans).strip()
                approved = choice_label.lower() in {"approve", "approved", "yes", "y", "ok"}
                return {"payload": {"approved": approved, "choice": choice_label}}
            except _NoInlineInput:
                print(
                    "\n[console] (no choice captured; will persist a continuation and wait for resume)"
                )
                return None

        # unreachable
        return None

    async def _readline(self, prompt: str | None = None) -> str:
        # Print prompt and flush so it’s visible before we block
        if prompt:
            print(prompt, end="", flush=True)

        loop = asyncio.get_running_loop()
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
        except KeyboardInterrupt:
            # User pressed Ctrl+C while we were blocked on input — treat as “no inline input”
            raise _NoInlineInput() from None

        if line is None:
            # Extremely defensive; run_in_executor should always give a str
            raise _NoInlineInput() from None

        line = line.rstrip("\n")
        if line == "":
            # Empty (e.g., Ctrl+C causing an immediate return on some terminals, or EOF)
            raise _NoInlineInput() from None

        return line.strip()


class _NoInlineInput(Exception):
    """Signal to the wait machinery that the adapter should not inline-resume."""

    pass
