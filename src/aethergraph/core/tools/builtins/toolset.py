from typing import Any

from aethergraph.contracts.services.channel import Button, FileRef

from ..toolkit import tool
from .channel_tools import ask_approval_ds, ask_files_ds, ask_text_ds, wait_text_ds


@tool(name="ask_text", outputs=["text"])
async def ask_text(
    *,
    resume=None,
    context=None,
    prompt: str | None = None,
    silent: bool = False,
    timeout_s: int = 3600,
    channel: str | None = None,
):
    return await ask_text_ds(
        resume=resume,
        context=context,
        prompt=prompt,
        silent=silent,
        timeout_s=timeout_s,
        channel=channel,
    )


@tool(name="wait_text", outputs=["text"])
async def wait_text(
    *, resume=None, context=None, timeout_s: int = 3600, channel: str | None = None
):
    return await wait_text_ds(resume=resume, context=context, timeout_s=timeout_s, channel=channel)


@tool(name="ask_approval", outputs=["approved", "choice"])
async def ask_approval(
    *,
    resume=None,
    context=None,
    prompt: str,
    options: list[str] | tuple[str, ...] = ("Approve", "Reject"),
    timeout_s: int = 3600,
    channel: str | None = None,
):
    return await ask_approval_ds(
        resume=resume,
        context=context,
        prompt=prompt,
        options=options,
        timeout_s=timeout_s,
        channel=channel,
    )


@tool(name="ask_files", outputs=["text", "files"])
async def ask_files(
    *,
    resume=None,
    context=None,
    prompt: str,
    accept: list[str] | None = None,
    multiple: bool = True,
    timeout_s: int = 3600,
    channel: str | None = None,
):
    return await ask_files_ds(
        resume=resume,
        context=context,
        prompt=prompt,
        accept=accept,
        multiple=multiple,
        timeout_s=timeout_s,
        channel=channel,
    )


@tool(name="send_text", outputs=["ok"])
async def send_text(
    *, text: str, meta: dict[str, Any] | None = None, channel: str | None = None, context=None
):
    ch = context.channel(channel)
    await ch.send_text(text, meta=meta or {})
    return {"ok": True}


@tool(name="send_image", outputs=["ok"])
async def send_image(
    *,
    url: str | None = None,
    alt: str = "image",
    title: str | None = None,
    channel: str | None = None,
    context=None,
):
    ch = context.channel(channel)
    await ch.send_image(url=url, alt=alt, title=title)
    return {"ok": True}


@tool(name="send_file", outputs=["ok"])
async def send_file(
    *,
    url: str | None = None,
    file_bytes: bytes | None = None,
    filename: str = "file.bin",
    title: str | None = None,
    channel: str | None = None,
    context=None,
):
    ch = context.channel(channel)
    await ch.send_file(url=url, file_bytes=file_bytes, filename=filename, title=title)
    return {"ok": True}


@tool(name="send_buttons", outputs=["ok"])
async def send_buttons(
    *,
    text: str,
    buttons: list[Button],
    meta: dict[str, Any] | None = None,
    channel: str | None = None,
    context=None,
):
    ch = context.channel(channel)
    await ch.send_buttons(text=text, buttons=buttons, meta=meta or {})
    return {"ok": True}


@tool(name="get_lastest_uploads", outputs=["files"])
async def get_latest_uploads(*, clear: bool = True, context) -> list[FileRef]:
    ch = context.channel()  # any channel session will expose the same get_latest_uploads
    files = await ch.get_latest_uploads(clear=clear)
    return {"files": files}
