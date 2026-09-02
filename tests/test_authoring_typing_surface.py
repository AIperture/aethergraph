from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, get_args, get_type_hints

from aethergraph.contracts.authoring import NodeContextProtocol
from aethergraph.contracts.services.llm import (
    EmbeddingClientProtocol,
    ImageGenerationClientProtocol,
    LLMClientProtocol,
)
from aethergraph.services.artifacts.canonical_public import CanonicalPublicArtifactFacade
from aethergraph.services.channel.session import ChannelSession


def _assert_closed_callable(value: object) -> None:
    signature = inspect.signature(value)
    assert all(
        parameter.kind is not inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    assert signature.return_annotation not in {inspect.Signature.empty, Any, "Any"}
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        assert parameter.annotation not in {inspect.Parameter.empty, Any, "Any"}


def _contains_any(annotation: object) -> bool:
    return annotation is Any or any(_contains_any(item) for item in get_args(annotation))


def test_authoring_protocols_have_closed_critical_call_signatures() -> None:
    for owner, names in (
        (
            LLMClientProtocol,
            ("estimate", "generate", "generate_stream", "estimate_chat_request", "chat"),
        ),
        (EmbeddingClientProtocol, ("embed_result", "embed", "embed_one")),
        (ImageGenerationClientProtocol, ("generate_image",)),
        (
            NodeContextProtocol,
            (
                "logger",
                "channel",
                "artifacts",
                "kv",
                "memory",
                "runner",
                "triggers",
                "viz",
                "llm",
                "embedding",
                "image_model",
                "state",
            ),
        ),
        (
            CanonicalPublicArtifactFacade,
            ("save_file", "save_url", "save_bytes", "save_text", "save_json"),
        ),
        (
            ChannelSession,
            (
                "send_structured_output",
                "send_tool_activity",
                "send_phase",
                "send_text",
                "send_message",
                "send_rich",
                "send_run_card",
                "send_image",
                "send_file",
                "send_buttons",
                "stream_text",
            ),
        ),
    ):
        for name in names:
            _assert_closed_callable(getattr(owner, name))


def test_authoring_protocol_fields_do_not_collapse_to_any() -> None:
    hints = get_type_hints(NodeContextProtocol)
    assert hints
    assert all(not _contains_any(annotation) for annotation in hints.values())


def test_distribution_declares_inline_typing() -> None:
    marker = Path(__file__).parents[1] / "src" / "aethergraph" / "py.typed"
    assert marker.is_file()
