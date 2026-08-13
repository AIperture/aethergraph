from __future__ import annotations

from types import SimpleNamespace

import pytest

from aethergraph.core.runtime.runtime_services import register_llm_client, use_services


class _ProfileService:
    def __init__(self, result=None) -> None:
        self.calls: list[dict] = []
        self.result = result

    def configure_profile(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def test_register_llm_client_keeps_chat_and_embedding_configuration_separate() -> None:
    chat_client = object()
    llm = _ProfileService(chat_client)
    embeddings = _ProfileService()
    services = SimpleNamespace(llm=llm, embed_service=embeddings)

    with use_services(services), pytest.warns(DeprecationWarning, match="embed_model"):
        result = register_llm_client(
            "search",
            "openai",
            "gpt-5-mini",
            embed_model="text-embedding-3-small",
            base_url="https://api.example/v1",
            api_key="secret",
            timeout=30.0,
        )

    assert result is chat_client
    assert llm.calls == [
        {
            "profile": "search",
            "provider": "openai",
            "model": "gpt-5-mini",
            "base_url": "https://api.example/v1",
            "api_key": "secret",
            "timeout": 30.0,
        }
    ]
    assert embeddings.calls == [
        {
            "name": "search",
            "provider": "openai",
            "model": "text-embedding-3-small",
            "base_url": "https://api.example/v1",
            "api_key": "secret",
            "timeout": 30.0,
        }
    ]


def test_register_llm_client_fails_before_chat_when_legacy_embedding_is_disabled() -> None:
    llm = _ProfileService(object())
    services = SimpleNamespace(llm=llm, embed_service=None)

    with (
        use_services(services),
        pytest.warns(DeprecationWarning, match="embed_model"),
        pytest.raises(RuntimeError, match="enabled embedding service"),
    ):
        register_llm_client(
            "default",
            "openai",
            "gpt-5-mini",
            embed_model="text-embedding-3-small",
        )

    assert llm.calls == []
