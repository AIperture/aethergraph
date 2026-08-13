from __future__ import annotations

from pathlib import Path

from pydantic import SecretStr
import pytest

from aethergraph.api.v1.schemas.settings import SlackPayload, SlackView
from aethergraph.config.config import AppSettings, SlackSettings, TelegramSettings
from aethergraph.config.dotenv_writer import read_dotenv, replace_dotenv
from aethergraph.config.llm import (
    EmbeddingProfile,
    ImageGenerationProfileSettings,
    LLMProfile,
)
from aethergraph.config.llm_env import (
    encode_embedding_profiles_env,
    encode_image_generation_profiles_env,
    encode_llm_profiles_env,
)
from aethergraph.config.loader import load_settings
from aethergraph.services.channel import factory as channel_factory
from aethergraph.services.llm.profiles import (
    ChatCapabilityOverrides,
    EmbeddingCapabilityOverrides,
    ImageGenerationCapabilityOverrides,
)


def test_explicit_settings_file_does_not_use_discovered_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = tmp_path / "cwd"
    discovered.mkdir()
    (discovered / ".env").write_text(
        "AETHERGRAPH_LLM__DEFAULT__MODEL=discovered-model\n",
        encoding="utf-8",
    )
    explicit = tmp_path / "managed.env"
    explicit.write_text(
        "AETHERGRAPH_LLM__DEFAULT__MODEL=managed-model\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(discovered)
    monkeypatch.delenv("AETHERGRAPH_LLM__DEFAULT__MODEL", raising=False)

    settings = load_settings(env_file=explicit)

    assert settings.llm.default.model == "managed-model"


def test_explicit_settings_file_must_exist(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="AETHERGRAPH_ENV_FILE not found"):
        load_settings(env_file=tmp_path / "missing.env")


def test_replace_dotenv_removes_stale_rows(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("STALE=value\nKEEP=old\n", encoding="utf-8")

    replace_dotenv(target, {"KEEP": "new", "ADDED": "value"}, header=("Managed",))

    assert read_dotenv(target) == {"KEEP": "new", "ADDED": "value"}
    assert target.read_text(encoding="utf-8").startswith("# Managed\n\n")


def test_encode_llm_profiles_env_is_deterministic_and_complete() -> None:
    rows = encode_llm_profiles_env(
        {
            "summarizer": LLMProfile(
                provider="anthropic",
                model="claude-test",
                vision_enabled=True,
                vision_accepted_mime_types=["image/png"],
            ),
            "default": LLMProfile(provider="openai", model="gpt-test"),
        }
    )

    assert next(iter(rows)) == "AETHERGRAPH_LLM__DEFAULT__PROVIDER"
    assert rows["AETHERGRAPH_LLM__PROFILES__SUMMARIZER__MODEL"] == "claude-test"
    assert (
        rows["AETHERGRAPH_LLM__PROFILES__SUMMARIZER__VISION_ACCEPTED_MIME_TYPES"] == '["image/png"]'
    )


def test_operation_profile_environment_writers_round_trip_independently(
    tmp_path: Path,
) -> None:
    rows = encode_llm_profiles_env(
        {
            "default": LLMProfile(
                provider="openai",
                model="gpt-test",
                capability_overrides=ChatCapabilityOverrides(
                    structured_output="supported"
                ),
            )
        }
    )
    rows.update(
        encode_embedding_profiles_env(
            {
                "default": EmbeddingProfile(
                    provider="openai",
                    model="text-embedding-3-small",
                ),
                "search": EmbeddingProfile(
                    provider="google",
                    model="gemini-embedding-001",
                    endpoint_id="gemini_embeddings",
                    capability_overrides=EmbeddingCapabilityOverrides(
                        dimensions="unsupported"
                    ),
                ),
            }
        )
    )
    rows.update(
        encode_image_generation_profiles_env(
            {
                "default": ImageGenerationProfileSettings(
                    provider="openai",
                    model="gpt-image-1",
                    endpoint_id="openai_images",
                ),
                "design": ImageGenerationProfileSettings(
                    provider="google",
                    model="imagen-4.0-generate-001",
                    endpoint_id="gemini_images",
                    count=2,
                    capability_overrides=ImageGenerationCapabilityOverrides(
                        image_editing="supported"
                    ),
                ),
            }
        )
    )
    target = tmp_path / "operation-profiles.env"
    replace_dotenv(target, rows)

    settings = load_settings(env_file=target)

    assert settings.llm.default.model == "gpt-test"
    assert settings.llm.default.capability_overrides.structured_output == "supported"
    assert settings.embed.default.model == "text-embedding-3-small"
    assert settings.embed.profiles["search"].model == "gemini-embedding-001"
    assert settings.embed.profiles["search"].endpoint_id == "gemini_embeddings"
    assert settings.embed.profiles["search"].capability_overrides.dimensions == "unsupported"
    assert settings.image_generation.default.endpoint_id == "openai_images"
    assert settings.image_generation.profiles["design"].count == 2
    assert (
        settings.image_generation.profiles["design"].capability_overrides.image_editing
        == "supported"
    )


@pytest.mark.parametrize(
    ("encoder", "unexpected_field", "label"),
    [
        (encode_embedding_profiles_env, "vision_enabled", "Embedding"),
        (encode_image_generation_profiles_env, "prompt_cache_policy", "Image Generation"),
    ],
)
def test_operation_profile_environment_writers_reject_cross_operation_fields(
    encoder,
    unexpected_field: str,
    label: str,
) -> None:
    with pytest.raises(ValueError, match=f"Unknown {label} profile fields"):
        encoder({"default": {unexpected_field: True}})


def test_provider_settings_expose_only_supported_transport_configuration() -> None:
    assert set(SlackSettings.model_fields) == {
        "integration_id",
        "enabled",
        "bot_token",
        "app_token",
    }
    assert set(TelegramSettings.model_fields) == {
        "integration_id",
        "enabled",
        "bot_token",
    }
    assert set(SlackView.model_fields) == {
        "integration_id",
        "enabled",
        "bot_token",
    }
    assert set(SlackPayload.model_fields) == {
        "integration_id",
        "enabled",
        "bot_token",
    }


def test_slack_delivery_adapter_does_not_require_webhook_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object()
    observed: dict[str, str] = {}

    def fake_slack_adapter(*, bot_token: str):
        observed["bot_token"] = bot_token
        return adapter

    monkeypatch.setattr(channel_factory, "SlackChannelAdapter", fake_slack_adapter)
    settings = AppSettings(
        workspace=str(tmp_path),
        slack=SlackSettings(enabled=True, bot_token=SecretStr("xoxb-test")),
    )

    adapters = channel_factory.make_channel_adapters_from_env(settings)

    assert adapters["slack"] is adapter
    assert observed == {"bot_token": "xoxb-test"}
