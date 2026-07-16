from __future__ import annotations

from pathlib import Path

import pytest

from aethergraph.config.dotenv_writer import read_dotenv, replace_dotenv
from aethergraph.config.llm import LLMProfile
from aethergraph.config.llm_env import encode_llm_profiles_env
from aethergraph.config.loader import load_settings


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
