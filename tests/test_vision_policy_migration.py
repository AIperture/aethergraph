import pytest

from aethergraph.config.llm import LLMProfile
from aethergraph.services.llm.compat import chat_profile_from_legacy, migrate_legacy_vision_policy


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('capability', ['unknown', 'supported', 'unsupported'])
def test_migration_preserves_exact_canonical_profile(enabled, capability):
    legacy = LLMProfile(vision_enabled=enabled, capability_overrides={'image_input': capability})
    before = chat_profile_from_legacy(legacy)
    migrated = migrate_legacy_vision_policy(legacy)
    assert migrated.vision_policy_version == 2
    assert chat_profile_from_legacy(migrated).model_dump() == before.model_dump()
    assert migrate_legacy_vision_policy(migrated) is migrated


def test_new_permission_is_independent_of_capability_and_remote_urls():
    profile = LLMProfile(vision_policy_version=2, vision_enabled=True)
    canonical = chat_profile_from_legacy(profile)
    assert canonical.input_policy.image_input_enabled
    assert not canonical.input_policy.allow_remote_urls
    assert canonical.capability_overrides.image_input == 'unknown'


def test_policy_version_survives_environment_serialization(tmp_path):
    from aethergraph.config.llm_env import encode_llm_profiles_env
    from aethergraph.config.dotenv_writer import replace_dotenv
    from aethergraph.config.loader import load_settings
    path = tmp_path / 'profiles.env'
    profile = migrate_legacy_vision_policy(LLMProfile(vision_enabled=True))
    replace_dotenv(path, encode_llm_profiles_env({'default': profile}))
    loaded = load_settings(env_file=path).llm.default
    assert loaded.vision_policy_version == 2
    assert chat_profile_from_legacy(loaded).model_dump() == chat_profile_from_legacy(profile).model_dump()
