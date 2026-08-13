from pathlib import Path

from aethergraph.core.runtime.node_context import NodeContext
from aethergraph.core.runtime.node_services import NodeServices
from aethergraph.plugins.agents.chat_agent.default_chat_agent import default_chat_agent
import aethergraph.runtime as runtime_api
from aethergraph.services.container.default_container import DefaultContainer


def test_default_chat_is_the_only_bundled_agent() -> None:
    agents_root = Path(__file__).parents[1] / "src" / "aethergraph" / "plugins" / "agents"
    bundled = {path.relative_to(agents_root).parts[0] for path in agents_root.rglob("*.py")}

    assert bundled == {"chat_agent"}
    assert default_chat_agent.name == "default_chat_agent"


def test_legacy_skill_runtime_surface_is_absent() -> None:
    assert not hasattr(NodeContext, "skills")
    assert "skills" not in NodeServices.__dataclass_fields__
    assert "skills_registry" not in DefaultContainer.__dataclass_fields__

    for name in (
        "get_skill_registry",
        "register_skill",
        "register_skill_file",
        "register_skill_inline",
        "register_skills_from_path",
    ):
        assert not hasattr(runtime_api, name)
