# redirect runtime service imports for clean imports

from aethergraph.core.runtime.ad_hoc_context import open_session
from aethergraph.core.runtime.runtime_services import (
    # logger service helpers
    current_logger_factory,
    current_services,
    ensure_services_installed,
    # channel service helpers
    get_channel_service,
    get_default_channel,
    get_ext_context_service,
    # llm service helpers
    get_llm_service,
    get_mcp_service,
    # general service management
    install_services,
    list_ext_context_services,
    list_mcp_clients,
    register_channel_adapter,
    # external context service helpers
    register_context_service,
    register_llm_client,
    register_mcp_client,
    set_channel_alias,
    set_default_channel,
    # mcp service helpers
    set_mcp_service,
    set_rag_index_backend,
    set_rag_llm_client,
)

__all__ = [
    # general service management
    "install_services",
    "ensure_services_installed",
    "current_services",
    # channel service helpers
    "get_channel_service",
    "set_default_channel",
    "get_default_channel",
    "set_channel_alias",
    "register_channel_adapter",
    # llm service helpers
    "get_llm_service",
    "register_llm_client",
    "set_rag_llm_client",
    "set_rag_index_backend",
    # logger service helpers
    "current_logger_factory",
    # external context service helpers
    "register_context_service",
    "get_ext_context_service",
    "list_ext_context_services",
    # mcp service helpers
    "set_mcp_service",
    "get_mcp_service",
    "register_mcp_client",
    "list_mcp_clients",
    # ad-hoc context
    "open_session",
]
