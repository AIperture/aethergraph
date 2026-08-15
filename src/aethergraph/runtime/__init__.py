# redirect runtime service imports for clean imports

from aethergraph.core.runtime.ad_hoc_context import open_session
from aethergraph.core.runtime.run_manager import RunManager
from aethergraph.core.runtime.run_types import (
    RunImportance,
    RunOrigin,
    RunRecord,
    RunStatus,
    RunVisibility,
)
from aethergraph.core.runtime.runtime_services import (
    # logger service helpers
    current_logger_factory,
    current_services,
    ensure_services_installed,
    # channel service helpers
    get_channel_service,
    get_ext_context_service,
    # llm service helpers
    get_llm_service,
    # general service management
    install_services,
    list_ext_context_services,
    register_channel_adapter,
    # external context service helpers
    register_context_service,
    register_llm_client,
)
from aethergraph.runtime.contracts import (
    RuntimeArtifactRecord,
    RuntimeArtifactScope,
    RuntimeGraphRegistration,
    RuntimeIdentity,
    RuntimeModelProfile,
    RuntimeOpenRequest,
    RuntimeRegistrationSnapshot,
    RuntimeRunRecord,
    RuntimeRunRequest,
    RuntimeRunStatus,
    RuntimeSemanticEvent,
    RuntimeStagedArtifact,
)
from aethergraph.runtime.embedded import (
    EmbeddedRuntime,
    RuntimeIntegration,
    open_embedded_runtime,
)
from aethergraph.runtime.errors import (
    EmbeddedRuntimeError,
    RuntimeGraphLoadError,
    RuntimeInteractionError,
    RuntimeNotReadyError,
)

__all__ = [
    # general service management
    "install_services",
    "ensure_services_installed",
    "current_services",
    # channel service helpers
    "get_channel_service",
    "register_channel_adapter",
    # llm service helpers
    "get_llm_service",
    "register_llm_client",
    # logger service helpers
    "current_logger_factory",
    # external context service helpers
    "register_context_service",
    "get_ext_context_service",
    "list_ext_context_services",
    # ad-hoc context
    "open_session",
    # run manager and types
    "RunManager",
    "RunRecord",
    "RunStatus",
    "RunOrigin",
    "RunImportance",
    "RunVisibility",
    # supported embedded Host boundary
    "EmbeddedRuntime",
    "EmbeddedRuntimeError",
    "RuntimeArtifactRecord",
    "RuntimeArtifactScope",
    "RuntimeGraphLoadError",
    "RuntimeGraphRegistration",
    "RuntimeIdentity",
    "RuntimeIntegration",
    "RuntimeInteractionError",
    "RuntimeNotReadyError",
    "RuntimeModelProfile",
    "RuntimeOpenRequest",
    "RuntimeRegistrationSnapshot",
    "RuntimeRunRecord",
    "RuntimeRunRequest",
    "RuntimeRunStatus",
    "RuntimeSemanticEvent",
    "RuntimeStagedArtifact",
    "open_embedded_runtime",
]
