from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from threading import RLock
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.core.runtime.base_service import Service
from aethergraph.services.llm.generic_client import GenericLLMClient
from aethergraph.services.skills.skill_registry import SkillRegistry
from aethergraph.services.skills.skills import Skill

_current = ContextVar("aeg_services", default=None)
# process-wide fallback (handles contextvar boundary issues)
_services_global: Any = None
# allow registering external services before main services are ready
_pending_ext_services: dict[str, Any] = {}


_pending_lock = RLock()

# Ordered operations (some things depend on earlier steps)
_pending_ops_order: list[str] = []
# Keyed storage so repeated registrations overwrite instead of duplicating
_pending_ops: dict[str, Callable[[Any], Any]] = {}
# Optional: store results if you want “handles” later
_pending_results: dict[str, Any] = {}


def _defer_op(key: str, op: Callable[[Any], Any]) -> None:
    """Register (or replace) a deferred operation."""
    with _pending_lock:
        if key not in _pending_ops:
            _pending_ops_order.append(key)
        _pending_ops[key] = op


def _flush_pending_ops(services: Any) -> None:
    """Apply all deferred operations once services exist."""
    with _pending_lock:
        keys = list(_pending_ops_order)
        _pending_ops_order.clear()
        ops = _pending_ops.copy()
        _pending_ops.clear()

    for key in keys:
        op = ops.get(key)
        if op is None:
            continue
        try:
            _pending_results[key] = op(services)
        except Exception:
            # You can choose to log here instead of raising,
            # but raising is usually better so startup fails loudly.
            raise


def _try_apply_or_defer(key: str, fn: Callable[[Any], Any]) -> Any | None:
    """
    If services installed: run now and return result.
    Else: defer it and return None.
    """
    try:
        svc = current_services()
    except RuntimeError:
        _defer_op(key, fn)
        return None
    else:
        return fn(svc)


def install_services(services: Any) -> None:
    global _services_global, _pending_ext_services
    _services_global = services

    # Attach pending ext services (your existing behavior)
    ext = getattr(services, "ext_services", None)
    if isinstance(ext, dict) and _pending_ext_services:
        for name, svc in _pending_ext_services.items():
            ext.setdefault(name, svc)
        _pending_ext_services = {}

    # NEW: apply all other pending mutations
    _flush_pending_ops(services)

    return _current.set(services)


def ensure_services_installed(factory: Callable[[], Any]) -> Any:
    global _services_global, _pending_ext_services
    svc = _current.get() or _services_global
    if svc is None:
        svc = factory()
        _services_global = svc

        # hydrate pending external services
        ext = getattr(svc, "ext_services", None)
        if isinstance(ext, dict) and _pending_ext_services:
            for name, s in _pending_ext_services.items():
                ext.setdefault(name, s)
            _pending_ext_services = {}

        # NEW: apply pending ops on first creation too
        _flush_pending_ops(svc)

    _current.set(svc)
    return svc


def current_services() -> Any:
    svc = _current.get() or _services_global
    if svc is None:
        raise RuntimeError(
            "No services installed. Call install_services(container) at app startup."
        )
    return svc


@contextmanager
def use_services(services):
    tok = _current.set(services)
    try:
        yield
    finally:
        _current.reset(tok)


# --------- Channel service helpers ---------
def get_channel_service() -> Any:
    svc = current_services()
    return svc.channels  # ChannelBus


def set_default_channel(key: str) -> None:
    def _op(svc: Any) -> None:
        svc.channels.set_default_channel_key(key)

    return _try_apply_or_defer(key, _op)


def get_default_channel() -> str:
    svc = current_services()
    return svc.channels.default_channel_key


def set_channel_alias(alias: str, channel_key: str) -> None:
    svc = current_services()
    svc.channels.register_alias(alias, channel_key)


def register_channel_adapter(name: str, adapter: Any) -> None:
    svc = current_services()
    svc.channels.register_adapter(name, adapter)


# --------- LLM service helpers ---------
def get_llm_service() -> Any:
    svc = current_services()
    return svc.llm


def register_llm_client(
    profile: str,
    provider: str,
    model: str,
    embed_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> None:
    def _op(svc: Any) -> LLMClientProtocol:
        client = svc.llm.configure_profile(
            profile=profile,
            provider=provider,
            model=model,
            embed_model=embed_model,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        return client

    key = f"llm_client:profile={profile}:provider={provider}:model={model}"
    return _try_apply_or_defer(key, _op)


# backend compatibility
set_llm_client = register_llm_client


def set_rag_llm_client(
    client: LLMClientProtocol | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    embed_model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> LLMClientProtocol:
    """Set the LLM client to use for RAG service.
    If client is provided, use it directly.
    Otherwise, create a new client using the provided parameters."""
    svc = current_services()
    if client is None:
        if provider is None or model is None or embed_model is None:
            raise ValueError(
                "Must provide provider, model, and embed_model to create RAG LLM client"
            )
        try:
            client = GenericLLMClient(
                provider=provider,
                model=model,
                embed_model=embed_model,
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create RAG LLM client: {e}") from e

    svc.rag.set_llm_client(client=client)
    return client


def set_rag_index_backend(
    *,
    backend: str | None = None,  # "sqlite" | "faiss"
    index_path: str | None = None,
    dim: int | None = None,
):
    """
    Configure the RAG index backend. If backend='faiss' but FAISS is missing,
    we log a warning and fall back to SQLite automatically.
    """
    from aethergraph.services.rag.index_factory import create_vector_index

    svc = current_services()
    # resolve defaults from settings
    s = svc.settings.rag  # AppSettings.rag bound into services
    backend = backend or s.backend
    index_path = index_path or s.index_path
    dim = dim if dim is not None else s.dim
    root = svc.settings.root

    index = create_vector_index(
        backend=backend, index_path=index_path, dim=dim, root=str(Path(root) / "rag")
    )
    svc.rag.set_index_backend(index)
    return index


# --------- Logger helpers ---------
def current_logger_factory() -> Any:
    svc = current_services()
    return svc.logger


# --------- External context services ---------
def register_context_service(name: str, service: Service) -> None:
    """
    Register an external service for NodeContext access.

    This function attaches an external service to the current service container
    under the specified name. If no container is installed yet, the service is
    stashed in a pending registry and will be attached automatically when
    install_services() is called.

    Examples:
        Register a custom database service:
        ```python
        register_context_service("mydb", MyDatabaseService())
        ```

    Args:
        name: The unique string identifier for the external service.
        service: The service instance to register.

    Returns:
        None

    Notes:
        - If called before install_services(), the service will be attached later.
        - Services are accessible via NodeContext.ext_services[name].
    """
    global _pending_ext_services

    try:
        svc = current_services()
    except RuntimeError:
        # No container yet: keep it in the staging area.
        _pending_ext_services[name] = service
        return

    # Container exists: attach immediately.
    svc.ext_services[name] = service


def get_ext_context_service(name: str) -> Service:
    """
    Retrieve an external context service by name.

    This function returns the external service registered under the given name
    from the current service container's ext_services registry.

    Examples:
        Access a registered service:
        ```python
        mydb = get_ext_context_service("mydb")
        ```

    Args:
        name: The string name of the external service to retrieve.

    Returns:
        The service instance registered under the given name, or None if not found.

    Raises:
        RuntimeError: If no services container is installed.
    """
    svc = current_services()
    return svc.ext_services.get(name)


def list_ext_context_services() -> list[str]:
    """
    List all registered external context service names.

    This function returns a list of all names for services currently registered
    in the ext_services registry of the current service container.

    Examples:
        List all available external services:
        ```python
        services = list_ext_context_services()
        print(services)
        ```

    Args:
        None

    Returns:
        A list of strings representing the names of all registered external services.
        Returns an empty list if no services are registered.

    Raises:
        RuntimeError: If no services container is installed.
    """
    svc = current_services()
    return list(svc.ext_services.keys())


# --------- MCP service helpers ---------
def set_mcp_service(mcp_service: Any) -> None:
    """
    Set the MCP service in the current service container.

    This function assigns the provided MCP service instance to the current application's
    service container, making it available for subsequent MCP client registrations and lookups.

    Examples:
        ```python
        from aethergraph.runtime import set_mcp_service
        set_mcp_service(MyMCPService())
        ```

    Args:
        mcp_service: An instance implementing the MCP service interface.

    Returns:
        None

    Notes:
        - This should be called once during application startup before registering MCP clients.
        - This is an internal function; users typically interact with MCP services via higher-level APIs.
    """
    svc = current_services()
    svc.mcp = mcp_service


def get_mcp_service() -> Any:
    """
    Retrieve the currently configured MCP service.

    This function returns the MCP service instance from the current application's
    service container. It is used to access MCP-related functionality throughout the app.

    Examples:
        ```python
        mcp = get_mcp_service()
        ```

    Args:
        None

    Returns:
        The MCP service instance currently set in the service container.

    Raises:
        RuntimeError: If no MCP service has been set.

    Notes:
        - Ensure that set_mcp_service() has been called during application initialization.
        - This is an internal function; users typically interact with MCP services via higher-level APIs.
    """
    svc = current_services()
    return svc.mcp


def register_mcp_client(name: str, client: Any) -> None:
    """
    Register a new MCP client with the current MCP service.

    This function adds a client instance to the MCP service under the specified name,
    allowing it to be accessed and managed by the MCP infrastructure.

    Examples:
        ```python
        from aethergraph.runtime import register_mcp_client
        from aethergraph.services.mcp import HttpMCPClient
        my_client = HttpMCPClient("https://mcp.example.com", ...)
        register_mcp_client("myclient", my_client)
        ```

    Args:
        name: The unique name to associate with the MCP client.
        client: The client instance to register.

    Returns:
        None

    Raises:
        RuntimeError: If no MCP service has been installed via set_mcp_service().

    """
    svc = current_services()
    if svc.mcp is None:
        raise RuntimeError("No MCP service installed. Call set_mcp_service() first.")
    svc.mcp.register(name, client)


def list_mcp_clients() -> list[str]:
    """
    List all registered MCP client names in the current MCP service.

    This function returns a list of all client names that have been registered
    with the MCP service, allowing for discovery and management of available clients.

    Examples:
        ```python
        from aethergraph.runtime import list_mcp_clients
        clients = list_mcp_clients()
        print(clients)
        ```

    Args:
        None

    Returns:
        A list of strings representing the names of registered MCP clients.
        Returns an empty list if no MCP service is installed or no clients are registered.
    """
    svc = current_services()
    if svc.mcp:
        return svc.mcp.list_clients()
    return []


# --------- Skill registry helpers ---------
def get_skill_registry() -> SkillRegistry:
    svc = current_services()
    return svc.skills_registry


def register_skill(skill: Skill, *, overwrite: bool = False) -> Skill:
    """
    Register an existing Skill object into the global registry.

    This method adds a `Skill` instance to the global `SkillRegistry`, making it
    available for use throughout the application. The `overwrite` flag determines
    whether an existing skill with the same ID will be replaced.

    Examples:
        Registering a skill object:
        ```python
        skill = Skill(id="example.skill", title="Example Skill")
        register_skill(skill)
        ```

        Overwriting an existing skill:
        ```python
        skill = Skill(id="example.skill", title="Updated Skill")
        register_skill(skill, overwrite=True)
        ```

    Args:
        skill: The `Skill` object to register.
        overwrite: Whether to overwrite an existing skill with the same ID. Default is `False`.

    Returns:
        Skill: The registered `Skill` instance.

    """

    def _op(svc: Any) -> "Skill":
        reg = svc.skills_registry
        reg.register(skill, overwrite=overwrite)
        return skill

    # Key should be stable and allow overwriting the deferred op if called again.
    # Usually skill.id is the right identity here.
    key = f"skills:obj:{skill.id}:overwrite={overwrite}"
    return _try_apply_or_defer(key, _op)


def register_skill_inline(
    *,
    id: str,
    title: str,
    description: str = "",
    tags: list[str] | None = None,
    domain: str | None = None,
    modes: list[str] | None = None,
    version: str | None = None,
    config: dict[str, Any] | None = None,
    sections: dict[str, str] | None = None,
    overwrite: bool = False,
) -> Skill:
    """
    Define and register a Skill entirely in Python.

    This method allows you to define a Skill inline with all its metadata and sections,
    and directly register it into the global Skill registry.

    Examples:
        Registering a skill with basic metadata and sections:
        ```python
        register_skill_inline(
            id="surrogate.workflow",
            title="Surrogate workflow planning",
            description="Prompts and patterns for surrogate planning.",
            tags=["surrogate", "planning"],
            modes=["planning"],
            sections={
                "planning.header": "...",
                "planning.binding_hints": "...",
                "chat.system": "...",
            },
        )
        ```

    Args:
        id (str): The unique identifier for the Skill. (Required)
        title (str): A human-readable title for the Skill. (Required)
        description (str): A short description of the Skill's purpose. (Optional)
        tags (list[str]): A list of tags for categorization. (Optional)
        domain (str): The domain or namespace for the Skill. (Optional)
        modes (list[str]): The operational modes supported by the Skill. (Optional)
        version (str): The version string for the Skill. (Optional)
        config (dict[str, Any]): Additional configuration data. (Optional)
        sections (dict[str, str]): A dictionary mapping section names to their content. (Optional)
        overwrite (bool): Whether to overwrite an existing Skill with the same ID. (Optional)

    Returns:
        Skill: The registered Skill instance.
    """

    def _op(svc: Any) -> "Skill":
        reg = svc.skills_registry
        return reg.register_inline(
            id=id,
            title=title,
            description=description,
            tags=tags,
            domain=domain,
            modes=modes,
            version=version,
            config=config,
            sections=sections,
            overwrite=overwrite,
        )

    # Include overwrite, and optionally version to avoid surprising replacements.
    key = f"skills:inline:{id}:overwrite={overwrite}:version={version or ''}"
    return _try_apply_or_defer(key, _op)


def register_skill_file(path: str | Path, *, overwrite: bool = False) -> Skill:
    """
    Load a single markdown skill file and register it.

    This function processes a markdown file containing skill definitions and
    registers it into the global skill registry. The file must adhere to the
    expected format for parsing skill metadata and sections.

    Examples:
        Registering a skill from a markdown file:
        ```python
        skill = register_skill_file("skills/surrogate-workflow.md")
        ```

    Args:
        path: The path to the markdown file to load.
        overwrite: Whether to overwrite an existing skill with the same ID. (Optional, default: False)

    Returns:
        Skill: The registered `Skill` instance.

    Notes:
        To start the server and load all desired packages:
        1. Open a terminal and navigate to the project directory.
        2. Run the server using the appropriate command (e.g., `python -m aethergraph.server`).
        3. Ensure all required dependencies are installed via `pip install -r requirements.txt`.

    """

    p = str(path)

    def _op(svc: Any) -> Skill:
        reg = svc.skills_registry
        return reg.load_file(path, overwrite=overwrite)

    p = str(path)

    def _op(svc: Any) -> "Skill":
        reg = svc.skills_registry
        return reg.load_file(path, overwrite=overwrite)

    key = f"skills:file:{p}:overwrite={overwrite}"
    return _try_apply_or_defer(key, _op)


def register_skills_from_path(
    root: str | Path,
    *,
    pattern: str = "*.md",
    recursive: bool = True,
    overwrite: bool = False,
) -> list[Skill]:
    """
    Load and register all skill markdown files under a directory.

    This method scans the specified directory for markdown files matching the
    given pattern, parses their content into `Skill` objects, and registers
    them into the global skill registry. The directory can have a flat or
    nested structure.

    Examples:
        Register all skills in a flat directory:
        ```python
        register_skills_from_path("skills/")
        ```

        Register skills in a nested directory structure:
        ```python
        register_skills_from_path("skills/", recursive=True)
        ```

        Use a custom file pattern to filter files:
        ```python
        register_skills_from_path("skills/", pattern="*.skill.md")
        ```

    Args:
        root: The root directory to scan for skill files.
        pattern: A glob pattern to match skill files. Default is `"*.md"`.
        recursive: Whether to scan subdirectories recursively. Default is `True`.
        overwrite: Whether to overwrite existing skills with the same ID. Default is `False`.

    Returns:
        list[Skill]: A list of all registered `Skill` objects.

    Notes:
        To start the server and load all desired packages:
        1. Open a terminal and navigate to the project directory.
        2. Run the server using the appropriate command (e.g., `python -m aethergraph.server`).
        3. Ensure all required dependencies are installed via `pip install -r requirements.txt`.

    """
    root_str = str(root)

    def _op(svc: Any) -> list[Skill]:
        return svc.skills_registry.load_path(
            root=root_str,
            pattern=pattern,
            recursive=recursive,
            overwrite=overwrite,
        )

    key = f"skills:path:{root_str}:pattern={pattern}:recursive={recursive}:overwrite={overwrite}"
    return _try_apply_or_defer(key, _op)


# --------- Scheduler helpers --------- - (Not used)
def ensure_global_scheduler_started() -> None:
    svc = current_services()
    sched = svc.schedulers.get("global")
    if sched and not sched.is_running():
        import asyncio

        asyncio.create_task(sched.run_forever())
