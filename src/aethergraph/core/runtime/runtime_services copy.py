from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
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


def install_services(services: Any) -> None:
    global _services_global, _pending_ext_services
    _services_global = services

    # Attach any services that were registered before install_services().
    ext = getattr(services, "ext_services", None)
    if isinstance(ext, dict) and _pending_ext_services:
        # Don't clobber anything that was already present.
        for name, svc in _pending_ext_services.items():
            ext.setdefault(name, svc)
        _pending_ext_services = {}

    return _current.set(services)


def ensure_services_installed(factory: Callable[[], Any]) -> Any:
    global _services_global, _pending_ext_services
    svc = _current.get() or _services_global
    if svc is None:
        svc = factory()
        _services_global = svc

        # hydrate pending external services here too
        ext = getattr(svc, "ext_services", None)
        if isinstance(ext, dict) and _pending_ext_services:
            for name, s in _pending_ext_services.items():
                ext.setdefault(name, s)
            _pending_ext_services = {}
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
    svc = current_services()
    svc.channels.set_default_channel_key(key)
    return


def get_default_channel() -> str:
    svc = current_services()
    return svc.channels.default_channel_key


def set_channel_alias(alias: str, channel_key: str) -> None:
    svc = current_services()
    svc.channels.register_alias(alias, channel_key)


def register_channel_adapter(name: str, adapter: Any) -> None:
    svc = current_services()
    svc.channel.register_adapter(name, adapter)


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
    svc = current_services()
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

    Returns the same Skill for convenience so you can use it inline:

        my_skill = register_skill(Skill(...))
    """
    reg = get_skill_registry()
    reg.register(skill, overwrite=overwrite)
    return skill


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

    Example:

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
    """
    reg = get_skill_registry()
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


def register_skill_file(path: str | Path, *, overwrite: bool = False) -> Skill:
    """
    Load a single markdown skill file and register it.

    The file must follow the same format that `parse_skill_markdown` expects:

    - Start with a YAML front matter block delimited by `---` lines, e.g.:

        ---
        id: surrogate.workflow
        title: Surrogate workflow prompts
        description: Prompts for surrogate planning and chat.
        tags: [surrogate, planning]
        domain: ml/surrogate
        modes: [planning, chat]
        version: "0.1.0"
        ---

      At minimum, `id` and `title` should be provided. Extra keys are allowed
      and kept in `Skill.config` if needed.

    - The body after the front matter is split into sections by H2 headings
      of the form:

        ## chat.system
        ## planning.header
        ## planning.binding_hints

      The heading text is used verbatim (after stripping) as the section key,
      so `## chat.system` creates a section `"chat.system"`.

    - Any text *before* the first `##` heading is stored in a special section
      named `"body"` and can be accessed via `section("body")`.

    - Headings deeper than H2 (e.g. `### ...`) are treated as normal content
      inside the current section and do *not* start a new section.

    - Empty/whitespace-only sections are ignored; section bodies are stored
      as raw markdown strings with surrounding whitespace stripped.

    This makes it possible to later retrieve structured prompt pieces with:

        skills = context.skills()
        system_prompt = skills.section("surrogate.workflow", "chat.system")

    Returns the registered `Skill` instance.
    """
    reg = get_skill_registry()
    return reg.load_file(path, overwrite=overwrite)


def register_skills_from_path(
    root: str | Path,
    *,
    pattern: str = "*.md",
    recursive: bool = True,
    overwrite: bool = False,
) -> list[Skill]:
    """
    Load and register all skill markdown files under a directory.

    Directory layout can be flat or nested:

        skills/
          surrogate-workflow.md
          coding-generic.md
          chat-default.md
          subdomain/
            planning-advanced.md

    Example:

        register_skills_from_path("skills/")
    """
    reg = get_skill_registry()
    return reg.load_path(
        root=root,
        pattern=pattern,
        recursive=recursive,
        overwrite=overwrite,
    )


# --------- Scheduler helpers --------- - (Not used)
def ensure_global_scheduler_started() -> None:
    svc = current_services()
    sched = svc.schedulers.get("global")
    if sched and not sched.is_running():
        import asyncio

        asyncio.create_task(sched.run_forever())
