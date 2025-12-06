from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.llm.generic_client import GenericLLMClient

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
def register_context_service(name: str, service: Any) -> None:
    """
    Register an external service that NodeContext can access.

    If a services container is already installed, attach it immediately.
    Otherwise, stash it in a global pending registry and attach it the
    next time install_services() is called.
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


def get_ext_context_service(name: str) -> Any:
    svc = current_services()
    return svc.ext_services.get(name)


def list_ext_context_services() -> list[str]:
    svc = current_services()
    return list(svc.ext_services.keys())


# --------- MCP service helpers ---------
def set_mcp_service(mcp_service: Any) -> None:
    svc = current_services()
    svc.mcp = mcp_service


def get_mcp_service() -> Any:
    svc = current_services()
    return svc.mcp


def register_mcp_client(name: str, client: Any) -> None:
    svc = current_services()
    if svc.mcp is None:
        raise RuntimeError("No MCP service installed. Call set_mcp_service() first.")
    svc.mcp.register(name, client)


def list_mcp_clients() -> list[str]:
    svc = current_services()
    if svc.mcp:
        return svc.mcp.list_clients()
    return []


# --------- Scheduler helpers --------- - (Not used)
def ensure_global_scheduler_started() -> None:
    svc = current_services()
    sched = svc.schedulers.get("global")
    if sched and not sched.is_running():
        import asyncio

        asyncio.create_task(sched.run_forever())
