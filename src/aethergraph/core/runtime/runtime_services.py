from contextvars import ContextVar
from typing import Any, Callable, Optional
from contextlib import contextmanager

from aethergraph.contracts.services.llm import LLMClientProtocol
from aethergraph.services.llm.generic_client import GenericLLMClient


_current = ContextVar("aeg_services", default=None)
# process-wide fallback (handles contextvar boundary issues)
_services_global: Any = None

def install_services(services: Any) -> None:
    global _services_global
    _services_global = services
    return _current.set(services)

def ensure_services_installed(factory: Callable[[], Any]) -> Any:
    global _services_global
    svc = _current.get() or _services_global
    if svc is None:
        svc = factory()
        _services_global = svc
    _current.set(svc)  # keep ContextVar in sync for this context
    return svc

def current_services() -> Any:
    svc = _current.get() or _services_global
    if svc is None:
        raise RuntimeError("No services installed. Call install_services(container) at app startup.")
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
    return svc.channels # ChannelBus

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
            embed_model: Optional[str] = None,
            base_url: Optional[str] = None,
            api_key: Optional[str] = None,
            timeout: Optional[float] = None) -> None:
    
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

def set_rag_llm_client(client: Optional[LLMClientProtocol] = None,
                       *,
                       provider: Optional[str] = None,
                       model: Optional[str] = None,
                       embed_model: Optional[str] = None,
                       base_url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       timeout: Optional[float] = None) -> LLMClientProtocol:
    """Set the LLM client to use for RAG service.
    If client is provided, use it directly.
    Otherwise, create a new client using the provided parameters."""
    svc = current_services()
    if client is None:
        if provider is None or model is None or embed_model is None:
            raise ValueError("Must provide provider, model, and embed_model to create RAG LLM client")
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

# --------- Logger helpers --------- 
def current_logger_factory() -> Any:
    svc = current_services()
    return svc.logger



# --------- External context services ---------
def register_context_service(name: str, service: Any) -> None:
    svc = current_services()
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