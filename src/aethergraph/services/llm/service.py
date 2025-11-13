from typing import Dict, Optional
from .generic_client import GenericLLMClient
from .providers import Provider
from ..secrets.base import Secrets

import asyncio
import httpx


class LLMService:
    """Holds multiple LLM clients (default + named profiles)."""
    def __init__(self, clients: Dict[str, GenericLLMClient], secrets: Secrets | None = None):
        self._clients = clients
        self._secrets = secrets

    def get(self, name: str = "default") -> GenericLLMClient:
        return self._clients[name]

    def has(self, name: str) -> bool:
        return name in self._clients

    async def aclose(self):
        for c in self._clients.values():
            await c.aclose()

    # --- Runtime profile helpers ---------------------------------
    def configure_profile(
        self,
        profile: str = "default",
        *,
        provider: Optional[Provider] = None,
        model: Optional[str] = None,
        embed_model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> GenericLLMClient:
        """
        Create or update a profile in memory. Returns the client.
        Does NOT persist anything outside this process.
        """
        if profile not in self._clients:
            client = GenericLLMClient(
                provider=provider,
                model=model,
                embed_model=embed_model,
                base_url=base_url,
                api_key=api_key,
                azure_deployment=azure_deployment,
                timeout=timeout or 60.0,
            )
            self._clients[profile] = client
            return client

        c = self._clients[profile]
        if provider is not None:
            c.provider = provider  # type: ignore[assignment]
        if model is not None:
            c.model = model
        if base_url is not None:
            c.base_url = base_url
        if api_key is not None:
            c.api_key = api_key
        if azure_deployment is not None:
            c.azure_deployment = azure_deployment
        if timeout is not None:
            # Recreate client with new timeout
            old_client = c._client
            c._client = httpx.AsyncClient(timeout=timeout)
            try:
                # best-effort async close
                asyncio.create_task(old_client.aclose())
            except RuntimeError:
                pass
        return c

    # --- Quick start helpers ---
    def set_key(self, provider: str, model: str, api_key: str, profile: str = "default") -> GenericLLMClient:
        """
        Quickly set/override an API key for a profile at runtime (in-memory).
        Creates the profile if it doesn't exist yet.
        """
        return self.configure_profile(
            profile=profile,
            provider=provider,   # type: ignore[arg-type]
            model=model,
            api_key=api_key,
        )

    def persist_key(self, secret_name: str, api_key: str):
        """
        Optional: store the key via the installed Secrets provider for later runs.
        Implement only after Secrets supports write (e.g., dev file store). Env-based usually won't.
        """
        raise NotImplementedError("persist_key not implemented in this Secrets provider")
        if not self._secrets or not hasattr(self._secrets, "set"):
            raise RuntimeError("Secrets provider is not writable")
        self._secrets.set(secret_name, api_key)  # type: ignore[attr-defined]
