# services/memory/facade/introspection.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .types import MemoryFacadeInterface


class IntrospectionMixin:
    """
    Convenience helpers for inspecting the memory scope / timeline.

    Purely debug / observability, no behavioral impact.
    """

    # ----- Core ID helpers -----

    def scope_id(self: MemoryFacadeInterface) -> str | None:
        """
        Return the effective memory scope ID for this facade, if available.

        This is usually the same value passed to `derive_timeline_id`,
        e.g. "session:...", "user:...", "run:...", "org:...", or a custom override.
        """
        # Prefer the facade-level attribute if present
        if hasattr(self, "memory_scope_id"):
            return self.memory_scope_id
        # Fallback to scope-based computation if someone removes that attribute
        scope = getattr(self, "scope", None)
        if scope is not None and hasattr(scope, "memory_scope_id"):
            try:
                return scope.memory_scope_id()
            except Exception:
                pass
        return None

    def memory_level(self: MemoryFacadeInterface) -> str | None:
        """
        Logical memory level requested for this facade (run/session/user/org/scope),
        if available from the Scope.
        """
        scope = getattr(self, "scope", None)
        if scope is not None and hasattr(scope, "memory_level"):
            return scope.memory_level
        return None

    def bucket_level(self: MemoryFacadeInterface) -> str | None:
        """
        Infer the logical memory "bucket level" from the memory_scope_id.

        Expected values:
          - "session"
          - "user"
          - "run"
          - "org"
          - "app"
          - None (if unknown / global)
        """
        scope_id = self.scope_id()
        if not scope_id:
            return None
        # memory_scope_id is of form "<prefix>:<value>" for scoped cases
        if ":" in scope_id:
            return scope_id.split(":", 1)[0]
        # could be "global" or some custom key
        return None

    def timeline(self: MemoryFacadeInterface) -> str | None:
        """
        Return the timeline ID for this memory facade.

        This is the primary key used for appending events.
        """
        return getattr(self, "timeline_id", None)

    # ----- Structured info -----

    def scope_info(self: MemoryFacadeInterface) -> dict[str, Any]:
        """
        Return a structured, debug-friendly description of the current memory scope.

        Example shape:
        {
            "timeline_id": "...",
            "memory_scope_id": "user:user-A",
            "bucket_level": "user",
            "run_id": "...",
            "session_id": "...",
            "graph_id": "...",
            "node_id": "...",
            "org_id": "...",
            "user_id": "...",
            "client_id": "...",
            "app_id": "...",
            "agent_id": "...",
            "flow_id": "...",
            "has_indices": True,
        }
        """
        scope = getattr(self, "scope", None)

        info: dict[str, Any] = {
            "timeline_id": getattr(self, "timeline_id", None),
            "memory_scope_id": self.scope_id(),
            "memory_level": self.memory_level(),
            "bucket_level": self.bucket_level(),
            "has_indices": bool(getattr(self, "scoped_indices", None) is not None),
            "run_id": getattr(self, "run_id", None),
            "session_id": getattr(self, "session_id", None),
            "graph_id": getattr(self, "graph_id", None),
            "node_id": getattr(self, "node_id", None),
        }

        if scope is not None:
            for attr in (
                "org_id",
                "user_id",
                "client_id",
                "app_id",
                "agent_id",
                "session_id",
                "run_id",
                "graph_id",
                "node_id",
                "flow_id",
                "mode",
            ):
                val = getattr(scope, attr, None)
                if val is not None:
                    info[attr] = val

        return info

    def debug_print_scope(self: MemoryFacadeInterface, prefix: str = "[MEM]") -> None:
        """
        Convenience method to print the current scope info to stdout.

        Useful inside quick scripts / demos / tests.
        """
        info = self.scope_info()
        print(
            f"{prefix} memory_scope_id={info.get('memory_scope_id')} "
            f"memory_level={info.get('memory_level')} "
            f"bucket_level={info.get('bucket_level')}"
        )
        print(
            f"{prefix} timeline_id={info.get('timeline_id')} has_indices={info.get('has_indices')}"
        )
        for key in (
            "org_id",
            "user_id",
            "client_id",
            "app_id",
            "agent_id",
            "session_id",
            "run_id",
            "graph_id",
            "node_id",
            "flow_id",
        ):
            if key in info and info[key] is not None:
                print(f"{prefix} {key}={info[key]}")
