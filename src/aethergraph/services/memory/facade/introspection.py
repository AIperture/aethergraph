# services/memory/facade/introspection.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import MemoryFacadeProtocol


class IntrospectionMixin:
    """
    Convenience helpers for inspecting the memory scope / timeline.

    Purely debug / observability, no behavioral impact.
    """

    # ----- Core ID helpers -----

    def scope_id(self: MemoryFacadeProtocol) -> str | None:
        """
        Return the effective memory scope ID for this facade.

        This value usually matches the scope identifier used to derive the
        timeline (for example `session:...`, `user:...`, or `run:...`).

        Examples:
            Read the scope ID:
            ```python
            scope_id = context.memory().scope_id()
            ```

            Handle missing scope IDs defensively:
            ```python
            scope_id = context.memory().scope_id() or "global"
            ```

        Args:
            None.

        Returns:
            str | None: The effective memory scope ID, or None if unavailable.
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

    def memory_level(self: MemoryFacadeProtocol) -> str | None:
        """
        Return the logical memory level requested for this facade.

        The value is read from the attached scope when available.

        Examples:
            Read the configured memory level:
            ```python
            level = context.memory().memory_level()
            ```

            Fallback to `"scope"` when unset:
            ```python
            level = context.memory().memory_level() or "scope"
            ```

        Args:
            None.

        Returns:
            str | None: The memory level (`"scope"`, `"session"`, `"run"`, `"user"`, `"org"`) or None.
        """
        scope = getattr(self, "scope", None)
        if scope is not None and hasattr(scope, "memory_level"):
            return scope.memory_level
        return None

    def bucket_level(self: MemoryFacadeProtocol) -> str | None:
        """
        Infer the bucket level from the current memory scope ID.

        This inspects the prefix of `scope_id()` values such as
        `session:...` or `user:...`.

        Examples:
            Infer the bucket level:
            ```python
            bucket = context.memory().bucket_level()
            ```

            Handle unknown/global buckets:
            ```python
            bucket = context.memory().bucket_level() or "unknown"
            ```

        Args:
            None.

        Returns:
            str | None: Parsed bucket prefix (`"session"`, `"user"`, `"run"`, `"org"`, `"app"`) or None.
        """
        scope_id = self.scope_id()
        if not scope_id:
            return None
        # memory_scope_id is of form "<prefix>:<value>" for scoped cases
        if ":" in scope_id:
            return scope_id.split(":", 1)[0]
        # could be "global" or some custom key
        return None

    def timeline(self: MemoryFacadeProtocol) -> str | None:
        """
        Return the timeline ID used by this memory facade.

        This value is the primary partition key used when appending and
        reading events from hotlog and persistence.

        Examples:
            Read the timeline ID:
            ```python
            timeline_id = context.memory().timeline()
            ```

            Fallback to a placeholder:
            ```python
            timeline_id = context.memory().timeline() or "<none>"
            ```

        Args:
            None.

        Returns:
            str | None: The timeline identifier, or None if not initialized.
        """
        return getattr(self, "timeline_id", None)

    # ----- Structured info -----

    def scope_info(self: MemoryFacadeProtocol) -> dict[str, Any]:
        """
        Return a structured snapshot of current memory scope metadata.

        The returned dictionary is intended for diagnostics and observability.

        Examples:
            Retrieve structured scope information:
            ```python
            info = context.memory().scope_info()
            ```

            Access timeline and level fields:
            ```python
            info = context.memory().scope_info()
            timeline = info.get("timeline_id")
            level = info.get("memory_level")
            ```

        Args:
            None.

        Returns:
            dict[str, Any]: Scope and runtime identifiers (timeline, memory scope,
            level, and available scope attributes such as run/session/user/org IDs).
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

    def debug_print_scope(self: MemoryFacadeProtocol, prefix: str = "[MEM]") -> None:
        """
        Print formatted scope diagnostics to stdout.

        This is a convenience helper for scripts and tests where a quick text
        view of memory scope fields is useful.

        Examples:
            Print with the default prefix:
            ```python
            context.memory().debug_print_scope()
            ```

            Print with a custom prefix:
            ```python
            context.memory().debug_print_scope(prefix="[DEBUG-MEM]")
            ```

        Args:
            prefix: Prefix string prepended to each printed line.

        Returns:
            None: This method prints diagnostics and does not return data.
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
