# ParamSpec, IOSpec, IOBindings, validators

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from .graph_refs import normalize_binding


@dataclass
class ParamSpec:
    """ParamSpec defines a single parameter's specification."""

    schema: dict[str, Any] = field(default_factory=dict)  # JSON schema or empty
    default: Any = None  # default value or None
    source: Literal["arg", "ctx", "memory", "env", "secret", "kv"] | None = (
        None  # where to bind from
    )
    doc: str | None = None  # optional description or docstring


@dataclass
class IOSpec:
    required: dict[str, "ParamSpec"] = field(default_factory=dict)
    optional: dict[str, "ParamSpec"] = field(default_factory=dict)
    outputs: dict[str, "ParamSpec"] = field(default_factory=dict)

    # Existing field (keep for back-compat)
    expose: list[str] = field(default_factory=list)

    # NEW: canonical bindings for exposed outputs (name -> Ref|literal)
    expose_bindings: dict[str, Any] = field(default_factory=dict)

    # ---- Convenience API (non-breaking) ----
    def set_expose(self, name: str, binding: Any) -> None:
        """Canonical way to record a public output and its binding."""
        if name not in self.expose:
            self.expose.append(name)
        self.expose_bindings[name] = normalize_binding(binding)

    def get_expose_names(self) -> list[str]:
        # Use dict keys if present; otherwise fall back to list
        if self.expose_bindings:
            # ensure order is stable: preserve original list order if possible
            ordered = [n for n in self.expose if n in self.expose_bindings]
            # include any names defined only in bindings (edge cases)
            ordered += [n for n in self.expose_bindings if n not in ordered]
            return ordered
        return list(self.expose)

    def get_expose_bindings(self) -> dict[str, Any]:
        # If only a list exists (legacy), return empty; caller can use heuristics if desired
        return dict(self.expose_bindings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IOBindings:
    """IO bindings are used to bind actual values to the inputs/outputs defined in IOSpec."""

    inbound: dict[str, str] = field(
        default_factory=dict
    )  # name -> source (arg, ctx, memory, env, secret, kv)
    outbound: dict[str, str] = field(
        default_factory=dict
    )  # name -> destination (ctx, memory, kv, output)
