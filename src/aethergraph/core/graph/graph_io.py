# ParamSpec, IOSpec, IOBindings, validators

from dataclasses import dataclass, field, asdict
from typing import Literal, Optional, Dict, Any, List
from .graph_refs import normalize_binding

@dataclass
class ParamSpec:
    """ParamSpec defines a single parameter's specification."""
    schema: Dict[str, Any] = field(default_factory=dict)  # JSON schema or empty
    default: Any = None  # default value or None
    source: Optional[Literal["arg", "ctx", "memory", "env", "secret", "kv"]] = None  # where to bind from
    doc: Optional[str] = None  # optional description or docstring


@dataclass
class IOSpec:
    required: Dict[str, "ParamSpec"] = field(default_factory=dict)
    optional: Dict[str, "ParamSpec"] = field(default_factory=dict)
    outputs: Dict[str, "ParamSpec"] = field(default_factory=dict)

    # Existing field (keep for back-compat)
    expose: List[str] = field(default_factory=list)

    # NEW: canonical bindings for exposed outputs (name -> Ref|literal)
    expose_bindings: Dict[str, Any] = field(default_factory=dict)


    # ---- Convenience API (non-breaking) ----
    def set_expose(self, name: str, binding: Any) -> None:
        """Canonical way to record a public output and its binding."""
        if name not in self.expose:
            self.expose.append(name)
        self.expose_bindings[name] = normalize_binding(binding)

    def get_expose_names(self) -> List[str]:
        # Use dict keys if present; otherwise fall back to list
        if self.expose_bindings:
            # ensure order is stable: preserve original list order if possible
            ordered = [n for n in self.expose if n in self.expose_bindings]
            # include any names defined only in bindings (edge cases)
            ordered += [n for n in self.expose_bindings.keys() if n not in ordered]
            return ordered
        return list(self.expose)

    def get_expose_bindings(self) -> Dict[str, Any]:
        # If only a list exists (legacy), return empty; caller can use heuristics if desired
        return dict(self.expose_bindings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class IOBindings:
    """IO bindings are used to bind actual values to the inputs/outputs defined in IOSpec."""
    inbound: Dict[str, str] = field(default_factory=dict)  # name -> source (arg, ctx, memory, env, secret, kv)
    outbound: Dict[str, str] = field(default_factory=dict) # name -> destination (ctx, memory, kv, output)