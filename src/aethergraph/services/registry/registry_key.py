from dataclasses import dataclass
import re

NS = {"tool", "graph", "graphfn", "agent", "app"}

# Simple ref regex to detect optional leading 'registry:'
_REG_PREFIX = re.compile(r"^registry:(.+)$", re.I)


@dataclass(frozen=True)
class Key:
    nspace: str
    name: str
    version: str | None = None  # None or "latest" means resolve latest

    def canonical(self) -> str:
        ver = self.version
        # Normalize "latest" to omitted for display
        return f"{self.nspace}:{self.name}" + (f"@{ver}" if ver and ver.lower() != "latest" else "")
