import re
from dataclasses import dataclass
from typing import Optional

NS = {"tool", "graph", "graphfn", "agent"}

# Simple ref regex to detect optional leading 'registry:'
_REG_PREFIX = re.compile(r"^registry:(.+)$", re.I)

@dataclass(frozen=True)
class Key:
    nspace: str
    name: str
    version: Optional[str] = None  # None or "latest" means resolve latest

    def canonical(self) -> str:
        ver = self.version
        # Normalize "latest" to omitted for display
        return f"{self.nspace}:{self.name}" + (f"@{ver}" if ver and ver.lower() != "latest" else "")