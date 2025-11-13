from typing import Protocol, Optional

class Secrets(Protocol):
    async def get(self, name: str) -> Optional[str]:
        """Retrieve the secret value by its name. Returns None if not found."""
        ...