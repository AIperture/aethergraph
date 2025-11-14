from typing import Protocol


class Secrets(Protocol):
    async def get(self, name: str) -> str | None:
        """Retrieve the secret value by its name. Returns None if not found."""
        ...
