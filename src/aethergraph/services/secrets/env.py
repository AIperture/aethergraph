import os

from .base import Secrets


class EnvSecrets(Secrets):
    def get(self, name: str) -> str | None:
        return os.getenv(name)
