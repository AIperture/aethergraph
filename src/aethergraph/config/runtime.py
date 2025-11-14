from functools import lru_cache

from .config import AppSettings
from .loader import load_settings


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return load_settings()
