from functools import lru_cache
from .loader import load_settings
from .config import AppSettings

@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return load_settings()
