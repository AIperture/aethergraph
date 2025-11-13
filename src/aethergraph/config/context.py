from contextvars import ContextVar
from .config import AppSettings

_current_cfg: ContextVar[AppSettings | None] = ContextVar("current_cfg", default=None)

def set_current_settings(cfg: AppSettings) -> None:
    _current_cfg.set(cfg)

def current_settings() -> AppSettings:
    cfg = _current_cfg.get()
    if cfg is None:
        raise RuntimeError("Settings not installed. Call set_current_settings() at startup.")
    return cfg
