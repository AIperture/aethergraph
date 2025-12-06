# aethergraph/config.py
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm import LLMSettings
from .storage import StorageSettings


class RateLimitSettings(BaseSettings):
    enabled: bool = True

    # Concurrency
    max_concurrent_runs: int = 8

    # Per-identity, per-window run limits (using metering)
    runs_window: str = "1h"
    max_runs_per_window: int = 100

    # Short-burst, in-memory limiter for POST /runs
    burst_max_runs: int = 10
    burst_window_seconds: int = 10

    # Optional LLM caps *per run*
    max_llm_calls_per_run: int = 200
    max_llm_tokens_per_run: int = 200_000


class LoggingSettings(BaseModel):
    nspace: str = Field("aethergraph", description="Root logger namespace")
    level: str = Field("INFO", description="Root log level")
    json_logs: bool = Field(False, description="Emit JSON logs")
    enable_queue: bool = Field(default=False, description="Enable async logging via queue")

    external_level: str = Field("WARNING", description="Level for third-party loggers")
    quiet_loggers: list[str] = Field(
        default_factory=lambda: ["httpx", "faiss", "faiss.loader", "slack_sdk"],
        description="Additional loggers to set to external_level",
    )


class SlackSettings(BaseModel):
    # Turn Slack integration on/off globally
    enabled: bool = Field(default=False)

    # Tokens
    bot_token: SecretStr | None = None  # xoxb-...
    app_token: SecretStr | None = None  # xapp-... (Socket Mode)
    signing_secret: SecretStr | None = None  # only needed for HTTP/webhook

    # Transport mode flags
    #
    # Local / individual default:
    #   enabled = true
    #   socket_mode_enabled = true
    #   webhook_enabled = false
    #
    # Production / webhook default:
    #   enabled = true
    #   socket_mode_enabled = false (optional)
    #   webhook_enabled = true

    socket_mode_enabled: bool = Field(
        default=True, description="Use Slack Socket Mode (WS outbound) when app_token is set."
    )
    webhook_enabled: bool = Field(
        default=False,
        description="Expose /slack/events & /slack/interact HTTP endpoints for Slack.",
    )

    # Default routing
    #
    # For simple setups likely only need default_channel_id (+ maybe default_team_id).
    # default_channel_key is the more general 'slack:team/T:chan/C' form.
    # TODO: later we might deprecate the default setting in .env and require explicit channel keys in code.
    default_team_id: str | None = None  # e.g. 'T...'
    default_channel_id: str | None = None  # e.g. 'C...'
    default_channel_key: str | None = None  # e.g. 'slack:team/T...:chan/C...'


class TelegramSettings(BaseModel):
    enabled: bool = Field(default=False)
    bot_token: SecretStr | None = None

    # for webhook mode
    webhook_enabled: bool = False
    webhook_secret: SecretStr | None = None  # used ONLY for HTTP webhook verification

    # for local / dev mode
    polling_enabled: bool = True  # use getUpdates loop by default for local

    # default chat key
    default_chat_id: str | None = None


class ContinuationStoreSettings(BaseModel):
    kind: Literal["fs", "inmem"] = "fs"
    secret: SecretStr | None = None
    root: str = "./artifacts/continuations"


class MemorySettings(BaseModel):
    hot_limit: int = 1000
    hot_ttl_s: int = 7 * 24 * 3600
    signal_threshold: float = 0.25


class ChannelSettings(BaseModel):
    # room for Telegram / Console etc.
    default: str = "console:stdin"


class RAGSettings(BaseModel):
    root: str = "./aethergraph_data/rag"  # base dir for rag; should not use it unless customized
    backend: str = "sqlite"  # "sqlite" | "faiss"
    index_path: str | None = None  # defaults set at runtime if None
    dim: int | None = None  # only for faiss; optional


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHERGRAPH_", env_nested_delimiter="__", extra="ignore", case_sensitive=False
    )

    # top-level for workspace root
    root: str = "./aethergraph_data"

    rate_limit: RateLimitSettings = RateLimitSettings()
    logging: LoggingSettings = LoggingSettings()
    slack: SlackSettings = SlackSettings()
    telegram: TelegramSettings = TelegramSettings()
    llm: LLMSettings = LLMSettings()
    cont: ContinuationStoreSettings = ContinuationStoreSettings()
    memory: MemorySettings = MemorySettings()
    channels: ChannelSettings = ChannelSettings()
    rag: RAGSettings = RAGSettings()
    storage: StorageSettings = StorageSettings()

    # Future fields:
    # authn: ...
    # authz: ...
    # tracer: ...
