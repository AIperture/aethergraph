# aethergraph/config.py
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm import EmbeddingSettings, LLMSettings
from .observability import ObservabilitySettings
from .search import KnowledgeSettings, SearchBackendSettings
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


class LLMUsageQuotaSettings(BaseSettings):
    """Configure optional infrastructure-owned LLM quotas.

    Quotas are disabled when their value is ``None``. Agent-loop budgets belong
    to the agent engine and are intentionally not given implicit AG defaults.
    """

    max_calls_per_run: int | None = Field(default=None, ge=0)
    max_input_tokens_per_run: int | None = Field(default=None, ge=0)
    max_output_tokens_per_run: int | None = Field(default=None, ge=0)
    max_total_tokens_per_run: int | None = Field(default=None, ge=0)


class LoggingSettings(BaseModel):
    nspace: str = Field("aethergraph", description="Root logger namespace")
    level: str = Field("INFO", description="Root log level")
    console_level: str | None = Field(None, description="Console log level")
    file_level: str | None = Field(
        None,
        description="Optional rotating-file log level; structured observation persistence is default.",
    )
    json_logs: bool = Field(False, description="Emit JSON logs")
    enable_queue: bool = Field(default=False, description="Enable async logging via queue")

    external_level: str = Field("WARNING", description="Level for third-party loggers")
    quiet_loggers: list[str] = Field(
        default_factory=lambda: ["httpx", "faiss", "faiss.loader", "slack_sdk"],
        description="Additional loggers to set to external_level",
    )


class SlackSettings(BaseModel):
    integration_id: str | None = None
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


class TelegramSettings(BaseModel):
    integration_id: str | None = None
    enabled: bool = Field(default=False)
    bot_token: SecretStr | None = None

    # for webhook mode
    webhook_enabled: bool = False
    webhook_secret: SecretStr | None = None  # used ONLY for HTTP webhook verification

    # for local / dev mode
    polling_enabled: bool = True  # use getUpdates loop by default for local


class ContinuationStoreSettings(BaseModel):
    kind: Literal["fs", "inmem"] = "fs"
    secret: SecretStr | None = None
    root: str = "./artifacts/continuations"


class MemorySettings(BaseModel):
    hot_limit: int = 1000
    hot_ttl_s: int = 7 * 24 * 3600
    signal_threshold: float = 0.25


class ChannelSettings(BaseModel):
    """Reserved host-level Channel configuration namespace."""


class RAGSettings(BaseModel):
    root: str = (
        "./aethergraph_workspace/rag"  # base dir for rag; should not use it unless customized
    )
    backend: str = "sqlite"  # "sqlite" | "faiss"
    index_path: str | None = None  # defaults set at runtime if None
    dim: int | None = None  # only for faiss; optional


class AuthSettings(BaseModel):
    cookie_name: str = "ag_auth_session"
    cookie_secure: bool = False
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    session_ttl_seconds: int = 24 * 3600
    grant_ttl_seconds: int = 7 * 24 * 3600
    public_demo_fallback_enabled: bool = True
    secret: SecretStr | None = None
    admin_api_key: SecretStr | None = None  # env: AETHERGRAPH_AUTH__ADMIN_API_KEY


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AETHERGRAPH_", env_nested_delimiter="__", extra="ignore", case_sensitive=False
    )

    # top-level workspace root directory
    workspace: str = "./aethergraph_workspace"

    # Browser origins allowed by CORS for the HTTP API. Covers the local dev
    # UIs, the AG Studio UI (both 127.0.0.1 and localhost forms — browsers treat
    # them as distinct origins), and the file:// admin page ("null"). Override
    # via env with a JSON list, e.g.:
    #   AETHERGRAPH_CORS_ALLOW_ORIGINS='["http://127.0.0.1:4186","http://localhost:4186"]'
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",  # dev UI
            "http://localhost:5185",  # sim UI
            "http://127.0.0.1:4186",  # AG Studio UI (127.0.0.1 form)
            "http://localhost:4186",  # AG Studio UI (localhost form)
            "null",  # file:// admin page
        ],
        description="Browser origins allowed by CORS for the HTTP API.",
    )

    # Deployment mode controls identity resolution and tenant scoping.
    #
    #   "local" (default / OSS):
    #       All requests resolve to a single local identity. No tenant
    #       isolation. CLI, scripts, and the UI all share the same view
    #       of runs, apps, and artifacts. X-Client-ID is recorded but
    #       does not create separate scopes.
    #
    #   "demo":
    #       Multi-tenant demo. Each browser gets isolated via X-Client-ID
    #       (user_id="demo:<client_id>"). Rate limits are enforced.
    #       Useful for hosted public demos where multiple users share
    #       one server but should not see each other's runs.
    #
    #   "cloud":
    #       Production. Expects an auth gateway to inject X-User-ID and
    #       X-Org-ID headers. Full tenant isolation and RBAC.
    #
    # Set via env: AETHERGRAPH_DEPLOY_MODE=demo
    deploy_mode: Literal["local", "demo", "cloud"] = "local"

    rate_limit: RateLimitSettings = RateLimitSettings()
    llm_usage_quota: LLMUsageQuotaSettings = LLMUsageQuotaSettings()
    logging: LoggingSettings = LoggingSettings()
    slack: SlackSettings = SlackSettings()
    telegram: TelegramSettings = TelegramSettings()
    llm: LLMSettings = LLMSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    embed: EmbeddingSettings = EmbeddingSettings()
    cont: ContinuationStoreSettings = ContinuationStoreSettings()
    memory: MemorySettings = MemorySettings()
    channels: ChannelSettings = ChannelSettings()
    rag: RAGSettings = RAGSettings()
    auth: AuthSettings = AuthSettings()
    storage: StorageSettings = StorageSettings()
    search: SearchBackendSettings = SearchBackendSettings()
    knowledge: KnowledgeSettings = KnowledgeSettings()

    # Optional path to demo-service directory (for admin routes).
    # Set via env: AETHERGRAPH_DEMO_SERVICE_DIR=/path/to/demo-service
    demo_service_dir: str | None = None

    # Future fields:
    # authn: ...
    # authz: ...
    # tracer: ...
