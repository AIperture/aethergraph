# aethergraph/config.py
from typing import Literal

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .llm import EmbeddingSettings, ImageGenerationSettings, LLMSettings
from .observability import ObservabilitySettings
from .search import SearchBackendSettings
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


class EmbeddingUsageQuotaSettings(BaseModel):
    """Configure optional per-run embedding-operation quotas.

    Exact call and text counts are reserved before provider dispatch. Token
    totals are reconciled only when the provider returns a usable receipt.
    Every limit is disabled when its value is `None`.
    """

    max_calls_per_run: int | None = Field(default=None, ge=0)
    max_texts_per_run: int | None = Field(default=None, ge=0)
    max_input_tokens_per_run: int | None = Field(default=None, ge=0)


class ImageGenerationUsageQuotaSettings(BaseModel):
    """Configure optional per-run image-generation quotas.

    Exact call and requested-image counts are reserved before provider
    dispatch. Token totals are reconciled only from provider usage receipts.
    Every limit is disabled when its value is `None`.
    """

    max_calls_per_run: int | None = Field(default=None, ge=0)
    max_images_per_run: int | None = Field(default=None, ge=0)
    max_input_tokens_per_run: int | None = Field(default=None, ge=0)
    max_output_tokens_per_run: int | None = Field(default=None, ge=0)
    max_total_tokens_per_run: int | None = Field(default=None, ge=0)


class ModelOperationUsageQuotaSettings(BaseModel):
    """Group non-Chat model-operation quotas under one host policy boundary."""

    embedding: EmbeddingUsageQuotaSettings = Field(default_factory=EmbeddingUsageQuotaSettings)
    image_generation: ImageGenerationUsageQuotaSettings = Field(
        default_factory=ImageGenerationUsageQuotaSettings
    )


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
    enabled: bool = Field(default=False)
    bot_token: SecretStr | None = None  # xoxb-...
    app_token: SecretStr | None = None  # xapp-... (Socket Mode)


class TelegramSettings(BaseModel):
    integration_id: str | None = None
    enabled: bool = Field(default=False)
    bot_token: SecretStr | None = None


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
    model_operation_usage_quota: ModelOperationUsageQuotaSettings = Field(
        default_factory=ModelOperationUsageQuotaSettings
    )
    logging: LoggingSettings = LoggingSettings()
    slack: SlackSettings = SlackSettings()
    telegram: TelegramSettings = TelegramSettings()
    llm: LLMSettings = LLMSettings()
    observability: ObservabilitySettings = ObservabilitySettings()
    embed: EmbeddingSettings = EmbeddingSettings()
    image_generation: ImageGenerationSettings = ImageGenerationSettings()
    cont: ContinuationStoreSettings = ContinuationStoreSettings()
    memory: MemorySettings = MemorySettings()
    channels: ChannelSettings = ChannelSettings()
    auth: AuthSettings = AuthSettings()
    storage: StorageSettings = StorageSettings()
    search: SearchBackendSettings = SearchBackendSettings()

    # Optional path to demo-service directory (for admin routes).
    # Set via env: AETHERGRAPH_DEMO_SERVICE_DIR=/path/to/demo-service
    demo_service_dir: str | None = None

    # Future fields:
    # authn: ...
    # authz: ...
    # tracer: ...
