from __future__ import annotations

from pydantic import BaseModel, Field


class ObservationRetentionSettings(BaseModel):
    max_age_days: int = Field(default=30, ge=1)
    error_max_age_days: int = Field(default=90, ge=1)
    max_full_prompt_age_days: int = Field(default=3, ge=1)
    max_bytes_per_trace: int = Field(default=64 * 1024 * 1024, ge=1)
    max_total_bytes: int = Field(default=512 * 1024 * 1024, ge=1)
    max_retained_traces: int = Field(default=10_000, ge=1)
    max_retained_runs: int = Field(default=10_000, ge=1)
    max_observations_per_purge: int = Field(default=1_000, ge=1)
    janitor_interval_seconds: int = Field(default=3_600, ge=60)


class ObservabilitySettings(BaseModel):
    persist_logs: bool = True
    retention: ObservationRetentionSettings = ObservationRetentionSettings()
