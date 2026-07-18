from __future__ import annotations

from dataclasses import dataclass

from .models import CaptureMode


@dataclass(frozen=True)
class ObservationPolicy:
    capture_mode: CaptureMode = "metadata"
    renderer_version: str = "ag-provider-request-v1"
    max_attributes_bytes: int = 16 * 1024
    max_fragment_bytes: int = 8 * 1024 * 1024
    metadata_preview_bytes: int = 512
    max_summary_chars: int = 2_000
    max_error_chars: int = 4_000
    full_prompt_ttl_days: int = 3

    def validate(self) -> None:
        if self.max_attributes_bytes <= 0:
            raise ValueError("max_attributes_bytes must be positive")
        if self.max_fragment_bytes <= 0:
            raise ValueError("max_fragment_bytes must be positive")
        if self.metadata_preview_bytes < 0:
            raise ValueError("metadata_preview_bytes cannot be negative")
        if self.max_summary_chars <= 0:
            raise ValueError("max_summary_chars must be positive")
        if self.max_error_chars <= 0:
            raise ValueError("max_error_chars must be positive")
        if self.full_prompt_ttl_days <= 0:
            raise ValueError("full_prompt_ttl_days must be positive")
