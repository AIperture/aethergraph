"""Immutable safe diagnostics for canonical storage startup failures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

StorageStartupStage = Literal[
    "bundle_validation",
    "health_check",
]


@dataclass(frozen=True, slots=True)
class StorageStartupDiagnostic:
    """Preserve one primary storage startup failure and optional cleanup failure."""

    diagnostic_id: str
    workspace_root: Path
    provider_name: str
    stage: StorageStartupStage
    exception_type: str
    message: str
    cleanup_exception_type: str | None = None
    cleanup_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Project the immutable diagnostic into a detached safe mapping.

        Intro:
            The mapping contains provider, data-root, stage, and bounded exception
            identity without serializing provider configuration or credentials.

        Examples:
            Publish a readiness diagnostic:
                ```python
                payload = diagnostic.to_dict()
                ```

            Preserve the stable identifier across retries:
                ```python
                assert diagnostic.to_dict()["diagnostic_id"] == diagnostic.diagnostic_id
                ```

        Args:
            None.

        Returns:
            dict[str, Any]: Detached credential-free diagnostic values.

        Notes:
            `workspace_root` is the exact authorized storage data-root identity.
        """

        return {
            "diagnostic_id": self.diagnostic_id,
            "workspace_root": str(self.workspace_root),
            "provider_name": self.provider_name,
            "stage": self.stage,
            "exception_type": self.exception_type,
            "message": self.message,
            "cleanup_exception_type": self.cleanup_exception_type,
            "cleanup_message": self.cleanup_message,
        }


__all__ = ["StorageStartupDiagnostic", "StorageStartupStage"]
