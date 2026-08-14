"""Typed failures exposed by the embedded runtime boundary."""


class EmbeddedRuntimeError(RuntimeError):
    """Base failure raised by the public embedded runtime boundary."""


class RuntimeNotReadyError(EmbeddedRuntimeError):
    """Raised when a required runtime capability was not constructed."""


class RuntimeGraphLoadError(EmbeddedRuntimeError):
    """Raised when an imported module does not register its declared graph."""


class RuntimeInteractionError(EmbeddedRuntimeError):
    """Stable public interaction-resolution failure."""

    def __init__(self, *, code: str, message: str) -> None:
        """Create a public interaction failure without leaking continuation data.

        Examples:
            Report a missing interaction:
            ```python
            error = RuntimeInteractionError(code="interaction_not_found", message="Missing")
            ```

            Preserve a stable code for a Host response:
            ```python
            assert RuntimeInteractionError(code="ambiguous", message="Many").code == "ambiguous"
            ```

        Args:
            code: Stable machine-readable interaction failure code.
            message: Human-readable failure explanation.

        Returns:
            None.

        Notes:
            Private continuation tokens are never included in this exception.
        """
        super().__init__(message)
        self.code = code


__all__ = [
    "EmbeddedRuntimeError",
    "RuntimeGraphLoadError",
    "RuntimeInteractionError",
    "RuntimeNotReadyError",
]
