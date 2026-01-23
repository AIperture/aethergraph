# aethergraph/contracts/services/execution.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

Language = Literal["python"]  # later: "bash", "r", etc.


@dataclass
class CodeExecutionRequest:
    """
    Generic request to execute code.

    For v0 we only support Python, but keep `language` so we can grow.
    """

    language: Language
    code: str

    # CLI-style args if you want to support them later
    args: list[str] = field(default_factory=list)

    # Execution constraints
    timeout_s: float = 30.0

    # Optional working directory and env (for future use)
    workdir: str | None = None
    env: dict[str, str] | None = None


@dataclass
class CodeExecutionResult:
    """
    Result of a code execution.

    - stdout / stderr: what the script printed
    - exit_code: OS-level exit code
    - error: high-level error if our runner failed (timeout, spawn failure, etc.)
    - metadata: free-form, e.g. timing info
    """

    stdout: str
    stderr: str
    exit_code: int
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionService(Protocol):
    """
    Abstract interface for code execution backends.

    This lets us later plug in Docker, VM, remote HTTP, etc,
    without changing NodeContext or node code.
    """

    async def execute(self, request: CodeExecutionRequest) -> CodeExecutionResult: ...
