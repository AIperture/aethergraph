# Typed exceptions (ValidationError, MigrationError, etc.)


class AetherGraphError(Exception):
    """Base class for all AetherGraph errors."""


class NodeContractError(AetherGraphError):
    """Raised when a TaskNodeRuntime violates its declared input/output contract."""


class MissingInputError(NodeContractError):
    """Raised when a required input key is missing."""


class MissingOutputError(NodeContractError):
    """Raised when a required output key is missing."""


class ExecutionError(AetherGraphError):
    """Raised when a node’s logic fails during execution."""


class GraphHasPendingWaits(RuntimeError):
    """Raised when attempting to finalize a graph that has pending waits."""

    def __init__(
        self, message: str, waiting_nodes: list[str], continuations: list[dict] | None = None
    ):
        super().__init__(message)
        self.waiting_nodes = waiting_nodes
        self.continuations = continuations or []


class ResumeIncompatibleSnapshot(RuntimeError):
    """
    Raised when a snapshot is not allowed for resume under the current policy
    (e.g., contains non-JSON outputs or external refs like __aether_ref__).
    """

    def __init__(self, run_id: str, reasons: list[str]):
        super().__init__(f"Resume blocked for run_id={run_id}. " + " / ".join(reasons))
        self.run_id = run_id
        self.reasons = reasons
