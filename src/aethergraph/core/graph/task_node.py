from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .node_spec import TaskNodeSpec
from .node_state import NodeStatus, TaskNodeState


@dataclass
class TaskNodeRuntime:
    spec: TaskNodeSpec
    state: TaskNodeState
    _parent_graph: Any  # back-reference to parent graph

    # ---- Spec pass-through ----
    @property
    def node_id(self) -> str:
        return self.spec.node_id

    @property
    def type(self) -> str:
        return self.spec.type

    @property
    def logic(self) -> Any:
        return self.spec.logic

    @property
    def dependencies(self) -> list[str]:
        return self.spec.dependencies

    @property
    def inputs(self) -> dict[str, Any]:
        return self.spec.inputs

    @property
    def expected_input_keys(self) -> list[str]:
        return self.spec.expected_input_keys

    @property
    def expected_output_keys(self) -> list[str]:
        return self.spec.output_keys

    @property
    def condition(self) -> Any:
        return self.spec.condition

    @property
    def metadata(self) -> dict[str, Any]:
        return self.spec.metadata

    @property
    def tool_name(self) -> str | None:
        return self.spec.tool_name

    @property
    def tool_version(self) -> str | None:
        return self.spec.tool_version

    # ---- State pass-through ----
    @property
    def status(self) -> NodeStatus:
        return self.state.status

    @property
    def outputs(self) -> dict[str, Any]:
        return self.state.outputs

    @property
    def output(self) -> Any:
        return self.state.output

    # --- Compat helpers ---
    def allow(self, reads: Iterable[str] | None, writes: Iterable[str] | None) -> "TaskNodeRuntime":
        """Return ad *new* spec via a patch rather than mutating in place."""
        patch = {"node_id": self.node_id}
        if reads:
            patch["reads_add"] = list(reads)
        if writes:
            patch["writes_add"] = list(writes)
        self._parent_graph.add_acl_patch(patch)
        return self
