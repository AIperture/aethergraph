from pydantic import BaseModel, Field  # type: ignore


class GraphListItem(BaseModel):
    graph_id: str
    name: str
    description: str | None = None
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    kind: str | None = None
    flow_id: str | None = None
    entrypoint: bool | None = None


class GraphNodeInfo(BaseModel):
    id: str
    type: str | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    expected_inputs: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    output_keys: list[str] = Field(default_factory=list)


class GraphEdgeInfo(BaseModel):
    source: str
    target: str


class GraphDetail(BaseModel):
    graph_id: str
    name: str
    description: str | None = None
    inputs: list[str]
    outputs: list[str]
    tags: list[str] = Field(default_factory=list)
    kind: str | None = None
    flow_id: str | None = None
    entrypoint: bool | None = None
    nodes: list[GraphNodeInfo] = Field(default_factory=list)
    edges: list[GraphEdgeInfo] = Field(default_factory=list)
