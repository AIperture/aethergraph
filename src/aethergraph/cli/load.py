from __future__ import annotations

import os
from pathlib import Path
import sys

from aethergraph.server.loading import GraphLoader, LoadSpec


def make_load_spec(args) -> LoadSpec:
    return LoadSpec(
        modules=list(args.load_module or []),
        paths=list(args.load_path or []),
        project_root=args.project_root,
        strict=bool(args.strict_load),
    )


def prepare_project_root(project_root: str) -> str:
    pr_str = str(Path(project_root).resolve())
    if pr_str not in sys.path:
        sys.path.insert(0, pr_str)
    return pr_str


def export_load_environment(
    *,
    workspace: str,
    project_root: str,
    modules: list[str],
    paths: list[str],
    strict_load: bool,
    log_level: str,
) -> None:
    os.environ["AETHERGRAPH_WORKSPACE"] = workspace
    os.environ["AETHERGRAPH_PROJECT_ROOT"] = str(project_root)
    os.environ["AETHERGRAPH_LOAD_MODULES"] = ",".join(modules)
    os.environ["AETHERGRAPH_LOAD_PATHS"] = os.pathsep.join(paths)
    os.environ["AETHERGRAPH_STRICT_LOAD"] = "1" if strict_load else "0"
    os.environ["AETHERGRAPH_LOG_LEVEL"] = log_level


def load_graphs(loader: GraphLoader, spec: LoadSpec):
    return loader.load(spec)
