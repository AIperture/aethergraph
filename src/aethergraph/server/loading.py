from __future__ import annotations

from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
import traceback
from typing import Any


@dataclass
class LoadSpec:
    modules: list[str] = field(default_factory=list)  # ["my_pkg.graphs"]
    paths: list[Path] = field(default_factory=list)  # [Path("./my_graphs.py")]
    project_root: Path | None = None  # for sys.path injection
    strict: bool = True  # raise on first error


@dataclass
class LoadError:
    source: str  # module or path
    error: str  # error message
    traceback: str | None = None  # optional traceback


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)  # successfully loaded modules/paths
    errors: list[LoadError] = field(default_factory=list)  # errors encountered during loading
    meta: dict[str, Any] = field(default_factory=dict)  # additional metadata


@contextmanager
def _temp_sys_path(root: Path | None):
    if not root:
        yield
        return
    if isinstance(root, str):
        root = Path(root)
    root_str = str(root.resolve())
    already = root_str in sys.path
    if not already:
        sys.path.insert(0, root_str)
    try:
        yield
    finally:
        if not already:
            # remove first occurrence in case user also added it
            with suppress(ValueError):
                sys.path.remove(root_str)


def _stable_module_name_for_path(path: Path) -> str:
    # stable across runs for the same absolute path
    h = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return f"aethergraph_userfile_{h}"


class GraphLoader:
    def __init__(self):
        self.last_report: LoadReport | None = None

    def load(self, spec: LoadSpec) -> LoadReport:
        report = LoadReport()
        with _temp_sys_path(spec.project_root):
            # 1) import modules
            for mod in spec.modules:
                try:
                    importlib.import_module(mod)
                    report.loaded.append(f"module:{mod}")
                except Exception as e:
                    report.errors.append(
                        LoadError(
                            source=f"module:{mod}",
                            error=repr(e),
                            traceback=traceback.format_exc(),
                        )
                    )
                    if spec.strict:
                        self.last_report = report
                        raise

            # 2) import paths
            for p in spec.paths:
                try:
                    if isinstance(p, str):
                        p = Path(p)
                    path = p.resolve()
                    name = _stable_module_name_for_path(path)
                    # Re-import strategy: if already imported, do nothing (Phase 1 design)
                    if name in sys.modules:
                        report.loaded.append(f"path:{path} (cached)")
                        continue

                    spec_obj = importlib.util.spec_from_file_location(name, str(path))
                    if spec_obj is None or spec_obj.loader is None:
                        raise ImportError(f"Cannot load spec for path: {path}")
                    module = importlib.util.module_from_spec(spec_obj)
                    sys.modules[name] = module
                    spec_obj.loader.exec_module(module)  # decorators @graphify etc. run here
                    report.loaded.append(f"path:{path}")
                except Exception as e:
                    report.errors.append(
                        LoadError(
                            source=f"path:{p}",
                            error=repr(e),
                            traceback=traceback.format_exc(),
                        )
                    )
                    if spec.strict:
                        self.last_report = report
                        raise
        self.last_report = report
        return report
