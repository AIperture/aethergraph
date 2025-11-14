from __future__ import annotations

from dataclasses import dataclass

_STATUS_COLOR = {
    "PENDING": "#d3d3d3",
    "RUNNING": "#ffd966",
    "DONE": "#b6d7a8",
    "FAILED": "#e06666",
    "FAILED_TIMEOUT": "#e69138",
    "SKIPPED": "#cccccc",
    "WAITING_HUMAN": "#9fc5e8",
    "WAITING_ROBOT": "#9fc5e8",
    "WAITING_EXTERNAL": "#9fc5e8",
    "WAITING_TIME": "#9fc5e8",
    "WAITING_EVENT": "#9fc5e8",
}

_NODE_SHAPE = {
    "tool": "box",
    "llm": "component",
    "human": "oval",
    "robot": "hexagon",
    "custom": "box3d",
}


def _safe_get(d, key, default=None):
    try:
        return d.get(key, default)
    except Exception:
        return default


def _escape(s: str) -> str:
    return str(s).replace('"', r"\"")


def _fmt_multiline(*lines):
    return "\\n".join(str(x) for x in lines if x is not None and str(x) != "")


@dataclass
class _VizConfig:
    rankdir: str = "LR"
    fontname: str = "Inter,Helvetica,Arial,sans-serif"
    fontsize: int = 11
    show_io_brief: bool = True
    show_logic: bool = True
    dashed_control_deps: bool = True


# === Add these methods onto TaskGraph ========================================


def to_dot(self, cfg: _VizConfig | None = None) -> str:
    """
    Build a Graphviz DOT string of the graph with status-aware coloring and
    dashed control-deps (if present in node.metadata['control_deps']).
    """
    cfg = cfg or _VizConfig()
    lines = []
    L = lines.append

    L("digraph TaskGraph {")
    L(f"  rankdir={cfg.rankdir};")
    L(f'  graph [fontname="{cfg.fontname}"];')
    L(
        f'  node  [fontname="{cfg.fontname}", fontsize={cfg.fontsize}, style="rounded,filled", shape=box];'
    )
    L(f'  edge  [fontname="{cfg.fontname}", fontsize={cfg.fontsize}];')

    # (optional) graph IO summary as a note
    try:
        io_lines = self.spec.io_summary_lines() if cfg.show_io_brief else []
    except Exception:
        io_lines = []
    if io_lines:
        label = _escape(_fmt_multiline("IO", *io_lines))
        L('  subgraph cluster_io { style=dashed; color="#cccccc"; label="";')
        L(
            f'    io_summary [shape=note, style="rounded,filled", fillcolor="#f7f7f7", label="{label}"];'
        )
        L("  }")

    # nodes
    for node_id, node in self.spec.nodes.items():
        # Node attributes
        status = _safe_get(self.state.node_status, node_id, "PENDING") or "PENDING"
        fill = _STATUS_COLOR.get(status, "#d3d3d3")
        ntype = getattr(node, "node_type", None)
        shape = _NODE_SHAPE.get(str(ntype).lower() if ntype else "", "box")

        # Label: id + type/status (+ logic)
        logic = getattr(node, "logic", None)
        logic_s = None
        if cfg.show_logic and logic is not None:
            logic_s = (
                logic
                if isinstance(logic, str)
                else getattr(logic, "__name__", type(logic).__name__)
            )
            # shorten registry path a bit
            if isinstance(logic_s, str) and logic_s.startswith("registry:"):
                logic_s = logic_s.replace("registry:", "")

        label = _escape(
            _fmt_multiline(
                f"{node_id}",
                f"[{str(ntype).lower()}]" if ntype else None,
                f"status: {status}",
                logic_s,
            )
        )

        L(f'  "{_escape(node_id)}" [shape={shape}, fillcolor="{fill}", label="{label}"];')

    # edges (data/control)
    for node_id, node in self.spec.nodes.items():
        deps = getattr(node, "dependencies", []) or []
        ctrl = set(_safe_get(getattr(node, "metadata", {}) or {}, "control_deps", []) or [])

        for dep in deps:
            style = "dashed" if (cfg.dashed_control_deps and dep in ctrl) else "solid"
            color = "#999999" if style == "dashed" else "#555555"
            L(f'  "{_escape(dep)}" -> "{_escape(node_id)}" [style={style}, color="{color}"];')

    # (optional) Legend
    L("  subgraph cluster_legend {")
    L('    label="Legend"; style=dashed; color="#cccccc";')
    L("    key_status [shape=plaintext, label=<")
    L('      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">')
    L('        <TR><TD COLSPAN="2"><B>Status Colors</B></TD></TR>')
    for k, v in _STATUS_COLOR.items():
        L(f'        <TR><TD>{k}</TD><TD BGCOLOR="{v}"> </TD></TR>')
    L("      </TABLE>")
    L("    >];")
    L("    key_edge [shape=plaintext, label=<")
    L('      <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="6">')
    L('        <TR><TD COLSPAN="2"><B>Edge Types</B></TD></TR>')
    L('        <TR><TD>data dependency</TD><TD><FONT COLOR="#555555">solid</FONT></TD></TR>')
    L('        <TR><TD>control dependency</TD><TD><FONT COLOR="#999999">dashed</FONT></TD></TR>')
    L("      </TABLE>")
    L("    >];")
    L("  }")

    L("}")
    return "\n".join(lines)


def visualize(
    self,
    outfile: str | None = None,
    fmt: str = "svg",
    view: bool = False,
    cfg: _VizConfig | None = None,
    return_dot: bool = False,
) -> str:
    """
    Render or export the graph.

    Args:
        outfile: Path prefix without extension (e.g., 'out/graph').
        fmt:     'svg' | 'png' | 'pdf' ...
        view:    If True, open the rendered file (if possible).
        cfg:     Visualization config (_VizConfig).
        return_dot:
            - If True, always return the raw DOT string so the user can paste
              it into webgraphviz or other tools.
            - If outfile is also provided, a .dot file will be exported to
              `outfile + ".dot"` in addition to any rendering.

    Returns:
        - DOT string if return_dot=True
        - Rendered file path if rendering was successful
        - DOT string fallback if rendering fails
    """
    dot = self.to_dot(cfg)

    dot_path = None
    if outfile and return_dot:
        dot_path = f"{outfile}.dot"
        with open(dot_path, "w", encoding="utf-8") as f:
            f.write(dot)
        import logging

        log = logging.getLogger("aethergraph.core.graph.visualize")
        log.info(
            f"DOT file written to: {dot_path}. View the graph at https://dreampuf.github.io/GraphvizOnline/"
        )

    if return_dot and (outfile is None):
        # Only return DOT (no rendering)
        return dot

    try:
        import graphviz

        src = graphviz.Source(dot, format=fmt)
        if outfile:
            path = src.render(
                outfile, view=view, cleanup=True
            )  # This will also create outfile but without .dot extension?
            # If return_dot, return DOT text (but still render the file)
            return dot if return_dot else path
        elif view:
            # View only, no outfile
            import tempfile

            tmp = tempfile.mkstemp(prefix="taskgraph_", suffix=f".{fmt}")[1]
            src.render(tmp[: -len(f".{fmt}")], view=True, cleanup=True)
            return dot if return_dot else tmp
    except Exception:
        import logging

        log = logging.getLogger("aethergraph.core.graph.visualize")
        log.warning("Graph rendering failed; returning DOT string instead.")
        pass

    return dot


def ascii_overview(self) -> str:
    """
    Minimal text view: one line per node with deps and status.
    """
    lines = []
    for node_id, node in self.spec.nodes.items():
        status = _safe_get(self.state.node_status, node_id, "PENDING") or "PENDING"
        deps = getattr(node, "dependencies", []) or []
        ntype = getattr(node, "node_type", "tool")
        logic = getattr(node, "logic", None)
        logic_s = (
            (logic if isinstance(logic, str) else getattr(logic, "__name__", type(logic).__name__))
            if logic
            else ""
        )
        lines.append(f"- {node_id} [{ntype} | {status}] deps={deps} logic={logic_s}")
    return "\n".join(lines)
