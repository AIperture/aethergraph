# src/aethergraph/utils/optdeps.py
def require(pkg: str, extra: str):
    try:
        __import__(pkg)
    except ImportError as e:
        raise RuntimeError(
            f"{pkg} is required for this feature. Install with: pip install 'aethergraph[{extra}]'"
        ) from e
