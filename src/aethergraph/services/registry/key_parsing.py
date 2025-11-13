from .registry_key import Key, _REG_PREFIX, NS 

def parse_ref(ref: str) -> Key:
    """
    Parse "<nspace>:<name>[@<version>]" or "<name>[@<version>]" (defaults to tool).
    Also accepts "registry:<...>" prefix.

    Examples:
      "tool:my_tool@0.1.0"
      "graph:my_graph"
      "agent:router@latest"
      "my_tool@0.1.0"           # -> tool
      "registry:tool:my_tool@0.1.0"
      "registry:my_tool@0.1.0"  # -> tool
    """
    if not ref:
        raise ValueError("Empty ref")
    m = _REG_PREFIX.match(ref)
    s = m.group(1) if m else ref

    # If a namespace is present, it looks like "ns:name..."
    if ":" in s:
        nspace, rest = s.split(":", 1)
        if nspace not in NS:
            # If the left side is not a namespace, treat whole thing as name (default to tool)
            nspace, rest = "tool", s
    else:
        nspace, rest = "tool", s  # default namespace

    if "@" in rest:
        name, ver = rest.split("@", 1)
        ver = ver or None
    else:
        name, ver = rest, None

    # normalize @latest → None (caller treats None as "pick latest")
    if ver and ver.lower() == "latest":
        ver = None

    if not name:
        raise ValueError(f"Invalid ref (missing name): {ref}")

    return Key(nspace=nspace, name=name, version=ver)