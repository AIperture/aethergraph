from __future__ import annotations


def _strip_front_matter(text: str) -> str:
    """
    Strip YAML-style front matter if present:

    ---
    key: value
    ---
    Rest of doc...
    """
    if not text.startswith("---"):
        return text

    parts = text.split("---", 2)
    if len(parts) == 3:
        # parts = ["", "<front matter>\n", "<rest>"]
        return parts[2].lstrip()
    return text


def _strip_simple_rst_noise(text: str) -> str:
    """
    Remove some Sphinx-ish noise that isn't helpful for search:

    - Label definitions: ``.. _label-name:``
    - Navigation-only toctrees and images (we drop the directive line,
      but keep body text if any appears later).
    """
    lines = text.splitlines()
    cleaned: list[str] = []
    skip_indented_block = False

    for line in lines:
        stripped = line.lstrip()

        if skip_indented_block:
            # Continue skipping while the line is indented or blank.
            if stripped and not line.startswith(" " * 3) and not line.startswith("\t"):
                skip_indented_block = False
            else:
                continue  # still skipping this directive's block

        # Anchors: .. _some-label:
        if stripped.startswith(".. _") and stripped.endswith(":"):
            continue

        # Navigation / non-content directives we can safely drop
        if stripped.startswith(".. toctree::") or stripped.startswith(".. image::"):
            skip_indented_block = True
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def extract_text(path: str) -> tuple[str, dict]:
    """
    Extract text from an .rst file, with lightweight cleaning:
    - strip optional YAML front matter
    - strip some Sphinx anchor / toctree noise
    """
    with open(path, encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    txt = _strip_front_matter(txt)
    txt = _strip_simple_rst_noise(txt)
    return txt, {}
