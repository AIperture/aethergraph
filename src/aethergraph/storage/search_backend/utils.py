import re

_DURATION_PATTERN = re.compile(r"^\s*(\d+)\s*([smhd])\s*$")


def _parse_time_window(window: str) -> float:
    """
    Parse a simple duration string like:
      - "30s" (seconds)
      - "15m" (minutes)
      - "2h"  (hours)
      - "7d"  (days)

    Returns duration in seconds.
    Raises ValueError on invalid format.
    """
    m = _DURATION_PATTERN.match(window)
    if not m:
        raise ValueError(f"Invalid time_window format: {window!r}")
    value = int(m.group(1))
    unit = m.group(2)

    if unit == "s":
        return float(value)
    if unit == "m":
        return float(value) * 60.0
    if unit == "h":
        return float(value) * 3600.0
    if unit == "d":
        return float(value) * 86400.0
    raise ValueError(f"Unknown time unit in time_window: {window!r}")
