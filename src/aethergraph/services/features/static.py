# services/features/static.py
class StaticFeatures:
    def __init__(self, flags: dict[str, bool]):
        self._f = set(k for k, v in flags.items() if v)

    def has(self, name: str) -> bool:
        return name in self._f

    def all(self) -> set[str]:
        return set(self._f)
