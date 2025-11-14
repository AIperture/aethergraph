# services/metering/noop.py
class NoopMetering:
    async def incr(self, metric: str, value: float = 1.0, **tags):
        return None
