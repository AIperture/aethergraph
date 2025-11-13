# services/tracing/noop.py
import contextlib, time
class NoopTracer:
    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        t = time.time(); yield; dt = (time.time()-t)*1000
        # optionally log duration
