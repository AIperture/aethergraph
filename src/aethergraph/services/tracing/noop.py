# services/tracing/noop.py
import contextlib
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aethergraph.tracing.noop")


class NoopTracer:
    @contextlib.contextmanager
    def span(self, name: str, **attrs):
        t = time.time()
        yield
        dt = (time.time() - t) * 1000  # milliseconds
        # optionally log duration
        logger.info(f"Span '{name}' took {dt} ms", **attrs)
        return
