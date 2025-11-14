from dataclasses import dataclass
from datetime import timedelta


@dataclass
class RetryPolicy:
    max_attempts: int = 0
    backoff_base: float = 2.0
    backoff_first: float = 1.0  # seconds
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def should_retry(self, attempt: int, error: BaseException) -> bool:
        """Determine if we should retry based on attempt count and error type."""
        if attempt >= self.max_attempts:
            return False
        return isinstance(error, self.retry_on)

    def backoff(self, attempt: int) -> float:
        """Calculate backoff time in seconds for the given attempt."""
        # attempt = 0 -> first failure
        delay = self.backoff_first * (self.backoff_base**attempt)
        return timedelta(seconds=delay)
