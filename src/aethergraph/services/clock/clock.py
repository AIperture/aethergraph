from datetime import datetime, timezone

class SystemClock:
    """ System clock service. """
    @staticmethod
    def now() -> datetime:
        return datetime.now(timezone.utc)
    