# services/redactor/simple.py
# PII/secret scrubbing for logs/events/artifacts
import re


class RegexRedactor:
    PATTERNS = [
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "[REDACTED:APIKEY]"),
        (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED:EMAIL]"),
        (re.compile(r"\b\d{16}\b"), "[REDACTED:NUM]"),
    ]

    def scrub(self, text: str) -> str:
        for pat, repl in self.PATTERNS:
            text = pat.sub(repl, text)
        return text
