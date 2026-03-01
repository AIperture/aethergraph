from __future__ import annotations

import re
from typing import Literal

SplitMode = Literal["plain", "markdown", "rst", "code"]


class TextSplitter:
    """
    A slightly smarter text splitter that supports different modes:

    - plain: word-based splitting (backwards compatible with old behavior).
    - markdown/rst: split on paragraph boundaries; keep headings with their sections.
    - code: split on lines; try to keep functions / blocks together.

    target_tokens and overlap_tokens are still approximate word-based sizes.
    """

    def __init__(
        self,
        target_tokens: int = 400,
        overlap_tokens: int = 60,
        mode: SplitMode = "plain",
    ):
        self.n = max(50, target_tokens)
        self.o = max(0, min(self.n - 1, overlap_tokens))
        self.mode: SplitMode = mode

    # Public API ---------------------------------------------------------

    def split(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        if self.mode in ("markdown", "rst"):
            return self._split_paragraphs(text)
        elif self.mode == "code":
            return self._split_code(text)
        else:
            # "plain" (backwards compatible)
            return self._split_words(text)

    # Internal helpers ---------------------------------------------------

    def _split_words(self, text: str) -> list[str]:
        """
        Old behavior: word-based chunks with overlap.
        """
        words = text.split()
        if not words:
            return []
        step = self.n - self.o
        chunks: list[str] = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.n])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Paragraph/section-based splitting for markdown and rst-like docs.

        - Split on blank lines.
        - Keep headings attached to their content.
        - Apply approximate token budgeting per chunk with paragraph overlap.
        """
        # Normalize newlines and collapse big blank runs
        norm = re.sub(r"\r\n?", "\n", text)
        # Split into blocks separated by >= 1 blank line
        raw_blocks = re.split(r"\n\s*\n", norm)
        blocks = [b.strip() for b in raw_blocks if b.strip()]
        if not blocks:
            return []

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        def count_tokens(s: str) -> int:
            # Approximate by words; good enough for our budget.
            return max(1, len(s.split()))

        for block in blocks:
            block_tokens = count_tokens(block)

            # If adding this block would overflow, close current chunk.
            if current and current_tokens + block_tokens > self.n:
                chunks.append("\n\n".join(current))

                # Overlap: keep the last block of previous chunk (if small)
                overlap_block = current[-1]
                overlap_tokens = count_tokens(overlap_block)

                if self.o > 0 and overlap_tokens < self.o:
                    current = [overlap_block, block]
                    current_tokens = overlap_tokens + block_tokens
                else:
                    current = [block]
                    current_tokens = block_tokens
            else:
                current.append(block)
                current_tokens += block_tokens

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_code(self, text: str) -> list[str]:
        """
        Line-based splitting that tries to keep functions/classes together.
        """
        lines = text.splitlines()
        if not lines:
            return []

        # Very rough mapping: assume ~4 tokens per line on average.
        max_lines = max(10, self.n // 4)
        overlap_lines = max(0, min(max_lines - 1, self.o // 4))

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def flush_chunk():
            nonlocal current, current_len
            if current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0

        i = 0
        while i < len(lines):
            line = lines[i]
            # Slightly prefer boundaries at blank lines or top-level defs/classes.
            is_boundary_candidate = (
                not line.strip()
                or line.lstrip().startswith("def ")
                or line.lstrip().startswith("class ")
            )

            if current_len >= max_lines and is_boundary_candidate:
                flush_chunk()
                # Overlap: keep last few lines from previous chunk
                if overlap_lines > 0 and chunks:
                    last_chunk_lines = chunks[-1].splitlines()
                    tail = last_chunk_lines[-overlap_lines:]
                    current = tail.copy()
                    current_len = len(current)

            current.append(line)
            current_len += 1
            i += 1

        flush_chunk()
        return chunks
