from __future__ import annotations

from collections.abc import Callable
import re


class TextSplitter:
    """
    A smarter text splitter that supports very long docs by being structure-aware.

    It supports three main modes:

      - "auto" (default): heuristically choose between markdown/code/paragraph splitting.
      - "markdown": treat the text as markdown (docs, READMEs, websites converted to md).
      - "code": treat the text as source code (Python/JS/etc.).
      - "paragraph": generic prose/HTML-derived text.

    The unit of budgeting is *approximate tokens*, estimated via whitespace-separated
    "words" (using a regex). This is cheap and good enough for chunk sizing;
    the actual embedding/tokenizer can still use its own true token counting.

    Example:
        splitter = TextSplitter(target_tokens=400, overlap_tokens=80, mode="markdown")
        chunks = splitter.split(long_markdown_doc)
    """

    def __init__(
        self,
        target_tokens: int = 400,
        overlap_tokens: int = 60,
        mode: str = "auto",
        token_counter: Callable[[str], int] | None = None,
    ):
        # token budget per chunk
        self.n = max(50, target_tokens)
        # overlap budget between consecutive chunks
        self.o = max(0, min(self.n - 1, overlap_tokens))
        # "auto", "markdown", "code", "paragraph"
        self.mode = mode
        # optional custom token counting function (not used in current implementation, but could be integrated for more accurate budgeting)
        self._token_counter = token_counter

    @property
    def has_true_tokenizer(self) -> bool:
        return self._token_counter is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(self, text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []

        if self.mode == "markdown" or (self.mode == "auto" and self._looks_like_markdown(text)):
            return self._split_markdown(text)

        if self.mode == "code" or (self.mode == "auto" and self._looks_like_code(text)):
            return self._split_code(text)

        # Fallback: prose/paragraph splitter
        return self._split_paragraphs(text)

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    def _approx_tokens(self, text: str) -> int:
        """
        Token count used for budgeting.

        If a token_counter callback is provided, use that (so chunk sizes match
        the actual model/embedding tokenizer). Otherwise, fall back to a cheap
        heuristic based on non-whitespace "words".
        """
        if not text:
            return 0

        if self._token_counter is not None:
            try:
                return int(self._token_counter(text))
            except Exception:
                # Fail soft: don't blow up chunking if the callback misbehaves.
                pass

        # Fallback heuristic
        return len(re.findall(r"\S+", text))

    def _looks_like_markdown(self, text: str) -> bool:
        """
        Heuristic: treat as markdown if we see headings, code fences, or bullet lists
        often enough.
        """
        # quick early exits
        if "# " in text or "## " in text:
            return True
        if "```" in text:
            return True
        # simple bullet detection
        if re.search(r"^\s*[-*+]\s+\S", text, flags=re.MULTILINE):
            return True
        # tables / links / emphasis
        if "|" in text and re.search(r"\|.*\|", text):
            return True
        if "[" in text and "]" in text and "(" in text and ")" in text:  # noqa SIM103
            return True
        return False

    def _looks_like_code(self, text: str) -> bool:
        """
        Heuristic: treat as code if there are many lines ending with ':' or '{',
        or common code keywords.
        """
        # code fences
        if "```python" in text or "```js" in text or "```ts" in text:
            return True

        lines = text.splitlines()
        code_like = 0
        for ln in lines:
            ls = ln.strip()
            if not ls:
                continue
            if ls.startswith(("def ", "class ", "async def ")):
                return True
            if ls.startswith(("import ", "from ")):
                code_like += 1
            if ls.endswith(":") or ls.endswith("{") or ls.endswith("};"):
                code_like += 1
        return code_like >= 5

    # ------------------------------------------------------------------
    # Paragraph-based splitter (generic prose)
    # ------------------------------------------------------------------
    def _split_paragraphs(self, text: str) -> list[str]:
        """
        Split text into chunks based on paragraphs (double newlines) and a token budget.
        Overlap is implemented in terms of paragraphs rather than raw words.
        """
        # 1) naive paragraph segmentation
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return self._split_words(text)

        return self._group_blocks(paragraphs, joiner="\n\n")

    # ------------------------------------------------------------------
    # Markdown-aware splitter
    # ------------------------------------------------------------------
    def _split_markdown(self, text: str) -> list[str]:
        """
        Split markdown into sections based on headings, keeping headings attached
        to their content, then group by token budget with overlap.

        If a single heading section is huge, we further paragraph-split inside it.
        """
        lines = text.splitlines()
        sections: list[str] = []
        current_lines: list[str] = []

        def flush_section():
            if current_lines:
                sec = "\n".join(current_lines).strip()
                if sec:
                    sections.append(sec)

        heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+\S")

        for ln in lines:
            if heading_pattern.match(ln):
                # new section starting at heading
                flush_section()
                current_lines[:] = [ln]
            else:
                current_lines.append(ln)
        flush_section()

        if not sections:
            # fallback to paragraphs if no headings found
            return self._split_paragraphs(text)

        # For very large sections, split internally by paragraphs before grouping.
        normalized_sections: list[str] = []
        for sec in sections:
            if self._approx_tokens(sec) <= self.n * 2:
                normalized_sections.append(sec)
            else:
                # break down this section by paragraphs, but keep the first heading
                sub = self._split_markdown_section_to_paragraphs(sec)
                normalized_sections.extend(sub)

        return self._group_blocks(normalized_sections, joiner="\n\n")

    def _split_markdown_section_to_paragraphs(self, section: str) -> list[str]:
        """
        Given a big markdown section (starting with a heading), break it into
        smaller blocks: heading + 1..k paragraphs per block.
        """
        # separate heading line from body
        lines = section.splitlines()
        if not lines:
            return []

        heading = lines[0]
        body = "\n".join(lines[1:])

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if not paragraphs:
            return [section]

        blocks: list[str] = []
        current_paras: list[str] = []
        current_len = 0

        for p in paragraphs:
            p_len = self._approx_tokens(p)
            if current_paras and current_len + p_len > self.n:
                # flush: heading + paras
                block = "\n\n".join([heading] + current_paras)
                blocks.append(block)
                # reset with overlap: keep last few paras as context
                if self.o > 0:
                    overlap_paras: list[str] = []
                    overlap_len = 0
                    for para in reversed(current_paras):
                        l = self._approx_tokens(para)  # noqa E741
                        if overlap_paras and overlap_len + l > self.o:
                            break
                        overlap_paras.insert(0, para)
                        overlap_len += l
                    current_paras = overlap_paras
                    current_len = sum(self._approx_tokens(x) for x in current_paras)
                else:
                    current_paras = []
                    current_len = 0

            current_paras.append(p)
            current_len += p_len

        if current_paras:
            block = "\n\n".join([heading] + current_paras)
            blocks.append(block)

        return blocks

    # ------------------------------------------------------------------
    # Code-aware splitter
    # ------------------------------------------------------------------
    def _split_code(self, text: str) -> list[str]:
        """
        Try to split code along function/class boundaries and then group blocks
        under a token budget with overlap.

        If heuristics fail, fall back to paragraph/word splitting.
        """
        lines = text.splitlines()
        if not lines:
            return []

        blocks: list[list[str]] = []
        current: list[str] = []

        def flush_block():
            if current:
                blk = "\n".join(current).rstrip()
                if blk:
                    blocks.append(blk)
                current.clear()

        # simple heuristics for "block starters"
        block_start = re.compile(
            r"^\s*(def |class |async def |@|\w+\s*=\s*function\s*\(|if __name__ == ['\"]__main__['\"]:)"
        )

        for ln in lines:
            if block_start.match(ln) and current:
                flush_block()
            current.append(ln)
        flush_block()

        # If we didn't find any reasonable blocks, fallback.
        if not blocks or len(blocks) == 1:
            return self._split_paragraphs(text)

        return self._group_blocks(blocks, joiner="\n")

    # ------------------------------------------------------------------
    # Generic block grouper with token budget + overlap
    # ------------------------------------------------------------------
    def _group_blocks(self, blocks: list[str], joiner: str) -> list[str]:
        """
        Take a sequence of pre-chunked "blocks" (paragraphs, sections, functions)
        and group them into chunks that respect the token budget `self.n`,
        with overlap budget `self.o`.
        """
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for block in blocks:
            block = block.strip()
            if not block:
                continue

            b_len = self._approx_tokens(block)

            # If a single block is larger than ~2x target, hard-split it by words
            # so we don't completely blow up the budget.
            if b_len > self.n * 2:
                if current:
                    chunks.append(joiner.join(current).strip())
                    current = []
                    current_len = 0
                huge_subchunks = self._split_words(block)
                for sub in huge_subchunks:
                    if self._approx_tokens(sub) > self.n:
                        # If even sub is too large, just push it as-is (rare).
                        chunks.append(sub)
                    else:
                        chunks.append(sub)
                continue

            # If adding this block would exceed the budget, flush current
            if current and current_len + b_len > self.n:
                chunks.append(joiner.join(current).strip())

                # Overlap: keep tail blocks as context
                if self.o > 0:
                    overlap_blocks: list[str] = []
                    overlap_len = 0
                    for b in reversed(current):
                        l = self._approx_tokens(b)  # noqa E741
                        if overlap_blocks and overlap_len + l > self.o:
                            break
                        overlap_blocks.insert(0, b)
                        overlap_len += l
                    current = overlap_blocks
                    current_len = sum(self._approx_tokens(x) for x in current)
                else:
                    current = []
                    current_len = 0

            current.append(block)
            current_len += b_len

        if current:
            chunks.append(joiner.join(current).strip())

        return chunks

    # ------------------------------------------------------------------
    # Word-level fallback splitter
    # ------------------------------------------------------------------
    def _split_words(self, text: str) -> list[str]:
        """
        Last-resort splitter: chunk purely by words with overlap.
        This preserves backward-compat behavior for edge cases.
        """
        words = re.findall(r"\S+", text)
        if not words:
            return []

        step = self.n - self.o
        if step <= 0:
            step = self.n

        chunks: list[str] = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.n])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
