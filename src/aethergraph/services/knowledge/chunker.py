from __future__ import annotations


class TextSplitter:
    """A simple text splitter that splits text into chunks of approximately target_tokens,
    with a specified overlap in tokens.

    Example:
        splitter = TextSplitter(target_tokens=400, overlap_tokens=60)
        chunks = splitter.split(long_text)
        for chunk in chunks:
            print(chunk)
    """

    def __init__(self, target_tokens: int = 400, overlap_tokens: int = 60):
        self.n = max(50, target_tokens)
        self.o = max(0, min(self.n - 1, overlap_tokens))

    def split(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.n - self.o
        chunks = []
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.n])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
