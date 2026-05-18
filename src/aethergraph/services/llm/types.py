from dataclasses import dataclass
from typing import Any, Literal

ChatOutputFormat = Literal[
    "text", "json_object", "json_schema", "raw", "json"
]  # "json" is a deprecated alias of "json_object"

ImageFormat = Literal["png", "jpeg", "webp"]
ImageResponseFormat = Literal["b64_json", "url"]  # url only for dall-e models typically


@dataclass(frozen=True)
class JsonSchemaSpec:
    name: str
    schema: dict[str, Any]
    strict: bool = True


@dataclass(frozen=True)
class ImageInput:
    data: bytes | None = None
    b64: str | None = None  # base64 without data: prefix
    mime_type: str | None = None
    url: str | None = None  # http(s) url OR provider file_uri
    is_file_uri: bool = False  # Gemini file URIs


class LLMUnsupportedFeatureError(RuntimeError):
    def __init__(self, provider: str, model: str | None, feature: str, detail: str | None = None):
        msg = f"Provider '{provider}' / model '{model or '?'}' does not support: {feature}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class LLMError(RuntimeError):
    """Base class for typed LLM service failures."""


class LLMCallBudgetExceededError(LLMError):
    def __init__(self, *, run_id: str, calls: int, limit: int):
        super().__init__(
            f"LLM call limit exceeded for this run ({calls} > {limit}). "
            "Consider simplifying the graph or raising the limit."
        )
        self.run_id = run_id
        self.calls = calls
        self.limit = limit


class LLMRunBudgetExceededError(LLMError):
    def __init__(
        self,
        *,
        run_id: str,
        total_tokens: int,
        limit: int,
        prompt_tokens: int,
        completion_tokens: int,
    ):
        super().__init__(
            f"LLM token limit exceeded for this run ({total_tokens} > {limit}). "
            "Consider simplifying the graph or raising the limit."
        )
        self.run_id = run_id
        self.total_tokens = total_tokens
        self.limit = limit
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class LLMInputTooLargeError(LLMError):
    def __init__(
        self,
        *,
        run_id: str,
        spent_tokens: int,
        estimated_input_tokens: int,
        reserved_output_tokens: int,
        projected_total_tokens: int,
        limit: int,
    ):
        super().__init__(
            "LLM request exceeds the remaining token budget for this run "
            f"({projected_total_tokens} > {limit}). "
            "Consider simplifying the graph or raising the limit."
        )
        self.run_id = run_id
        self.spent_tokens = spent_tokens
        self.estimated_input_tokens = estimated_input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.projected_total_tokens = projected_total_tokens
        self.limit = limit


@dataclass
class GeneratedImage:
    # Exactly one of these is typically present.
    b64: str | None = None
    url: str | None = None
    mime_type: str | None = None
    revised_prompt: str | None = None


@dataclass
class ImageGenerationResult:
    images: list[GeneratedImage]
    usage: dict[str, int]  # often empty for image endpoints
    raw: dict[str, Any] | None = None
