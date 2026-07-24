import copy
from dataclasses import dataclass
from typing import Any, Literal

ChatOutputFormat = Literal[
    "text", "json_object", "json_schema", "raw", "json"
]  # "json" is a deprecated alias of "json_object"

ImageFormat = Literal["png", "jpeg", "webp"]
ImageResponseFormat = Literal["b64_json", "url"]  # url only for dall-e models typically


@dataclass(frozen=True)
class StructuredOutputRequest:
    """
    Request provider-neutral structured output from an LLM chat call.

    Examples:
        Define a compact object response:
            ```python
            request = StructuredOutputRequest(
                name="Answer",
                schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            )
            ```

        Pass the request through the ordinary client:
            ```python
            text, usage = await context.llm().chat(
                [{"role": "user", "content": "Answer briefly."}],
                structured_output=request,
            )
            ```

    Args:
        name: Stable logical name for the root response schema.
        schema: Canonical caller-owned JSON Schema used for local validation.

    Notes:
        The request is provider-neutral. Provider capability selection and any
        provider-specific schema projection remain internal to AetherGraph.
    """

    name: str
    schema: dict[str, Any]

    def __post_init__(self) -> None:
        """
        Validate and detach the request from caller-owned mutable schema data.

        Examples:
            Preserve the supplied schema:
                ```python
                request = StructuredOutputRequest("Answer", {"type": "object"})
                assert request.schema["type"] == "object"
                ```

            Reject an empty schema name:
                ```python
                try:
                    StructuredOutputRequest("", {"type": "object"})
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized structured-output request.

        Returns:
            None: The frozen instance is validated and its schema is detached.

        Notes:
            JSON Schema remains arbitrary provider-neutral data; AetherGraph
            does not import an engine contract to interpret its domain meaning.
        """

        normalized_name = str(self.name or "").strip()
        if not normalized_name:
            raise ValueError("structured output schema name must not be empty")
        if not isinstance(self.schema, dict):
            raise TypeError("structured output schema must be a JSON object")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "schema", copy.deepcopy(self.schema))


@dataclass(frozen=True)
class ImageInput:
    data: bytes | None = None
    b64: str | None = None  # base64 without data: prefix
    mime_type: str | None = None
    url: str | None = None  # http(s) url OR provider file_uri
    is_file_uri: bool = False  # Gemini file URIs


class LLMError(RuntimeError):
    """Base class for typed LLM service failures."""


class LLMUnsupportedFeatureError(LLMError):
    def __init__(self, provider: str, model: str | None, feature: str, detail: str | None = None):
        msg = f"Provider '{provider}' / model '{model or '?'}' does not support: {feature}"
        if detail:
            msg += f" ({detail})"
        super().__init__(msg)


class LLMStructuredOutputError(LLMError):
    """Base class for structured-output failures outside caller validation."""


class LLMStructuredOutputCapabilityError(LLMStructuredOutputError):
    """Fail a structured request before transport when policy cannot be met."""

    def __init__(
        self,
        *,
        provider: str,
        model: str | None,
        policy: str,
        detail: str,
    ) -> None:
        """
        Build one provider-neutral structured-output capability failure.

        Examples:
            Construct a native-required failure:
                ```python
                error = LLMStructuredOutputCapabilityError(
                    provider="deepseek",
                    model="deepseek-chat",
                    policy="native_required",
                    detail="No native schema capability.",
                )
                assert "native_required" in str(error)
                ```

            Inspect stable fields:
                ```python
                error = LLMStructuredOutputCapabilityError(
                    provider="custom",
                    model=None,
                    policy="native_required",
                    detail="Unknown capability.",
                )
                assert error.provider == "custom"
                ```

        Args:
            provider: Configured provider name.
            model: Configured model or deployment identifier.
            policy: Profile-owned structured-output policy.
            detail: Deterministic capability or projection explanation.

        Returns:
            None: Initializes the exception.

        Notes:
            This error occurs before provider transport and should not be
            treated as malformed model output.
        """

        super().__init__(
            f"Structured output policy '{policy}' cannot be satisfied by "
            f"provider '{provider}' / model '{model or '?'}': {detail}"
        )
        self.provider = provider
        self.model = model
        self.policy = policy
        self.detail = detail


class LLMStructuredOutputProviderRequestError(LLMStructuredOutputError):
    """Provider rejected a prepared structured-output request."""


class LLMStructuredOutputRefusalError(LLMStructuredOutputError):
    """Provider returned an explicit refusal instead of structured output."""


class LLMStructuredOutputTruncationError(LLMStructuredOutputError):
    """Provider ended a structured response before it was complete."""


class LLMStructuredOutputParseError(LLMStructuredOutputError):
    """Provider output was not one complete JSON value."""


class LLMStructuredOutputValidationError(LLMStructuredOutputError):
    """Parsed output failed the caller's canonical JSON Schema."""


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
