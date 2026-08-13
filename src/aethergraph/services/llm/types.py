import copy
from dataclasses import dataclass, field
from typing import Any, Literal

ChatOutputFormat = Literal[
    "text", "json_object", "json_schema", "raw", "json"
]  # "json" is a deprecated alias of "json_object"
PromptCacheStrategy = Literal["stable_prefix"]
StructuredOutputValidationOwner = Literal["aethergraph", "caller"]

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
        validation_owner: Component responsible for canonical response
            validation after provider generation. The default keeps validation
            in AetherGraph; `caller` returns the provider response for one
            domain-specific validator.

    Notes:
        The request is provider-neutral. Provider capability selection and any
        provider-specific schema projection remain internal to AetherGraph.
    """

    name: str
    schema: dict[str, Any]
    validation_owner: StructuredOutputValidationOwner = "aethergraph"

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
        if self.validation_owner not in {"aethergraph", "caller"}:
            raise ValueError("structured output validation_owner must be 'aethergraph' or 'caller'")
        object.__setattr__(self, "name", normalized_name)
        object.__setattr__(self, "schema", copy.deepcopy(self.schema))


@dataclass(frozen=True)
class PromptCacheRequest:
    """
    Request provider-neutral caching for stable message-prefix boundaries.

    The request identifies message indexes whose complete prefixes are stable.
    Provider capability resolution and request translation remain internal to
    AetherGraph and never change the semantic message order.

    Examples:
        Mark a stable system header:
            ```python
            request = PromptCacheRequest(
                stable_message_indexes=(0,),
                prefix_family="assistant.instructions.v1",
            )
            ```

        Mark an append-only transcript:
            ```python
            request = PromptCacheRequest(
                stable_message_indexes=(0, 2, 4),
                prefix_family="session.transcript.v3",
            )
            assert request.strategy == "stable_prefix"
            ```

    Args:
        stable_message_indexes: Sorted, unique zero-based message indexes that
            end persistent cache-eligible stable prefixes. Callers retain an
            index while its corresponding prefix remains in the prompt; the
            provider decides whether that breakpoint is a new write or a
            read-only match.
        prefix_family: Caller-owned stable identity for the prompt family.
        strategy: Provider-neutral cache strategy. Only `stable_prefix` is
            supported.

    Returns:
        PromptCacheRequest: An immutable, validated cache request.

    Notes:
        The prefix family is used to derive an opaque provider cache key. It
        must not contain credentials or other secret values.
    """

    stable_message_indexes: tuple[int, ...]
    prefix_family: str
    strategy: PromptCacheStrategy = "stable_prefix"

    def __post_init__(self) -> None:
        """
        Validate and normalize one prompt-cache request.

        The method converts the supplied indexes to an immutable tuple and
        rejects ambiguous or non-prefix ordering before provider dispatch.

        Examples:
            Normalize a list supplied by a dynamic caller:
                ```python
                request = PromptCacheRequest([0, 2], "ledger.v1")
                assert request.stable_message_indexes == (0, 2)
                ```

            Reject duplicate boundaries:
                ```python
                try:
                    PromptCacheRequest((0, 0), "ledger.v1")
                except ValueError:
                    pass
                ```

        Args:
            self: Newly initialized prompt-cache request.

        Returns:
            None: The frozen instance is validated and normalized in place.

        Notes:
            Message-list bounds are validated by `GenericLLMClient.chat()`
            because the request value intentionally does not own messages.
        """

        if self.strategy != "stable_prefix":
            raise ValueError("prompt cache strategy must be 'stable_prefix'")
        family = str(self.prefix_family or "").strip()
        if not family:
            raise ValueError("prompt cache prefix_family must not be empty")
        if len(family) > 256:
            raise ValueError("prompt cache prefix_family must be at most 256 characters")
        indexes = tuple(self.stable_message_indexes)
        if not indexes:
            raise ValueError("prompt cache stable_message_indexes must not be empty")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indexes):
            raise TypeError("prompt cache message indexes must be integers")
        if any(index < 0 for index in indexes):
            raise ValueError("prompt cache message indexes must be non-negative")
        if tuple(sorted(set(indexes))) != indexes:
            raise ValueError("prompt cache message indexes must be sorted and unique")
        object.__setattr__(self, "stable_message_indexes", indexes)
        object.__setattr__(self, "prefix_family", family)


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


class LLMStructuredOutputRefusalError(LLMStructuredOutputError):
    """Provider returned an explicit refusal instead of structured output."""


class LLMStructuredOutputTruncationError(LLMStructuredOutputError):
    """Provider ended a structured response before it was complete."""


class LLMStructuredOutputResponseError(LLMStructuredOutputError):
    """Describe one bounded local structured-response validation failure."""

    def __init__(
        self,
        *,
        code: str,
        summary: str,
        path: str = "",
        schema_path: str = "",
        validator: str = "",
        invalid_value: str = "",
        expected: tuple[Any, ...] = (),
        canonical_schema_fingerprint: str = "",
        response_state: str,
    ) -> None:
        super().__init__(summary)
        self.code = str(code)
        self.summary = str(summary)
        self.path = str(path)
        self.schema_path = str(schema_path)
        self.validator = str(validator)
        self.invalid_value = str(invalid_value)
        self.expected = tuple(expected)
        self.canonical_schema_fingerprint = str(canonical_schema_fingerprint)
        self.response_state = str(response_state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "summary": self.summary,
            "path": self.path,
            "schema_path": self.schema_path,
            "validator": self.validator,
            "invalid_value": self.invalid_value,
            "expected": list(self.expected),
            "canonical_schema_fingerprint": self.canonical_schema_fingerprint,
            "response_state": self.response_state,
        }


class LLMStructuredOutputParseError(LLMStructuredOutputResponseError):
    """Provider output was not one complete JSON value."""


class LLMStructuredOutputValidationError(LLMStructuredOutputResponseError):
    """Parsed output failed the caller's canonical JSON Schema."""


@dataclass(frozen=True)
class LLMRequestEstimate:
    """Describe one provider-neutral chat request estimate.

    The estimate covers the current request only. It never includes usage from
    earlier calls in the run.
    """

    model: str
    estimated_input_tokens: int
    reserved_output_tokens: int
    estimated_total_tokens: int
    context_window_tokens: int | None
    source: str


class LLMContextWindowExceededError(LLMError):
    """Signal that one current request cannot fit its configured model window."""

    def __init__(
        self,
        *,
        model: str,
        estimated_input_tokens: int,
        reserved_output_tokens: int,
        estimated_total_tokens: int,
        limit: int,
        estimate_source: str,
    ) -> None:
        super().__init__(
            "LLM request exceeds the configured model context window "
            f"(estimated_input={estimated_input_tokens}, "
            f"reserved_output={reserved_output_tokens}, "
            f"estimated_total={estimated_total_tokens}, limit={limit}, "
            f"model='{model}', estimate_source='{estimate_source}')."
        )
        self.model = model
        self.estimated_input_tokens = estimated_input_tokens
        self.reserved_output_tokens = reserved_output_tokens
        self.estimated_total_tokens = estimated_total_tokens
        self.limit = limit
        self.estimate_source = estimate_source


class LLMRunQuotaError(LLMError):
    """Base class for infrastructure-owned per-run LLM quota failures."""

    def __init__(
        self,
        *,
        run_id: str,
        quota: str,
        consumed: int,
        requested: int,
        projected: int,
        limit: int,
        phase: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"LLM infrastructure quota '{quota}' {phase} "
            f"(consumed={consumed}, requested={requested}, "
            f"projected={projected}, limit={limit}, run_id='{run_id}')."
        )
        self.run_id = run_id
        self.quota = quota
        self.consumed = consumed
        self.requested = requested
        self.projected = projected
        self.limit = limit
        self.phase = phase
        self.usage = copy.deepcopy(usage) if usage is not None else None


class LLMRunQuotaWouldExceedError(LLMRunQuotaError):
    """Reject a call before transport when it would cross an AG quota."""


class LLMRunQuotaExceededError(LLMRunQuotaError):
    """Report actual provider usage that crossed an AG quota."""


class ModelOperationRunQuotaError(LLMError):
    """Describe one infrastructure-owned non-Chat model quota failure.

    Intro:
        Retains the operation, metric, run identity, requested amount, projected
        consumption, configured limit, and optional provider usage receipt so
        callers can distinguish admission rejection from post-response overage.

    Examples:
        Inspect a preflight embedding rejection:
            ```python
            error = ModelOperationRunQuotaError(
                operation="embedding",
                run_id="run-1",
                quota="texts",
                consumed=2,
                requested=2,
                projected=4,
                limit=3,
                phase="would be exceeded before provider dispatch",
            )
            assert error.operation == "embedding"
            ```

        Retain a post-response image receipt:
            ```python
            error = ModelOperationRunQuotaError(
                operation="image_generation",
                run_id="run-2",
                quota="total_tokens",
                consumed=0,
                requested=8,
                projected=8,
                limit=4,
                phase="was exceeded by actual provider usage",
                usage={"total_tokens": 8},
            )
            assert error.usage == {"total_tokens": 8}
            ```

    Args:
        self: Newly allocated typed quota error.
        operation: Stable model operation identity.
        run_id: Runtime run whose shared ledger rejected or exceeded the quota.
        quota: Stable operation-specific metric name.
        consumed: Committed plus concurrently reserved units before this request.
        requested: Units requested or reported by this provider call.
        projected: Resulting units compared with the configured limit.
        limit: Configured inclusive maximum for the metric.
        phase: Human-readable admission or reconciliation phase.
        usage: Optional detached provider usage receipt.

    Returns:
        None: Initializes the exception and stable inspection fields.

    Notes:
        Agent-loop budgets remain outside this infrastructure exception family.
    """

    def __init__(
        self,
        *,
        operation: str,
        run_id: str,
        quota: str,
        consumed: int,
        requested: int,
        projected: int,
        limit: int,
        phase: str,
        usage: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            f"Model operation '{operation}' infrastructure quota '{quota}' {phase} "
            f"(consumed={consumed}, requested={requested}, "
            f"projected={projected}, limit={limit}, run_id='{run_id}')."
        )
        self.operation = operation
        self.run_id = run_id
        self.quota = quota
        self.consumed = consumed
        self.requested = requested
        self.projected = projected
        self.limit = limit
        self.phase = phase
        self.usage = copy.deepcopy(usage) if usage is not None else None


class ModelOperationRunQuotaWouldExceedError(ModelOperationRunQuotaError):
    """Reject a non-Chat model call before transport when it crosses a quota."""


class ModelOperationRunQuotaExceededError(ModelOperationRunQuotaError):
    """Report actual non-Chat provider usage that crossed a quota."""


UsageAvailability = Literal["complete", "partial", "unavailable"]


def _operation_usage_int(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, int) and value >= 0:
            return value
    return None


@dataclass(frozen=True)
class EmbeddingUsage:
    """Retain normalized embedding input usage and its provider receipt.

    Intro:
        Distinguishes unavailable provider usage from a real zero-token receipt
        without changing the vector-only compatibility facade.

    Examples:
        Normalize an OpenAI embedding receipt:
            ```python
            usage = EmbeddingUsage.from_provider_usage(
                {"prompt_tokens": 4, "total_tokens": 4}
            )
            assert usage.input_tokens == 4
            ```

        Preserve unavailable usage:
            ```python
            usage = EmbeddingUsage.from_provider_usage(None)
            assert usage.availability == "unavailable"
            ```

    Args:
        availability: Whether normalized provider usage is complete, partial,
            or unavailable.
        input_tokens: Provider-reported embedding input tokens when known.
        provider_usage_raw: Detached provider usage receipt.

    Returns:
        EmbeddingUsage: Immutable operation-specific usage receipt.

    Notes:
        Character estimates are deliberately excluded because they are not
        provider billing truth.
    """

    availability: UsageAvailability
    input_tokens: int | None = None
    provider_usage_raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.availability not in {"complete", "partial", "unavailable"}:
            raise ValueError("embedding usage availability is invalid")
        if self.input_tokens is not None and (
            isinstance(self.input_tokens, bool)
            or not isinstance(self.input_tokens, int)
            or self.input_tokens < 0
        ):
            raise ValueError("embedding input_tokens must be a non-negative integer or None")
        if self.availability == "unavailable" and self.input_tokens is not None:
            raise ValueError("unavailable embedding usage cannot contain input tokens")
        if self.availability != "unavailable" and self.input_tokens is None:
            raise ValueError("available embedding usage requires input tokens")
        object.__setattr__(self, "provider_usage_raw", copy.deepcopy(self.provider_usage_raw))

    @classmethod
    def from_provider_usage(cls, usage: dict[str, Any] | None) -> "EmbeddingUsage":
        """Normalize one embedding provider receipt.

        Intro:
            Accepts common snake-case and Gemini camel-case token fields while
            retaining the exact detached provider mapping.

        Examples:
            Normalize prompt tokens:
                ```python
                usage = EmbeddingUsage.from_provider_usage({"prompt_tokens": 3})
                assert usage.availability == "complete"
                ```

            Normalize Gemini metadata:
                ```python
                usage = EmbeddingUsage.from_provider_usage({"promptTokenCount": 2})
                assert usage.input_tokens == 2
                ```

        Args:
            cls: The `EmbeddingUsage` class.
            usage: Optional provider-owned usage mapping.

        Returns:
            EmbeddingUsage: Detached normalized embedding usage.

        Notes:
            A non-empty receipt without a recognized token counter is retained
            as unavailable rather than coerced to zero.
        """

        raw = copy.deepcopy(dict(usage or {}))
        tokens = _operation_usage_int(
            raw,
            "input_tokens",
            "prompt_tokens",
            "total_tokens",
            "inputTokenCount",
            "promptTokenCount",
            "totalTokenCount",
        )
        if tokens is None:
            return cls(availability="unavailable", provider_usage_raw=raw)
        return cls(availability="complete", input_tokens=tokens, provider_usage_raw=raw)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the embedding usage receipt.

        Intro:
            Produces a detached JSON-compatible observation and metering value.

        Examples:
            Serialize available usage:
                ```python
                payload = EmbeddingUsage.from_provider_usage(
                    {"prompt_tokens": 3}
                ).to_dict()
                assert payload["input_tokens"] == 3
                ```

            Serialize unavailable usage:
                ```python
                payload = EmbeddingUsage.from_provider_usage(None).to_dict()
                assert payload["availability"] == "unavailable"
                ```

        Args:
            self: Immutable embedding usage receipt.

        Returns:
            dict[str, Any]: Detached usage payload.

        Notes:
            Missing counters remain `None` and are never represented as zero.
        """

        return {
            "availability": self.availability,
            "input_tokens": self.input_tokens,
            "provider_usage_raw": copy.deepcopy(self.provider_usage_raw),
        }


@dataclass
class EmbeddingResult:
    """Carry normalized embedding vectors with retained provider usage.

    Intro:
        Keeps the canonical embedding result intact until accounting completes,
        while allowing the public facade to return only vectors.

    Examples:
        Build an unavailable-usage result:
            ```python
            result = EmbeddingResult(vectors=[[0.1, 0.2]])
            assert result.usage.availability == "unavailable"
            ```

        Build a provider-reported result:
            ```python
            result = EmbeddingResult(
                vectors=[[0.1]],
                usage=EmbeddingUsage.from_provider_usage({"prompt_tokens": 2}),
            )
            ```

    Args:
        vectors: Ordered normalized embedding vectors.
        usage: Typed operation-specific provider usage.

    Returns:
        EmbeddingResult: Detached vectors and retained provider usage.

    Notes:
        Vector detachment prevents adapters and compatibility consumers from
        sharing mutable nested lists.
    """

    vectors: list[list[float]]
    usage: EmbeddingUsage = field(
        default_factory=lambda: EmbeddingUsage(availability="unavailable")
    )

    def __post_init__(self) -> None:
        """Detach normalized vector rows after construction.

        Intro:
            Copies every vector row so provider response containers cannot be
            mutated through the canonical result.

        Examples:
            Detach one vector:
                ```python
                source = [[0.1]]
                result = EmbeddingResult(source)
                source[0][0] = 0.2
                assert result.vectors == [[0.1]]
                ```

            Preserve an empty batch:
                ```python
                assert EmbeddingResult([]).vectors == []
                ```

        Args:
            self: Newly initialized embedding result.

        Returns:
            None: Completes after replacing vectors with detached rows.

        Notes:
            Numeric element validation remains the physical adapter's response-
            shape responsibility.
        """

        self.vectors = [list(vector) for vector in self.vectors]


@dataclass(frozen=True)
class ImageGenerationUsage:
    """Retain normalized image-generation token usage and provider receipt.

    Intro:
        Keeps image usage distinct from Chat usage while preserving providers
        that expose only a subset of token counters.

    Examples:
        Normalize a complete image receipt:
            ```python
            usage = ImageGenerationUsage.from_provider_usage(
                {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10}
            )
            assert usage.total_tokens == 10
            ```

        Preserve unavailable image usage:
            ```python
            usage = ImageGenerationUsage.from_provider_usage({})
            assert usage.availability == "unavailable"
            ```

    Args:
        availability: Whether the provider supplied complete, partial, or no
            normalized usage.
        input_tokens: Provider-reported input tokens when known.
        output_tokens: Provider-reported image output tokens when known.
        total_tokens: Provider-reported or exactly derived total when known.
        provider_usage_raw: Detached provider usage receipt.

    Returns:
        ImageGenerationUsage: Immutable operation-specific usage receipt.

    Notes:
        Image count, size, and quality are invocation dimensions, not tokens.
    """

    availability: UsageAvailability
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    provider_usage_raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.availability not in {"complete", "partial", "unavailable"}:
            raise ValueError("image usage availability is invalid")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"image {name} must be a non-negative integer or None")
        if self.availability == "unavailable" and any(
            value is not None
            for value in (self.input_tokens, self.output_tokens, self.total_tokens)
        ):
            raise ValueError("unavailable image usage cannot contain token counters")
        object.__setattr__(self, "provider_usage_raw", copy.deepcopy(self.provider_usage_raw))

    @classmethod
    def from_provider_usage(cls, usage: dict[str, Any] | None) -> "ImageGenerationUsage":
        """Normalize one image-generation provider receipt.

        Intro:
            Accepts common provider token keys, derives totals only from two
            known components, and retains unknown receipt fields verbatim.

        Examples:
            Derive a total:
                ```python
                usage = ImageGenerationUsage.from_provider_usage(
                    {"input_tokens": 2, "output_tokens": 5}
                )
                assert usage.total_tokens == 7
                ```

            Preserve a partial total-only receipt:
                ```python
                usage = ImageGenerationUsage.from_provider_usage({"total_tokens": 7})
                assert usage.availability == "partial"
                ```

        Args:
            cls: The `ImageGenerationUsage` class.
            usage: Optional provider-owned usage mapping.

        Returns:
            ImageGenerationUsage: Detached normalized image usage.

        Notes:
            Missing provider usage remains unavailable and is not estimated.
        """

        raw = copy.deepcopy(dict(usage or {}))
        input_tokens = _operation_usage_int(
            raw, "input_tokens", "prompt_tokens", "inputTokenCount", "promptTokenCount"
        )
        output_tokens = _operation_usage_int(
            raw, "output_tokens", "completion_tokens", "outputTokenCount"
        )
        total_tokens = _operation_usage_int(raw, "total_tokens", "totalTokenCount")
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        counters = (input_tokens, output_tokens, total_tokens)
        if all(value is None for value in counters):
            availability: UsageAvailability = "unavailable"
        elif input_tokens is not None and output_tokens is not None:
            availability = "complete"
        else:
            availability = "partial"
        return cls(
            availability=availability,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            provider_usage_raw=raw,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the image-generation usage receipt.

        Intro:
            Produces a detached JSON-compatible value for meters and traces.

        Examples:
            Serialize complete usage:
                ```python
                payload = ImageGenerationUsage.from_provider_usage(
                    {"input_tokens": 2, "output_tokens": 3}
                ).to_dict()
                assert payload["total_tokens"] == 5
                ```

            Serialize unavailable usage:
                ```python
                payload = ImageGenerationUsage.from_provider_usage(None).to_dict()
                assert payload["availability"] == "unavailable"
                ```

        Args:
            self: Immutable image-generation usage receipt.

        Returns:
            dict[str, Any]: Detached usage payload.

        Notes:
            The raw receipt remains available for provider diagnostics.
        """

        return {
            "availability": self.availability,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "provider_usage_raw": copy.deepcopy(self.provider_usage_raw),
        }


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
    usage_receipt: ImageGenerationUsage | None = None

    def __post_init__(self) -> None:
        self.images = list(self.images)
        self.usage = copy.deepcopy(dict(self.usage or {}))
        self.raw = copy.deepcopy(self.raw) if self.raw is not None else None
        if self.usage_receipt is None:
            self.usage_receipt = ImageGenerationUsage.from_provider_usage(self.usage)
