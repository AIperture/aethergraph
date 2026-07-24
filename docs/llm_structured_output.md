# Structured LLM output

AetherGraph (AG) owns provider-neutral structured-output requests, provider
capability selection, provider request translation, canonical JSON validation,
typed LLM failures, and generic LLM-call observability.

Callers own the meaning of their schemas. Agent Engine, Studio, and application
packages may supply arbitrary JSON Schema, but they must not implement provider
branches or select a provider-specific strictness mode.

## Request contract

Use `StructuredOutputRequest` with the ordinary `context.llm().chat()` path:

```python
from aethergraph.services.llm import StructuredOutputRequest


schema = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["answer", "confidence"],
    "additionalProperties": False,
}

text, usage = await context.llm().chat(
    [{"role": "user", "content": "Answer the question."}],
    structured_output=StructuredOutputRequest(
        name="QuestionAnswer",
        schema=schema,
    ),
)
```

The request contains only a stable name and the caller's canonical schema. AG
detaches the schema from caller-owned mutable data. The returned text is one
canonical JSON value, serialized as a string.

Do not pass `output_format`, `json_schema`, `schema_name`, `strict_schema`,
`validate_json`, or `fail_on_unsupported` with `structured_output`.

## Profile policy

`LLMProfile.structured_output_policy` has two values:

- `best_available` selects the strongest safe mode AG declares for the
  configured provider/model and schema.
- `native_required` rejects the request before transport unless AG can send a
  native provider schema for that provider/model and canonical schema.

The default is `best_available`. A caller such as Agent Engine does not override
this profile policy per request.

The profile schema belongs to AG. Configuration values belong to the host:

- a standalone AG application may load them through normal AG settings;
- Studio persists its complete application-managed profile set in
  `<AG_STUDIO_DATA_DIR>/settings/.env` (normally `.data/settings/.env`) and
  supplies that exact file to Assistant and test workers.

No separate LLM-profile YAML system is required. Project YAML may select a
profile by name, but secrets and provider configuration remain application
settings rather than authored Agent source.

The deterministic environment key is:

```text
AETHERGRAPH_LLM__DEFAULT__STRUCTURED_OUTPUT_POLICY=best_available
```

Named profiles use
`AETHERGRAPH_LLM__PROFILES__<NAME>__STRUCTURED_OUTPUT_POLICY`.

## Capability selection

AG resolves a conservative capability record for the configured provider and
model, then prepares one provider request:

| Effective mode | Provider behavior | AG behavior |
| --- | --- | --- |
| `native_strict` | Native JSON Schema with strict enforcement | Validate again against the canonical schema |
| `native_schema` | Native JSON Schema without strict enforcement | Validate against the canonical schema |
| `json_object` | Provider JSON-object mode | Add bounded schema guidance and validate locally |
| `prompt_json` | Text generation only | Add bounded schema guidance, parse, and validate locally |

The canonical schema never changes. A provider projection is a detached
transport view. For example:

- compatible OpenAI and Azure OpenAI models use native strict mode only when
  every object is explicitly closed and every property is required;
- schemas with optional or free-form object fields use non-strict native schema
  when available;
- Gemini uses its declared JSON Schema subset and falls back when unsupported
  keywords are present;
- DeepSeek uses JSON-object mode plus prompt guidance and canonical local
  validation;
- OpenRouter, LM Studio, Ollama, and unknown endpoints use their declared
  conservative capabilities rather than inheriting an engine policy.

`chat_stream()` remains text-only. It rejects structured-output requests instead
of silently changing their semantics.

## Validation and errors

AG parses exactly one complete JSON value and validates it against the original
canonical schema. Provider enforcement never replaces local canonical
validation.

Structured failures use generic AG exception types:

- `LLMStructuredOutputCapabilityError`
- `LLMStructuredOutputProviderRequestError`
- `LLMStructuredOutputRefusalError`
- `LLMStructuredOutputTruncationError`
- `LLMStructuredOutputParseError`
- `LLMStructuredOutputValidationError`

Only parse and canonical-validation failures describe invalid model output.
Callers may repair those with another model turn. Capability failures, provider
request rejection, refusal, truncation, transport errors, cancellation, and
budget errors should remain authoritative.

## Observability

Generic AG LLM-call records expose the following `request_args` fields for
structured requests:

- `structured_output_policy`
- `structured_output_effective_mode`
- `structured_output_capability_source`
- `structured_output_canonical_schema_fingerprint`
- `structured_output_provider_schema_fingerprint`, when a provider schema exists
- `structured_output_projection_diagnostics`
- `structured_output_validation_outcome`
- `structured_output_response_state`
- `deprecated_parameters`, when the compatibility form was used

`provider_request_args` contains the exact prepared structured-output fragment
merged into the real HTTP request. `compatibility_notes` records projection and
deprecation notes. Studio's generic Observability explorer renders these fields
without adding Assistant-specific provider logic.

## Compatibility window

AG `0.1.x` continues to accept the former keyword form:

```python
text, usage = await context.llm().chat(
    messages,
    output_format="json_schema",
    json_schema=schema,
    schema_name="QuestionAnswer",
    strict_schema=True,
    validate_json=True,
)
```

Use emits `DeprecationWarning` and is recorded in generic observability.
The compatibility keywords are scheduled for removal in AG `0.2.0`.
New code should use `StructuredOutputRequest`.

AG contains no Agent Engine or Studio imports. Engine action schemas and Studio
profile storage remain downstream consumers of this independent AG contract.
