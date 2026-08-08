# Tool discovery transport

AetherGraph owns the provider-neutral transport contract for deferred Tool discovery.
Engine owns catalog search, activation, and callability; provider adapters only encode
and normalize the selected transport mode.

## Ordered request and response contract

`ToolCallRequest.discovery` selects one exact mode:

- `engine_projected` uses ordinary Tool calling. Engine exposes its immediate
  `tool_search` and `tool_load` operations and AetherGraph sends no provider-native
  discovery request.
- `native_client` asks the provider to call an AetherGraph/Engine-managed catalog
  search and returns normalized discovery events.
- `native_hosted` uses a provider-owned search implementation and records only the
  evidence the provider actually supplies.

`ToolCallResponse.items` is the sole ordered response authority. Discovery events and
callable Tool calls stay in provider order; the derived `calls` and `discovery_events`
views never authorize a call independently. Engine consumes discovery events before
decoding later calls in that same sequence.

Every native request binds an exact provider, model or deployment, endpoint family,
mode, protocol version, result-limit behavior, and replay rule. Unsupported bindings
fail before provider traffic. A provider name or model-family prefix never grants a
capability.

The built-in exact native bindings are intentionally narrow:

| Binding | Endpoint | Supported discovery mode |
| --- | --- | --- |
| OpenAI `gpt-5.6` | Responses | `native_client` |
| Azure `gpt-5.5` | Responses | `native_client` |
| Anthropic `claude-sonnet-4-5-20250929` | Messages | `native_hosted`, `native_client` |
| Google `gemini-2.5-pro` | `generateContent` | declared `engine_projected` transport capability |

OpenAI and Azure hosted search are not declared because the implemented contract cannot
yet prove an enforceable result maximum. Applications may bind another exact immutable
capability record only when they own equivalent provider evidence and validation.

## Replay and prompt-cache behavior

Native discovery may return one opaque, cumulative `ToolTransportCheckpoint` for the
provider/model/semantic-turn identity. The latest monotonic revision replaces its
predecessor and is the only replay authority. A reference from another turn, model,
provider, or superseded revision fails closed.

Engine retains the latest checkpoint through same-turn Ledger compaction and releases
it at the new-turn boundary. Stable native catalogs remain separate from the current
active callable surface so loading a deferred Tool does not rewrite catalog identity.
Provider observations record the catalog and surface fingerprints and the selected
mode-specific replay/limit/protocol facts.

`engine_projected` needs no transport checkpoint. It works with any configured provider
binding that already supports ordinary Tool calling; it does not inherit or require one
of the exact native-discovery rows above.

## Structured failures and semantic outcomes

`aethergraph.semantic-event/v2` carries a failed `tool.activity` with the bounded safe
fields `kind`, `code`, `summary`, `retryable`, `details`, `repair_hints`,
`allowed_actions`, and optional opaque `reference`. Only failed Tool activity may carry
this envelope. Private exceptions, tracebacks, credentials, request bodies, and
unrestricted file content remain outside the semantic event.

The same protocol emits one `turn.outcome` from Engine's `agent_outcome`:
`completed`, `failed`, `budget_exhausted`, `paused`, or `cancelled`. Infrastructure run
completion is a separate observability fact and must never be used as a semantic-success
fallback. The closed v1 event types remain importable for legacy integrations but are
not extended with v2 payloads.

