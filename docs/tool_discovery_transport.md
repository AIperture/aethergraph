# Tool discovery transport and observability

This note describes AetherGraph's provider-neutral Tool discovery boundary. AG does
not know Engine activation leases, Studio Timeline rows, or application Tool policy.
It transports the exact discovery mode selected by its caller and records what the
provider request and response actually contained.

## Hosted and client-executed search

| AG mode | Inventory owner | Provider exchange | When definitions become available |
| --- | --- | --- | --- |
| `native_hosted` | The request already contains the deferred inventory. | The provider searches it and returns the selected definitions in the same response. | During that provider response. |
| `native_client` | The application owns the inventory and authorization decision. | The model returns a search call; the application returns a correlated search output with selected definitions. | In the continuation request after the search call. |
| `engine_projected` | The application performs ordinary deterministic search. | No provider-native discovery protocol is used. | When the application includes ordinary callable definitions in a later request. |

OpenAI's current Responses contract calls the hosted execution value `server` and
the client execution value `client`. Hosted search can omit `execution`; client mode
requires the application to echo the search `call_id` in `tool_search_output`.
Provider adapters own those wire names. The public AG contract continues to use the
provider-neutral `native_hosted` and `native_client` modes.

For `native_client`, the caller owns both the strict search input schema and bounded
`search_instructions`. AetherGraph places those instructions before its generic
transport description and includes non-empty instructions in the request fingerprint.
This lets an Engine communicate authorization boundaries such as required paths and
activation semantics without moving policy ownership into a provider adapter. Empty
instructions retain the prior request-fingerprint input for compatibility.

Official protocol reference:
[OpenAI Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search).

## LLM-call and Timeline meaning

A search or load is not a synthetic LLM call. One real provider request produces one
LLM-call observation:

- its Tool surface records the exact active, searchable, and immediate definitions
  dispatched with that request;
- ordered request items record prior Tool results or client search output;
- ordered response items record model messages, Tool calls, and native discovery
  items in provider order; and
- lifecycle status is persisted as `in_progress`, then atomically finished as
  `completed`, `failed`, or `cancelled`.

For client search, the search call belongs to the first LLM call and the loaded
surface belongs to the continuation call. For hosted search, search and selected
definitions can appear in one provider response. An Engine may project linked
search/activation domain events into its Timeline, but those events remain separate
from AG's provider-neutral LLM ledger.

After the adapter constructs the final provider body, AG records a bounded
`provider_tool_projection` alongside the caller-owned canonical Tool surface. The
projection contains the provider family, exact declaration count and names,
selection controls, and a deterministic fingerprint of the actual outgoing Tool
array. The canonical catalog and provider projection are deliberately separate: an
adapter may translate or temporarily inject declarations, and cache analysis must
describe the body that was actually sent rather than infer it from the catalog.

AetherGraph transports caller-owned `trace_context` as bounded generic observation
data and never interprets it as Ledger or repair authority. AG Engine's consumer
contract for Ledger compilation, exchange-root recovery, Tool closure, and cache
interpretation is documented in
[Ledger compilation, errors, and repair](https://github.com/AIperture/ag-engine/blob/main/docs/19_ledger_compilation_error_and_repair.md).

## Continuation purpose and Tool-result replay

`ToolTransportCheckpoint` is provider replay state, not Tool-discovery or execution
authority. Its public `purpose` is one of:

- `pending_discovery_result`: the next request must return a client-executed search
  result containing the exact selected Tool names and the correlated provider
  reference;
- `pending_tool_outputs`: the next request must return the exact results for the
  provider call identities stored in the opaque adapter payload; or
- `consumed`: replay responsibility has ended and a caller must not retain the
  checkpoint as pending work.

Callers may use the safe purpose, provider, model, contract version, turn, and revision
to manage lifecycle. They must not inspect `opaque_payload`. The adapter validates the
purpose against its private payload before provider traffic.

The discovery result is mandatory for both successful and failed client-search
continuations. Provider adapters do not infer selection by subtracting the prior
active-name set from the current set: that projection may change for independent
application policy reasons. A completed result names the exact newly selected Tools;
a failed result carries the typed discovery error and no Tool names.

OpenAI Responses client discovery has a verified failed/no-new-result continuation:
AG sends the correlated `tool_search_output` with client execution, incomplete
status, and no selected Tools while retaining `previous_response_id`. Azure Responses
does not inherit that conclusion and remains explicitly unsupported until its own
wire contract is verified. An unsupported adapter result is a typed capability
outcome for the caller; it is not permission for AG to invent a provider payload.

Checkpoint declaration provenance distinguishes Tools declared in the current root
from Tools injected by the correlated search output. The adapter may avoid duplicating
the injected definitions in that exact search-result request. It must preserve Tools
owned by the root on later Tool-output continuations and on a new semantic-turn root.

Every continuation request resends the request-owned Tool declarations and selection
controls. A response identifier or replayed assistant content preserves conversation
items; it does not imply that the provider retained request-scoped Tools. OpenAI and
Azure Responses replay exact function-call IDs with matching `function_call_output`
items. Anthropic Messages replays the exact assistant content blocks and appends user
`tool_result` blocks with matching `tool_use_id` values. OpenAI-compatible Chat
Completions replays the exact assistant Tool-call message and matching Tool messages.

Adapter lifecycle tests must continue through search, selected Tool, returned Tool
result, and a following decision. A mock must not return a Tool call that was absent
from the outgoing request it received.

## Prompt caching is diagnostic

Tool search is designed to append loaded Tools at the end of model context so a
provider can preserve the preceding cacheable prefix. Cache admission is still a
provider outcome, not an AG correctness contract. Production code must not change
search policy, retry, or synthesize a cache hit based on observed usage.

The opt-in Luna diagnostics under `tests/live/` send and report:

1. a client-executed search and an identical replay;
2. the first client activation continuation and an identical replay; and
3. a hosted search and an identical replay.

They require successful, complete, non-truncated responses, stable request-body
fingerprints for each replay pair, and internally consistent usage counters. They log
input, output, cache-read, and cache-write tokens, but deliberately do not require a
cache hit.

Run the raw provider control and AG adapter parity checks with:

```powershell
$env:AG_RUN_OPENAI_CACHE_SMOKE = "1"
$env:AG_OPENAI_CACHE_SMOKE_MODEL = "gpt-5.6-luna"
$env:OPENAI_API_KEY = "<credential>"
python -m pytest tests/live/test_openai_tool_search_cache_raw.py `
  tests/live/test_openai_tool_search_cache_client.py -q -s -o log_cli=true
```

On an HTTP failure, the raw control logs the phase label, model, credential-free
request fingerprint, status code, and a bounded provider response before re-raising.
AG transport and observation failures likewise propagate after logging; the
diagnostic never converts a provider error into zero usage or a passing cache result.

Exact post-adapter request facts are terminal observation metadata, not request
identity. The immutable `provider_request_args` recorded before dispatch must not be
mutated when OpenAI or Azure Responses returns the exact Tool projection. The
projection is retained as `provider_request_facts.tool_projection`; usage, response,
latency, attempts, and those facts are committed together at observation finish.
This separation is provider-neutral: any future adapter may report bounded request
facts without changing the identity validated between begin and finish.

### Sanitized Luna measurement — 2026-08-20

The corrected diagnostic completed against `gpt-5.6-luna`. Each replay pair had an
identical credential-free request fingerprint. The observed usage was:

| Raw Responses phase | Input | Output | Cache read | Cache write |
| --- | ---: | ---: | ---: | ---: |
| Client search | 3,529 | 50 | 0 | 3,526 |
| Client search replay | 3,529 | 63 | 3,526 | 0 |
| Client activation | 3,710 | 126 | 0 | 3,707 |
| Client activation replay | 3,710 | 119 | 3,707 | 0 |
| Hosted search | 5,571 | 62 | 0 | 5,375 |
| Hosted search replay | 5,573 | 64 | 5,375 | 0 |

| AG adapter phase | Input | Output | Cache read | Cache write |
| --- | ---: | ---: | ---: | ---: |
| Client search | 3,570 | 57 | 0 | 3,567 |
| Client search replay | 3,570 | 53 | 3,567 | 0 |
| Client activation | 3,717 | 127 | 0 | 3,714 |
| Client activation replay | 3,717 | 127 | 3,714 | 0 |

This run observed cache admission on every exact replay. It is evidence about that
provider run, not a production guarantee or a reason to add a cache-dependent path.
