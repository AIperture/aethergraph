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

