# Storage provider migration S0 evidence

Status: complete  
Captured: 2026-08-14/15 UTC  
AG base: `integration` at `14d8ef3808730b66496222b534349c36e4b1a5ca`  
Implementation branch: `refactor/storage-provider-migration-20260814`

This document freezes the pre-provider inventory, query shapes, compatibility audit,
correctness gates, and local performance baseline required by S0. It is evidence, not
the target architecture. No provider implementation or storage deletion preceded it.

## Repository and prerequisite audit

The complete migration plan, Plan 1 completion record, repository `AGENTS.md`, and
Metalens `AGENTS.md` were read before edits. The suite root is a repository container,
not a Git repository; each project was inspected independently. Local branches,
worktree heads, remote heads (`git ls-remote --heads origin`), tracking divergence,
and dirty state were enumerated. The required base branches had not advanced; some
unrelated remote branches had.

| Repository | Checked-out branch/head | Tracking/dirty evidence | Disposition |
|---|---|---|---|
| `aethergraph` | `integration` / `14d8ef3` | Clean; six commits ahead of `origin/integration` (`c75f1fd`) | New isolated worktree created from this exact reviewed head. |
| `ag-engine` | `integration` / `fb29c2b` | Clean; one commit ahead of `origin/integration` (`54b8bea`) | Compatibility consumer only. |
| `ag-studio` | `electron-app` / `3348851` | Clean; remote `electron-app` at `2380d0f`; existing tracking points to `origin/control-redesign` | Production source is frozen behind Plan 1. |
| `others/ag-metalens` | `main` / `d2b7708` | Matches `origin/main`; pre-existing changes in `agents/main.yaml`, `tools/apply_config_patch.py`, `tools/inspect_parameter_space.py`, `tools/run_feasibility.py`, and `.data/` | User changes preserved and excluded from edits. |

The retired Plan 1 worktrees were also inspected. Their tracked state is clean; two
contain inaccessible pytest temporary directories already recorded by Plan 1. They
were not used as migration bases.

The new AG worktree is:

```text
C:\Users\zcliu\Documents\Github\aethergraph-suite\.worktrees\aethergraph-storage-provider-migration-20260814
```

## Construction and ownership inventory

### Runtime composition

`services/container/default_container.py` is the main construction point. It creates
the following persistence resources independently and then exposes many of them as
container fields:

| Area | Current construction/implementation | Container/service consumer | S0 classification |
|---|---|---|---|
| Runtime event log | `build_event_log` -> SQLite/FS/none | `eventlog`, channels, triggers, integration, semantic events | Adapt to provider event store. |
| Memory persistence | `build_memory_persistence`; event-log mode opens a second service-named event DB | `memory_factory`, Agent/Engine plan and state history | Adapt, then replace with canonical memory/event records. |
| Memory hot log | `build_memory_hotlog` over ephemeral KV | Memory facade | Adapt; keep explicitly ephemeral behavior. |
| Observations/LLM/logs | direct `SQLiteObservationStore` | `observability`, metering, inspect APIs | Adapt to provider observations. |
| Runs/results | `build_run_store`, `build_run_result_store`; SQLite stores share `runs/runs.db` via distinct handles | `run_manager`, observability reader, APIs | Adapt to transactional provider control stores. |
| Sessions | `build_session_store` | run/session APIs and artifact counters | Adapt to provider control store. |
| Continuations | `build_continuation_store` over independently built doc/KV/event stores | resume router, integration resolver | Replace with one transactional continuation store. |
| Timer leases | direct `SQLiteContinuationTimerLeaseStore` | continuation timer | Replace with provider lease store. |
| Graph state | `build_graph_state_store` over doc/event storage | graph scheduler/runtime | Replace with shared canonical state primitive. |
| Artifacts | `build_artifact_store` plus `build_artifact_index` | ArtifactFacade, channel resources, run/session counters | Replace with content/occurrence/lineage protocols. |
| Search | `build_search_backend` plus legacy `build_vector_index` path | `GlobalIndices`, Memory/corpus search | Consolidate to one explicit search backend. |
| KV/documents | `build_doc_store`; ephemeral `kv_hot` | registry and supporting services | Adapt to provider supporting stores. |
| Registry | registration doc store over general doc store | App/agent/graph gallery and runtime registry | Adapt; retain App identity semantics. |
| Triggers | direct `SQLiteTriggerStore` | trigger engine/service | Adapt to provider control store. |
| Auth grants/invites | two direct `SQLiteKVSync` instances with `grant:` and `invite:` prefixes | auth services | Previously omitted from plan; now explicitly provider-owned. |
| Integration idempotency/bindings | `services/integration/factory.py` directly creates `SQLiteIngressIdempotencyStore` and `SQLiteExternalSessionBindingStore` | Host ingress coordinator | Previously omitted from plan; now explicitly provider-owned. |
| Semantic events | adapter over general EventLog | public Runtime/Host integration | Adapt to provider event protocol without changing Studio. |
| Runtime output sink | installed on container dynamically | runtime output capture | Give explicit provider/bundle disposition in S7. |

`DefaultContainer` exposes `cont_store`, `state_store`, `trigger_store`, `doc_store`,
`kv_hot`, artifacts/index, registration stores, `eventlog`, memory/search facades,
observability, run/result/session stores, and dynamically installed Host/runtime-output
objects. `ext_services` is an untyped extension dictionary. The target bundle must be
composition-only and must not reproduce this container as a second service locator.

### Protocol inventory

Storage-facing protocols currently live in two namespaces:

- `contracts/storage`: artifact index/store, async KV, blob, document, event log,
  lexical index, search backend, trigger store, vector index;
- `contracts/services`: artifacts, continuations, KV, memory, runs, sessions, state
  stores, triggers, runtime output, and metering-facing contracts.

Duplications and signature conflicts that S1/S6 must resolve:

- `contracts/storage.AsyncKV` and `contracts/services.AsyncKV` disagree on bulk and
  scan signatures;
- artifact store/index contracts exist in both service and storage namespaces;
- `VectorIndex` exists in `contracts/storage/vector_index.py` and independently in
  `contracts/services/memory.py`;
- run protocol listing omits concrete user/org/session filters;
- several public lists are unbounded or offset-paginated, and records mix dicts with
  mutable dataclasses.

No protocol is deleted based on namespace alone. S1 owns the canonical contract map.

### Implementations and factories

Active implementation families are:

- event logs: SQLite and filesystem, plus sync SQLite core;
- documents: SQLite and filesystem, plus sync SQLite core;
- KV: SQLite, in-memory, layered, plus sync SQLite core;
- runs/results/sessions: SQLite, document/filesystem, and in-memory;
- graph state: document plus event composition;
- continuations: KV/document, filesystem, in-memory, and separate SQLite leases;
- artifacts: filesystem/S3 content and SQLite/JSONL metadata indexes;
- search: generic backend, null backend, SQLite vector, SQLite lexical, FAISS, Chroma,
  and copied/vanilla SQLite vector variants;
- observations: SQLite; triggers: SQLite; integration idempotency/bindings: SQLite.

The active builders are `storage/factory.py`, `storage/search_factory.py`, the direct
constructors in `DefaultContainer`, `services/integration/factory.py`, and the
historical observability opener in `observability/facade.py`.
`services/memory/factory.py` contains a commented standalone SQLite example only; it
is preserved as an unrelated placeholder until its owning documentation/code cleanup.

### Lifecycle inventory

- `EmbeddedRuntime.close()` dynamically probes only selected resources (`eventlog`,
  run/session/state stores) and cannot close the full persistence graph coherently.
- `ObservabilityFacade` and `SQLiteObservationStore` have close behavior, but the
  runtime does not consistently own it through one boundary.
- several SQLite doc/KV/run/artifact implementations retain connections without a
  public async close operation; some event/search/observation operations instead open
  a new connection per call.
- `open_observability_workspace` constructs concrete read-only SQLite readers using
  hard-coded current layout paths.

The provider bundle therefore owns one idempotent close, health, and maintenance
boundary. Services must not probe or close individual provider stores.

## Settings, environment, and physical paths

`AppSettings` uses `env_prefix="AETHERGRAPH_"` and
`env_nested_delimiter="__"`. Every nested field below is therefore independently
addressable by a generated environment key such as
`AETHERGRAPH_STORAGE__RUNS__SQLITE_PATH`; direct keys also include
`AETHERGRAPH_WORKSPACE` and `AETHERGRAPH_ENV_FILE`.

Current storage setting families:

- `storage.docs`: backend, `docs/doc_store.db`, `docs/doc_store`;
- `storage.eventlog`: backend, `events/events.db`, `events`;
- `storage.kv`: backend, `kv/kv_store.db`, prefix;
- `storage.artifacts`: FS `artifacts`; S3 bucket/prefix and
  `./.aethergraph_tmp/artifacts` staging;
- `storage.artifact_index`: `artifacts/index.sqlite`,
  `artifacts/index.jsonl`, optional occurrence JSONL;
- `storage.graph_state`: `graph_state/graph_state.db`, `graph_state`;
- `storage.continuation`: backend, namespace, secret, KV/doc/event/FS paths under
  `continuations/`;
- `storage.vector_index`: SQLite/FAISS/Chroma selectors and paths under
  `vector_index/`;
- `storage.memory`: FS `mem`, URI prefix, hotlog/index TTL and limits;
- `storage.runs`: memory/FS/SQLite and `runs/runs.db`;
- `storage.sessions`: memory/FS/SQLite and `sessions/sessions.db`;
- `search`: none/SQLite vector/FAISS/SQLite lexical selectors, vector/FAISS paths,
  lexical `search/sqlite_lexical/index.sqlite`, and lexical-enable flag;
- `observability`: `events/observability.db` and retention policy.

There is a second `AppSettings.cont` model (`fs`/`inmem`, root and secret) separate
from `AppSettings.storage.continuation`, which is the model used by the current
builder. The storage model's description references
`AETHERGRAPH_CONT__SECRET_KEY`, while its actual nested key is
`AETHERGRAPH_STORAGE__CONTINUATION__SECRET_KEY`. S2 must remove this ambiguity in one
cut, not alias both paths.

Hard-coded paths outside those models are:

- `memory_events/events.db` (historical reader) and the service-name-derived memory
  event path used during composition;
- `continuations/timer_leases.db`;
- `auth/auth_kv.db`;
- `triggers/triggers.db`;
- `integration/operations.db`;
- historical reader paths `events/events.db`, `events/observability.db`, and
  `runs/runs.db`.

S2 replaces the per-store environment/path forest with one provider selection/config
model. No old variable is read as a fallback.

## Query shapes and indexes

| Domain | Actual frequent query/write shapes | Existing index/transaction behavior | Baseline gap |
|---|---|---|---|
| Memory/events | Single append; recent scope reads; kind/tag/scope/time and many identity filters; cursor IDs available | Single-column scope/kind/time and selected identity/time indexes; tag join; scope+kind+ID index | Multi-filter shapes can miss composite indexes; current handles contend heavily; `get_many` and Memory view helpers may scan/filter in Python. |
| State | Latest snapshot by scope/kind/`state:<key>` tag; compare expected revision then append | `BEGIN IMMEDIATE`; state revision parsed from event JSON; tag join and reverse event scan | Correct CAS but no indexed current-state row; history and current lookup share append log. |
| Graph state | Snapshot by doc key; append history; enumerate runs | Snapshot document key lookup; event log for history | `list_run_ids` scans every document ID. |
| Continuations | Save/get `(run,node)`, token resolve, correlator resolve, last-open/list waits, timer due/claim | Document and token KV writes are separate; timer index `(status,next_attempt_at,lease_until)` and `BEGIN IMMEDIATE` claim | Non-atomic save/index/event; list/last-open/waits scan all docs; separate lease DB/connection lifecycle. |
| Runs/results | Create/get/update status, filtered recent list, artifact/result metadata update | Primary key; graph/status/user/org/session + started-time indexes; WAL; JSON read-modify-write | Offset pagination; combined filters may use one index; run/result handles share a file without shared transactions; counters/status use read-modify-write. |
| Sessions | Create/get; user/org/kind listing; artifact metadata updates | Primary key and separate user/org/kind indexes; JSON plus promoted fields | Offset pagination and read-modify-write counters. |
| Observations | Append; trace/log pages; LLM pages and attempt hydration; retention/resource lookups | Indexes for occurred/category/run/session/trace/LLM call; write transaction for LLM record | Tenant/project/org/user/agent/graph/node/app combinations can scan; offset pages; LLM page uses correlated attempt aggregates; resource listing is unbounded. |
| Artifacts | Content upsert/dedup; occurrence append; run/session occurrence pages; labels/metrics ranking | Artifact PK; occurrence run/session + creation indexes | Duplicate occurrence fields; JSON label `LIKE`; metric/tag selection loads candidates in Python; offset pagination. |
| Search | Per-item embed/vector/FTS upsert; structural, semantic, lexical, hybrid queries with metadata/time | Corpus+scope/user/org/kind/time indexes; brute-force/optional FAISS; FTS5 | Missing promoted indexes for client/session/run/graph/node/source; per-operation connections; silent lexical/hybrid/null fallback; configured lexical filename becomes a directory containing `lexical.sqlite`. |
| Integration | Idempotent ingress claim/complete and external-session binding lookup | Dedicated SQLite operations DB | Direct Host factory construction bypasses planned provider bundle. |
| Auth | Grant/invite KV get/set/expiry | Two prefixes over direct SQLite KV sync handle | Direct container construction and hidden physical path. |

The benchmark captured these concrete `EXPLAIN QUERY PLAN` results:

- run polling uses the run primary-key index;
- graph run paging uses `idx_runs_graph_started`;
- observation trace paging uses `ix_observations_trace`;
- artifact occurrence paging uses `idx_occ_run_created`;
- vector scope search uses `idx_emb_corpus_scope_time`;
- state latest uses `idx_events_scope_kind_id` plus the event-tag unique index;
- memory recent scope reads choose `idx_events_scope`, not an ordering-covering index.

## `app_id` audit

`rg` found `app_id` in 71 AG source/test files. No `application_id` alias exists.
The occurrences split into distinct semantics:

1. **App/gallery identity to retain:** App API, graph decorators/builders, registration
   services/stores. This remains optional compatibility identity and is not
   `StorageScope`.
2. **Propagated runtime metadata to deprecate:** run requests/records, NodeContext and
   scope objects, channels, continuations, memory, artifacts, visualization,
   metering, observability, triggers, and runtime integration DTOs.
3. **Persisted columns/filters to remove from new schemas:** observations and trace
   management, artifact metadata, run JSON, continuation documents, memory/event
   payloads and filters.
4. **Public compatibility schemas:** run API `appId`, trigger/runtime/observability
   request contracts, artifact/memory DTOs. These require consistent deprecation
   metadata without hot-path warnings.
5. **Noncanonical partitioning:** the LLM prompt cache currently includes `app_id` in
   `_CACHE_SCOPE_KEYS`; S1/S4 must replace that partition dependency with canonical
   scope.

Cross-package results:

- Engine source: no `app_id` occurrence;
- Metalens source: no `app_id` occurrence;
- Studio source: one optional field in `ui/src/lib/observabilityTypes.ts`; no provider
  or path dependency.

Canonical provider contracts and schemas must contain no `app_id`. Retained public
fields receive: "Deprecated; retained for compatibility and scheduled for removal in
a future breaking release."

## Correctness and boundary baseline

All test runs used the isolated AG source through `PYTHONPATH`, a unique explicit
pytest base directory, and disabled pytest cache because the machine's default stale
pytest temporary root is ACL-inaccessible.

| Scope | Result |
|---|---|
| AG focused storage/runtime suite: memory ordering/API, Agent state, graph state, runs, continuation/default timer, integration operational stores, observations, artifact occurrences, runtime isolation, embedded runtime | `69 passed in 18.20s`; one Starlette deprecation warning. |
| Engine observability source against isolated AG | `1 passed in 1.33s`. |
| Studio production integration-boundary release audit against isolated AG + Engine | `1 passed`; one Starlette deprecation warning. |

The initial source-only attempts without the combined `PYTHONPATH` failed during
collection because the checkouts were not installed; these were environment setup
failures, not product failures, and the corrected commands above passed.

Studio production source was not edited. The Plan 1 boundary remains sufficient, so
there is no critical prerequisite blocker.

## Performance baseline

Reproduction command:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
python scripts/storage_provider_s0_baseline.py --samples 100 --contenders 8
```

Environment: Windows 11 `10.0.26200`, Intel Family 6 Model 183, Python 3.13.1.
Peak Python memory reported by `tracemalloc`: 582,822 bytes. Generated storage was
1,236,992 data bytes, 4,185,952 WAL bytes, and 65,536 SHM bytes. The observation
store alone reported 344,064 database bytes and 4,185,952 WAL bytes before close.

| Workload | p50 ms | p95 ms | p99 ms | Throughput ops/s | Errors/notes |
|---|---:|---:|---:|---:|---|
| Memory append, 4 writers | 33.193 | 98.721 | 2321.808 | 26.040 | 399/400 succeeded; one `OperationalError`. |
| Memory recent read, 2 readers | 1.549 | 701.687 | 4010.115 | 13.053 | 200 succeeded; max 4653.654 ms. |
| State CAS, 8 independent handles | 34.018 | 84.708 | 84.708 | 94.063 | Correct: one winner, seven typed conflicts. |
| Continuation save | 10.940 | 11.413 | 11.938 | 91.000 | No errors; document/token writes are non-atomic. |
| Continuation token resolve | 0.442 | 0.620 | 0.791 | 2133.456 | No errors. |
| Run create | 0.421 | 0.681 | 1.774 | 2060.500 | No errors. |
| Run status poll | 0.228 | 0.388 | 1.608 | 3449.995 | No errors. |
| Run filtered offset page | 0.565 | 1.015 | 1.823 | 1785.086 | No errors. |
| Run status update | 0.293 | 0.429 | 0.712 | 3096.570 | No errors. |
| Observation trace page | 0.675 | 1.469 | 4.821 | 1117.987 | No errors. |
| Observation log page | 1.309 | 1.659 | 3.540 | 871.651 | No errors. |
| Observation LLM page | 2.151 | 2.439 | 4.757 | 576.494 | No errors. |
| Artifact occurrence page | 0.243 | 1.302 | 1.453 | 2041.229 | No errors. |
| Search indexing | 24.334 | 31.686 | 53.629 | 35.656 | Per-item vector + FTS upsert; no errors. |
| Semantic search | 1.132 | 1.210 | 1.788 | 855.795 | 100 indexed items, 25 queries. |
| Lexical search | 1.083 | 1.271 | 2.711 | 863.057 | 100 indexed items, 25 queries. |
| Hybrid search | 2.362 | 2.526 | 2.783 | 419.138 | 100 indexed items, 25 queries. |

Current stores expose no direct busy-wait duration, so lock errors and tail latency
are the S0 lock-wait proxies. The very high EventLog tail and one lock error are an
explicit target for the local provider; they do not justify adding a fallback.

## Frozen deletion/disposition manifest

Deletion occurs only after the named replacement and gates exist.

### Adapt before removal

- `storage/factory.py` store builders and direct `DefaultContainer` constructors;
- current SQLite/FS/in-memory run, session, event, continuation, state, artifact,
  observation, trigger, integration, auth, document, and KV implementations;
- `open_observability_workspace` concrete layout reader;
- existing service facades and `NodeContext` entrypoints, preserving their public
  semantics while changing injected stores.

### Superseded after clean cut

- all per-store storage settings/environment mappings and direct physical path joins;
- method-name probing/fallback in Agent state and runtime lifecycle;
- KV/document/event composite continuation persistence and separate lease DB;
- duplicated artifact contracts, MIME alias, occurrence metadata duplication, JSONL
  metadata path, and service-owned schema migrations;
- legacy `StorageSettings.vector_index`, independent `build_vector_index`, duplicate
  `VectorIndex`, copied/vanilla SQLite vector modules, and silent null/mode fallbacks;
- old runtime workspace opener/layout recognition after the new manifest opener and
  explicit history retirement are verified;
- direct integration and auth SQLite construction after provider stores are wired.

### Preserve unless separately proven superseded

- App/gallery registration semantics and optional deprecated `app_id` compatibility
  surfaces;
- graph authoring, LLM/provider, channel, visualization, and simulation abstractions
  that do not own persistence;
- Metalens project source and `project-data`;
- Studio catalog/settings/project data and both completed Plan 1 integration modules;
- commented/example placeholders whose owning plans are unrelated.

FAISS/Chroma and optional external artifact implementations receive an explicit S6/S8
capability disposition; they are not deleted merely because SQLite is the required
local provider.

## Frozen external-provider conformance scenarios

The shared conformance suite must cover:

1. exact explicit registration; duplicate and unknown names fail;
2. invalid config, secret resolution failure, missing capability, health failure, and
   open failure propagate directly with no local fallback or local database creation;
3. one immutable bundle per open, provider identity/schema/capabilities available,
   and idempotent close called exactly once;
4. read/write and read-only modes enforced, including unsupported format versions;
5. canonical tenant/workspace/project scope isolation with no `app_id` dependency;
6. bounded stable cursor pagination and documented ordering for every list/query;
7. ordered/bulk event append, idempotency, restart cursors, and exact reads;
8. state CAS race/stale revision/history ordering and atomic outbox behavior;
9. continuation token/correlator/resume plus lease claim/renew/release atomicity;
10. run/result/session transitions and artifact counters transactionally consistent;
11. observation/LLM/log/resource hydration, retention, and pagination;
12. artifact staged commit, content deduplication, occurrence/lineage integrity, range
    read, and orphan cleanup after metadata failure;
13. structural/semantic/lexical/hybrid capability reporting, indexing freshness, and
    typed unsupported-mode failures;
14. integration ingress idempotency, external-session binding, auth grant/invite, KV,
    documents, registry, triggers, semantic events, and runtime-output sinks;
15. cancellation and injected failures leave no partial authoritative state.

## S0 exit decision

S0 is complete. The inventory changed the plan in two places—Host integration stores
and auth grant/invite persistence—but found no critical prerequisite defect and no
need to change Studio production code. S1 may begin from this evidence. Clean-cut,
no-fallback, no-dual-read/write, and deprecated-optional-only `app_id` rules remain
unchanged.
