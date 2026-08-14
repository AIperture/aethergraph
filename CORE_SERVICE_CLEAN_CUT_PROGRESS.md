# AG Core Service Clean-Cut Implementation Ledger

Authoritative plan:
`../../others/AETHERGRAPH_CORE_SERVICE_CLEAN_CUT_REFACTOR_PLAN.md` in the suite checkout.

## Baseline

- Started: 2026-08-13
- Branch: `refactor/ag-core-service-clean-cut-20260813`
- Base: `integration` at `6aa355f`
- Worktree: `.worktree/aethergraph-core-service-clean-cut-20260813`
- Main AetherGraph checkout was clean at branch creation.
- `ag-engine` and `ag-studio` are read-only during the initial AG-local phases.

## Preservation boundary

The branch starts after the substantial LLM/runtime-container refactor. Initial
phases must not modify model adapters, model profiles, capability resolution, usage
or quota handling, provider transports, or unrelated model-service construction.
Targeted container edits must be limited to removing the explicitly deleted service
wiring.

## Phase status

| Phase | Status | Evidence |
|---|---|---|
| A0 baseline and isolation | Complete | Dedicated worktree at base `6aa355f`. |
| A1 legacy agents and skills | Complete | Commit `1758fa3`; only `default_chat_agent` remains; focused gate `18 passed`; full AG gate `721 passed, 2 skipped, 2 deselected`. |
| A2 dead leaf services | Complete | Commit `a9f847d`; full AG gate `726 passed, 2 skipped, 2 deselected`; Engine `770 passed`; no causal Studio failure. |
| A3 legacy planning | Complete | Commit `7f6bffc`; full AG gate `729 passed, 2 skipped, 2 deselected`; Engine `770 passed`; Studio container gate `35 passed`. |
| A4 optional capabilities | Complete | Commit `2a89e35`; full AG gate `739 passed, 2 skipped, 2 deselected`; Engine `770 passed`; Studio causal gate `148 passed`; wheel residue clean. |
| A5a public observation boundary | Complete | AG `99d7fc2`, Engine `822324f`, Studio `1af2aa8`; no legacy import aliases; AG `739 passed, 2 skipped, 2 deselected`; Engine `770 passed, 1 non-causal path test deselected`; Studio causal gate `27 passed`; clean wheel has 432 entries. |
| A5b telemetry consolidation | Complete | Commit `5ae889c`; logger and metering moved under `aethergraph.observability`; no-op tracing replaced by persisted operation observations; AG full clean gate passed; Engine `770 passed, 1 deselected`; Studio causal gate `27 passed`; clean wheel has 430 entries and zero legacy telemetry paths. |
| A6 scheduler/resume simplification | Complete | Commit `ec8b433`; deleted unused global scheduler; one local scheduler registry and exactly-once dispatch path; AG `754 passed, 2 skipped, 2 deselected`; Engine `770 passed, 1 deselected`; Studio causal gate `25 passed`; clean wheel has 429 entries. |
| A7 continuation timers | Complete | Commits `a3ec887`, `347c916`; durable SQLite leases, canonical ResumeRouter delivery, lifespan ownership, legacy wakeup deletion; AG `764 passed, 2 skipped, 2 deselected`; Engine `770 passed, 1 deselected`; Studio causal gate `25 passed`; clean wheel has 423 entries. |
| A8 trigger repair | Complete | Commit `d5a82c5`; atomic SQLite occurrence claims/receipts, deterministic run IDs, catch-up policy, timezone recurrence, paginated overlap enforcement, scoped CRUD/event firing, and global route removal; focused gate `19 passed`; AG causal gate `782 passed, 2 skipped, 3 deselected`; Engine `770 passed, 1 deselected`; Studio causal gate `25 passed`; clean wheel has 424 entries. |
| A9 security and admission | Complete | AG commit `9df3c7d`, Studio `66d2442`; canonical `server/admission` limiter and `server/security` credential/redaction boundaries; AG focused gate `93 passed`; full causal gate `793 passed, 2 skipped, 3 deselected`; Engine `770 passed, 1 deselected`; Studio expanded causal gate `46 passed`; clean wheel has 425 entries. |
| A10 documentation, tests, and packaging | Complete | AG `9585d5d`, docs `3e6f4b4`; AG causal gate `794 passed, 2 skipped, 2 deselected`; Engine `770 passed, 1 deselected`; Studio immutable-wheel gate `46 passed`; clean wheel has 425 entries and only `default_chat_agent`. |
| B1-B5 external reconciliation | Complete | Public observation/runtime-output cutovers are committed in Engine `822324f` and Studio `1af2aa8`; Studio security cutover is `66d2442`; no legacy fallback imports remain. |

## Checkpoints

### A1 - legacy agents and skills

- Deleted Graph Builder, explored agents, all bundled legacy skill assets, and the
  graph-builder-only test.
- Removed skill construction from the container, runtime projection, NodeContext,
  startup, and public runtime exports.
- Added a boundary test proving the only bundled Agent source is `chat_agent` and
  that the legacy skill API is absent.
- Focused validation: `18 passed`.
- Complete AG collection: `725` tests. The clean gate passed `721` with `2 skipped`
  and exactly `2 deselected` Host integration tests. Those two reach the canonical
  release guard and fail only because the installed Engine distribution version is
  older than the checked-out Engine source used by the compiler.
- Ruff gate for every changed Python file: passed.
- No LLM implementation file changed. The container edit removes only legacy skill
  import, field, construction, and assignment.

### External A1 causality gate

- `ag-engine`: full read-only suite against the AG worktree passed: `770 passed in
  43.91s`.
- Neither `ag-engine` nor `ag-studio` references the removed skill or bundled-agent
  implementation paths.
- The Studio slice covering AG container construction, worker/supervisor execution,
  Studio-AI hosting, release gates, LLM settings, observability, inspection, and
  deployment collected 117 tests against A1: 111 passed and 6 failed.
- Exactly those 6 tests fail in the same way against the unchanged pre-A1 AG
  baseline. Four terminate while Studio resolves a sandbox-inaccessible Windows
  user path; the local-host and deployment cases are likewise environment/process
  failures. They are not A1 regressions.
- Studio contains no reference to the removed skill registry, runtime skill helper,
  Graph Builder, or explored-agent implementation path. Direct AG container
  consumers passed against A1.
- The stale Studio worker-bridge assertion (`9` expected while Studio defines `10`)
  is an internal pre-existing Studio issue and is excluded from the AG refactor
  causality gate.
- Engine and Studio were tested without bytecode or pytest caches; all temp output
  was directed into this AG worktree. No external source mutation was made.

### A2 - dead leaf services

- Deleted `services/features`, `services/eventbus`, the obsolete `services/kv`,
  `services/redactor`, the unused EventBus contract, and stray `services/__init__.pu`.
- Removed the dead `event_bus` and `redactor` container fields and wiring.
- Preserved `NodeContext.kv()` and its canonical `aethergraph.storage.kv` providers;
  boundary tests assert that the deleted service-level KV package is absent.
- Centralized the existing persistence sanitizer under observability and applied it
  to observation summaries/attributes, prompt captures, agent events, and runtime
  output. Its contract is deliberately narrow: embedded data URLs are redacted and
  binary values become bounded metadata; it does not claim generic secret or PII
  detection.
- Focused AG gate: 69 existing tests passed; 7 clean-cut and persistence-boundary
  tests passed.
- Full AG gate: 730 collected, 726 passed, 2 skipped, and the same 2 non-causal Host
  tests deselected.
- External causal gate: Engine `770 passed`; the 117-test Studio slice produced 111
  passes and exactly the same 6 baseline-proven environment failures. Neither
  external repository imports an A2-removed surface.
- No model adapter, model profile, capability, quota, provider transport, or LLM
  service-construction implementation changed.

### A3 - legacy planning

- Moved `graph_io_to_slots` from the planning service tree into
  `aethergraph.core.graph.io_schema`, the graph schema layer consumed by the input
  schema API.
- Deleted the AG planning service tree and its unused planning contract.
- Removed planner construction and `planner_service` from the container,
  `NodeServices`, and runtime projection; removed `NodeContext.planner()` with no
  compatibility facade or Engine bridge.
- Added graph I/O projection coverage and a negative boundary test proving that the
  legacy planner package and runtime surfaces are absent.
- Focused AG gate: `26 passed`. Full AG gate: 733 collected, 729 passed, 2 skipped,
  and the same 2 non-causal Host tests deselected.
- External causal gate: neither Engine nor Studio imports AG planning; Engine
  `770 passed`, and all 35 Studio suites that directly construct or consume AG
  containers passed.
- No LLM implementation file changed.

### A4 - optional capabilities

- Deleted AG's web-search/page-fetch, local code-execution, MCP, Knowledge/KB, and
  evaluation-harness implementation trees, their dedicated contracts, and tests.
- Deleted the duplicate `plugins/mcp` servers and orphaned
  `plugins/utils/data_io.py`; removed mandatory `pypdf` from AG packaging.
- Removed all corresponding `DefaultContainer`, `RuntimeEnv`, `NodeServices`,
  `NodeContext`, runtime-helper, package-export, configuration, search-factory, and
  Knowledge-specific scope wiring.
- Removed the production `RuntimeEnv` harness override hook instead of retaining a
  no-op or optional compatibility path.
- Reserved the removed first-class capability names in the explicit external-service
  registry so `NodeContext.__getattr__` cannot silently recreate them.
- Added a boundary suite proving the implementations, contracts, fields, accessors,
  runtime helpers, settings, and dynamic extension aliases are absent.
- Added reference-only future-capability blueprints under
  `others/ag-capability-blueprints/`; they define Engine Tool/plugin and Host-provider
  ownership without copied source, stubs, or runtime imports.
- Focused AG gate: `30 passed`. Full AG collection: `743` tests; clean gate `739
  passed`, `2 skipped`, and the same `2` non-causal Host tests deselected.
- Ruff, formatting hooks, `git diff --check`, source forbidden-residue scan, import
  smoke test, and wheel forbidden-residue inspection passed. The built wheel contains
  433 entries and none of the removed paths.
- External causal gate: Engine full suite `770 passed`; Studio's AG-facing runtime,
  Host, Tool-result, and release slice `148 passed`. Neither external source tree
  imports the removed implementations, and neither repository was modified.
- No LLM adapter, model profile, capability-resolution, usage/quota, provider
  transport, or unrelated model-service construction implementation changed.

No compatibility alias or fallback is accepted as completion evidence.

## A5a - coordinated public observation boundary

- Moved the observation facade, policy, storage, translation, redaction, retention,
  logging integration, agent-event emission, and bounded runtime-output capture to
  the public `aethergraph.observability` package.
- Moved `RuntimeInspectionService` and its diagnostic DTOs to
  `aethergraph.core.runtime.inspection`.
- Deleted the old `aethergraph.services.observability`,
  `aethergraph.services.inspect`, and `aethergraph.services.runtime_output` source
  paths. No forwarding modules or re-export aliases remain.
- Added explicit facade reads for suppression scopes, authoritative runs, canonical
  Engine events, and prompt-manifest hydration. Engine no longer reaches through
  `.store`, `.run_store`, or `.engine_event_log` internals.
- Updated Engine and Studio manually in isolated worktrees. The original checkouts
  were not modified.
- Focused AG boundary gate: `64 passed`; public facade follow-up: `2 passed`.
- Full AG clean gate: `739 passed`, `2 skipped`, and the same `2` packaging-only Host
  tests deselected. Engine clean gate: `770 passed` with one worktree-path-coupled
  source-layout assertion deselected. Studio import/inspection/trace/worker causal
  gate: `27 passed`.
- Ruff and Python compilation passed in all three worktrees. Cross-repository import
  smoke passed with AG and Engine worktree sources ordered ahead of Studio.
- A clean wheel rebuild contains 432 entries, includes the new public modules, and
  contains zero files under the three removed service paths.

## A5b - telemetry consolidation

- Moved the logger implementation and `EventLogMeteringService` under the canonical
  `aethergraph.observability` package and updated every AG consumer to that boundary.
- Replaced the installed no-op tracer with `OperationObserver`. Service operations
  now preserve trace/span context and persist bounded request/response summaries,
  allowlisted metrics, lifecycle state, and errors through the existing observation
  facade and SQLite store.
- Reused the canonical observation-store redaction boundary; a persisted operation
  test proves embedded data URLs are redacted before storage.
- Deleted `services/logger`, `services/metering`, and `services/tracing`, including
  the unused logger compatibility helper, `NoopMetering`, `NoopTracer`, tracing
  protocols, container/runtime tracer fields, and runtime tracer helper. No forwarding
  module, no-op provider, or legacy import alias remains.
- Metering absence is now explicit as `None`; guarded consumers record only when a
  real container or context-local metering implementation is configured.
- Focused post-format gate: `31 passed`. Full AG clean gate passed with the two
  packaging-only Host tests deselected. Engine full causal gate: `770 passed` with
  one worktree-path-coupled source-layout assertion deselected. Studio AG-facing
  causal gate: `27 passed`.
- Ruff, Ruff-format, `git diff --check`, public import smoke, forbidden import scan,
  and deleted-module assertions passed. A clean wheel has 430 entries, includes
  `observability/logger`, `observability/metering.py`, and
  `observability/operations.py`, and has zero files under the six removed observation
  and telemetry service paths.
- No Engine or Studio source update was required for A5b; their A5a isolated
  worktrees remain at `822324f` and `1af2aa8`, and the original checkouts remain
  unchanged.

## A6 - scheduler and resume simplification

- Deleted the 701-line `GlobalForwardScheduler`, its unused container construction,
  the `container.schedulers` dictionary, the `RuntimeEnv.schedulers` projection, and
  the unused global-start helper. Every graph run continues to construct exactly one
  `ForwardScheduler`.
- Replaced the scheduler registry's concrete global-scheduler annotation with the
  minimal local control protocol required for resume dispatch and cancellation.
  `RunRegistrationGuard` remains the sole execution-lifetime registration owner; the
  duplicate async registration context manager was deleted.
- Removed `post_resume_event_threadsafe` and the bus's second resume submission.
  Same-loop delivery is awaited directly; cross-thread delivery is posted once to the
  scheduler's owning loop.
- Invalid tokens, missing schedulers, missing loops, and dispatch failures now raise
  explicit errors. The durable continuation is retained until a dispatch succeeds,
  so an inactive run is recoverable instead of being logged and silently lost.
- Repaired scheduler-level cancellation fallback: a handle without an installed
  cancellation adapter now reaches the registered scheduler instead of becoming
  unreachable after the handle labels its adapter kind as `none`.
- Added nine scheduler/resume tests covering same-loop and cross-thread delivery,
  duplicate delivery, invalid tokens, absent scheduler/loop, dispatch failure,
  registration lifetime, cancellation, and the deleted public/container boundary.
- Focused scheduler/resume/runtime gate: `34 passed`. Full AG clean gate: `754
  passed`, `2 skipped`, and the same `2` packaging-only Host tests deselected. Engine
  full causal gate: `770 passed`, `1` worktree-path assertion deselected. Studio
  execution contract/API/worker-bridge causal gate: `25 passed`.
- Ruff, Ruff-format, Python compilation, `git diff --check`, deleted-module import
  smoke, and boundary scans passed. A clean wheel contains 429 entries and no
  `core/execution/global_scheduler.py`.
- No Engine or Studio source update was required. Their isolated A5a worktrees remain
  clean at `822324f` and `1af2aa8`; their original checkouts were not mutated.

## A7 - canonical continuation timers

- Added `ContinuationTimerService` under the runtime continuation layer and a durable
  SQLite lease/receipt store. Stable fire identities derive from run, node, token,
  and scheduled occurrence, so repeated poll waits receive distinct claims without a
  parallel in-memory queue.
- Claims use SQLite `BEGIN IMMEDIATE`, worker identities, lease deadlines, durable
  attempts, retry timestamps, delivery receipts, and dead-letter state. Expired
  leases are atomically reclaimable by a restarted or competing worker.
- Due deadline and polling continuations deliver through the same `ResumeRouter` as
  human/event responses. Token validation and continuation deletion therefore retain
  one canonical path, while delivered receipts suppress duplicate timer fires.
- Added injected-clock scheduling, bounded exponential retry, explicit absent-
  scheduler retry/dead-letter behavior, and canonical operation observations for
  claim, delivery, retry, lease expiry, and dead letter.
- `DefaultContainer` now requires `continuation_timer`; FastAPI lifespan starts and
  stops it explicitly. `wakeup_queue` and its container construction are absent.
- Deleted `ThreadSafeWakeupQueue`, `ScannerProducer`, `WakeWorker`, `WakeupWatcher`,
  the wakeup contract/event, the broken unused continuation factory, the obsolete
  recovery helper calling nonexistent APIs, and two duplicate unused continuation
  store implementations. No alternate timer delivery path remains.
- Added ten timer tests covering deadline and poll payloads, canonical router
  delivery, duplicate workers, worker restart, stale-lease reclamation, durable retry,
  absent scheduler, dead letter, observations, shutdown, and deleted boundaries.
- Focused continuation/scheduler/integration/lifespan gate: `88 passed`. Full AG clean
  gate: `764 passed`, `2 skipped`, and the same `2` packaging-only Host tests
  deselected. Engine full causal gate: `770 passed`, `1` worktree-path assertion
  deselected. Studio execution contract/API/worker-bridge causal gate: `25 passed`.
- Ruff, Ruff-format, Python compilation, `git diff --check`, deleted-module import
  smoke, and source residue scans passed. A clean wheel has 423 entries, includes the
  timer service and lease store, and contains none of the eight deleted wakeup and
  duplicate-store paths.
- No Engine or Studio source update was required; both isolated external worktrees
  remain clean, and their original checkouts were not mutated.

## A8 - trigger repair

- Replaced the non-atomic DocStore trigger scan with one dedicated SQLite trigger
  authority. Trigger definitions, stable occurrence identities, leases, retries,
  skip receipts, delivery receipts, and deterministic run IDs now participate in
  the same durable scheduling model; the legacy `trigger_docstore` path is deleted.
- `claim_due(now, worker_id, lease_until, limit)` uses `BEGIN IMMEDIATE` so competing
  processes cannot claim the same occurrence. Expired leases are reclaimable, and a
  restarted worker deduplicates against the deterministic run record before marking
  the occurrence delivered.
- Trigger schedules advance in the claim transaction. `catch_up_missed=True`
  advances one recurrence at a time; the default policy skips startup misses and
  advances directly to the first future recurrence. The policy also applies to
  one-shots, while one-shots created at the current instant remain due.
- Cron recurrence resolves the configured timezone on every occurrence, including
  daylight-saving transitions. Creation rejects invalid cron expressions,
  timezones, non-positive intervals, negative overlap limits, and missing
  kind-specific configuration.
- Overlap checks now enforce `running >= max_overlap_runs`, fail closed without a
  run store, and paginate every non-terminal status rather than truncating at the
  first 1,000 runs.
- Facade and API reads, cancelation, deletion, and event firing pass explicit tenant
  scope into the service/store boundary. Engine-owned time scans remain deliberately
  all-tenant. The unauthenticated `/triggers/fire-event-global` endpoint is removed.
- Added nineteen tests covering validation, zero-delay one-shots, DST recurrence,
  interval cadence, startup skip/catch-up, atomic competing workers, stale-lease
  restart deduplication, paginated overlap counting, tenant-scoped events and CRUD,
  all-tenant engine scans, facade scope forwarding, and the deleted global route.
- Focused gate: `19 passed`. Full AG causal gate: `782 passed`, `2 skipped`, and `3`
  exact non-causal tests deselected. Engine full causal gate: `770 passed`, `1`
  worktree-path assertion deselected. Studio execution contract/API/worker-bridge
  causal gate: `25 passed`.
- Ruff, Ruff-format, Python compilation, container construction, `git diff --check`,
  deleted-path scans, and wheel inspection passed. The clean wheel contains 424
  entries, includes only `sqlite_trigger_store.py` under trigger storage, and has no
  legacy DocStore trigger implementation or global-fire route.
- No Engine or Studio source update was required. Their isolated worktrees remain at
  `822324f` and `1af2aa8`, and the original checkouts were not mutated.

## A9 - security and admission reconciliation

- Moved the process-local run burst limiter to `aethergraph.server.admission` as
  `RunBurstLimiter`. It now uses an injected monotonic clock and a lock-protected
  sliding window. The container field is exactly `run_burst_limiter`, matching the
  API dependency; the disconnected `rate_limiter` field and service tree are gone.
- Added real-container HTTP tests for both demo and cloud modes proving the second
  request receives HTTP 429 at a configured one-request burst boundary. Rate-limit
  configuration now rejects non-positive concurrency, window, and burst values.
- Moved the synchronous exact-name `SecretStore` contract and
  `EnvironmentSecretStore` implementation under `aethergraph.server.security`.
  Chat, embedding, image-generation, and hot-reconfiguration paths all consume that
  one contract. Deleted the inconsistent async `services/secrets` protocol and its
  synchronous implementation, and removed the container's node-visible `secrets`
  field.
- The model credential selector remains at its stable model-service boundary but
  delegates every environment lookup to `EnvironmentSecretStore`; provider
  precedence and behavior are unchanged. Studio now imports the canonical store and
  deletes its private duplicate adapter in isolated commit `66d2442`.
- Authentication signing material is resolved before operational store creation.
  Demo and cloud startup reject missing `auth.secret`; the fixed development secret
  is available only in explicit local mode.
- Consolidated observation persistence redaction and UI credential masking under
  `aethergraph.server.security.redaction`. The boundary now removes keyed credentials,
  bearer values, credential assignments, embedded data URLs, and binary payloads.
  Deleted `aethergraph.observability.redaction` with no forwarding module.
- Added eleven security/admission tests covering atomic windows, real HTTP 429s,
  container and NodeServices boundaries, startup secret policy, synchronous
  credential resolution, credential/data persistence redaction, removed modules,
  and rate configuration validation.
- Focused AG security/model/profile gate: `93 passed`. Full AG causal gate: `793
  passed`, `2 skipped`, and `3` exact non-causal tests deselected. Engine full causal
  gate: `770 passed`, `1` worktree-path assertion deselected. Studio application
  settings plus execution causal gate: `46 passed`.
- Ruff, Ruff-format, Python compilation, `git diff --check`, source residue scans,
  container construction, and wheel inspection passed. The clean wheel contains 425
  entries, includes only the new admission/security modules, and contains none of
  the three deleted service/redaction paths.
- Engine required no source change. Studio is clean at `66d2442`; the original
  Engine and Studio checkouts were not mutated.

## A10 - documentation, tests, and packaging cleanup

- Deleted the final commented-out graph test invocation. Retained the default-agent
  and removed-skill boundary tests, which confirm that `default_chat_agent` is the
  only bundled Agent and that the legacy skill runtime surface is absent.
- Reconciled the separate documentation worktree in commit `3e6f4b4`. Deleted the
  planning, skills, MCP, and knowledge Context reference pages plus the obsolete MCP
  and RAG tutorials. Rewrote the service, DI, scheduling, server, Context, runtime,
  KV, artifact, and memory pages around the final core/Engine/plugin ownership.
- Documentation navigation and relative-link validation has zero missing targets;
  all 113 remaining mkdocstrings API targets resolve against the exact AG source;
  the current docs contain zero forbidden legacy service/accessor references.
- The clean AG wheel contains 425 entries, no removed service paths, and exactly
  `aethergraph/plugins/agents/chat_agent/default_chat_agent.py` under bundled Agent
  sources. An isolated installed-wheel import smoke passed.
- Final AG causal gate: `794 passed`, `2 skipped`, and the two immutable-release Host
  tests deselected because they require installed distribution metadata. Engine:
  `770 passed`, `1` worktree-path assertion deselected. Studio: `46 passed` against
  freshly built immutable AG and Engine wheels, including its subprocess release
  probe.
- Engine and Studio worktrees are clean at their existing cutover commits. Their
  original checkouts and the original docs checkout were not mutated.
