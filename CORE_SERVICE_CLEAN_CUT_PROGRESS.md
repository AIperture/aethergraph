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
| A5-A10 | Pending | Not started. |
| B1-B5 external reconciliation | Pending | No external repository mutation authorized; stop if a later phase creates a causal external failure. |

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
