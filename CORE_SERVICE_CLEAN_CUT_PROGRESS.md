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
| A2-A10 | Pending | Not started; A1 external causality gate is complete. |
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

No compatibility alias or fallback is accepted as completion evidence.
