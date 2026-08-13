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
| A2-A10 | Paused | Not started; stopped at the external compatibility gate. |
| B1-B5 external reconciliation | Blocked for decision | Engine passes, but a Studio test expects worker protocol `9` while Studio's own `BRIDGE_VERSION` is `10`; no external repository mutation is authorized. |

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

### External A1 compatibility gate

- `ag-engine`: full read-only suite against the AG worktree passed: `770 passed in
  43.91s`.
- Neither `ag-engine` nor `ag-studio` references the removed skill or bundled-agent
  implementation paths.
- `ag-studio` backend: the first-failure diagnostic stopped after `120 passed` at
  `tests/test_context_api.py::test_compatibility_reports_product_and_protocol_contracts`.
  Studio's test expects worker protocol `9`; Studio's own
  `execution.worker.bridge.BRIDGE_VERSION` is `10`.
- The broader Studio backend run reached 50% and displayed multiple failures before
  its 10-minute cap. Further AG phases are paused pending an explicit Engine/Studio
  reconciliation decision.
- Engine and Studio were tested without bytecode or pytest caches; all temp output
  was directed into this AG worktree. No external source mutation was made.

No compatibility alias or fallback is accepted as completion evidence.
