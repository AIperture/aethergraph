# AG Authoritative Commit and Runtime Outcome Alignment Plan

Status: implementation in progress; A0-A7 complete in working trees; A8 qualification active  
Prepared: 2026-08-16  
Reviewed and amended: 2026-08-16  
Primary implementation branch: `refactor/storage-provider-migration-20260814`  
Primary worktree: `.worktrees/aethergraph-storage-provider-migration-20260814`  
Primary AG head at preparation: `092a4ee95c8acfca2623ab8022c80f4d4734cf4a`  
Decision: correct the existing Engine implementation; do not create a parallel Engine  
Merge policy: do not merge the AG migration worktree until every required AG, Engine,
and Studio boundary change in this plan is qualified together  
Migration policy: clean cut; one authoritative provider; no fallback, dual write,
dual read, legacy history opener, compatibility implementation, shadow path, or
duplicated execution path

## 0. Cold-start handoff

This document is intentionally self-contained. It is written for an implementation
agent that may not have access to the discussion or live debugging session that
produced it.

Read these sources completely before editing:

1. repository `AGENTS.md` at the suite root;
2. `others/AETHERGRAPH_STORAGE_PROVIDER_MIGRATION_PLAN.md`, including its complete
   implementation ledger and post-S10 correction;
3. `others/AG_STUDIO_AETHERGRAPH_RUNTIME_API_ISOLATION_PLAN.md`, including its
   completion record;
4. this plan;
5. the current source and tests named in section 4 rather than trusting the line
   numbers or implementation sketches in this document.

The storage-provider migration itself is complete through S10 plus the live
observability-reader correction at AG commit `092a4ee`. This follow-up exists because
a real Studio + Engine + Metalens run found several cross-package semantic mismatches
that package-local tests did not detect. Do not reopen the provider migration or
reintroduce any retired storage implementation. Correct the boundary contracts on
top of the completed provider architecture.

### 0.1 Repository state recorded during planning

| Repository | Branch | Head | State and role |
|---|---|---|---|
| AG migration worktree | `refactor/storage-provider-migration-20260814` | `092a4ee95c8acfca2623ab8022c80f4d4734cf4a` | Primary implementation and cross-package evidence ledger. |
| Main `aethergraph` checkout | `integration` | `37367515d8e83d131ae2e91996e5d2749bf124c1` | Clean at planning time; do not implement this follow-up here. |
| `ag-engine` | `integration` | `85c19c92f42b3c849fb28ab54cf42bee5eb06595` | Clean at planning time; existing ReAct v3 is the adoption target. |
| `ag-studio` | `electron-app` | `695ea64` | Clean isolated implementation base verified on 2026-08-16. This supersedes the review-time `d92a4cb` base with UI-only Model Settings/readiness/status presentation work; preserve that work and include its Timeline/status tests in qualification. |
| `others/ag-metalens` | `main` | `d2b7708a9a2475656eaf002df810dfec5f673a27` | Acceptance consumer; preserve the pre-existing dirty files listed below. |

The Metalens checkout contained pre-existing user changes at planning time:

- `agents/main.yaml`
- `tools/apply_config_patch.py`
- `tools/inspect_parameter_space.py`
- `tools/run_feasibility.py`
- `.data/`

Do not edit, clean, reset, copy over, or include those paths in a commit. Metalens is
read-only unless the user separately authorizes an agent-source change.

The AG worktree also contains sandbox-owned temporary pytest directories that may be
unreadable to the normal user. They are not product source and must not be treated as
migration input or removed without resolving their exact ownership and scope.

### 0.2 Mandatory first-turn actions for the implementing agent

Before implementation:

1. verify branches, heads, remotes, worktrees, and dirty state for all four
   repositories;
2. stop if any current head has moved in a way that changes the contracts described
   here;
3. create dedicated Engine and Studio worktrees before their adoption phases; do not
   edit their primary checkouts;
4. preserve the current AG migration worktree and add this follow-up to its milestone
   ledger;
5. reproduce or re-read the recorded live evidence using public/read-only inspection;
6. run the focused baselines in section 9 before changing a contract;
7. resolve and record the exact interpreter, installed distribution metadata, and
   wheel hashes before treating any test result as qualification evidence;
8. update section 11 with evidence, commits, tests, and deviations after every
   milestone.

## 1. Architectural decision

This work does **not** justify a new Engine implementation, ReAct v4, or a parallel
agent loop.

The failures are below and around agent reasoning:

- authoritative storage commit versus derived search projection;
- typed error propagation across AG and Engine;
- historical observability event projection;
- transport status versus semantic agent outcome;
- complete operation-profile pinning;
- artifact commit bookkeeping;
- deprecated compatibility-metadata hygiene.

Creating a new Engine would duplicate the current ReAct implementation while leaving
the AG and Studio boundary contracts unresolved. The existing Engine owns semantic
event and agent-outcome interpretation and must adopt the corrected AG contract.

The implementation is therefore one coordinated correction with three ownership
layers:

| Owner | Responsibility |
|---|---|
| AG | Authoritative durability, derived projection state, provider-neutral receipts, historical event shape, artifact bookkeeping, compatibility-metadata hygiene. |
| Engine | Consume authoritative receipts correctly, retain error fidelity, publish outputs only after authoritative commit, and keep ReAct v3 semantics. |
| Studio | Preserve the completed runtime integration boundary while distinguishing transport completion from agent outcome and pinning every required model-operation binding. |

### 1.1 Implementation discipline

Apply these rules to every milestone:

1. Keep one authoritative implementation path. Do not add fallback, compatibility,
   shadow, dual-read, dual-write, or duplicated retry/publication paths.
2. Inventory existing contracts, receipts, outboxes, transaction helpers, profile
   resolvers, digest helpers, test fixtures, and scripts before adding a new one.
3. Centralize genuinely identical behavior in the narrowest existing owner. Group
   related helpers and scripts by responsibility instead of scattering per-call-site
   variants across AG, Engine, and Studio.
4. Do not create a generic framework when one focused extension to an existing
   contract is sufficient. Any new abstraction must remove duplication, preserve a
   clear owner, and be exercised by at least two real call sites.
5. Prefer extending an existing exact semantic type over creating parallel Memory,
   Artifact, Engine, or Studio receipt types with overlapping meanings.
6. Tests must use the production public path and shared boundary fixtures wherever
   possible. Do not maintain a second test-only behavior path.

## 2. Live failure evidence

### 2.1 Exact logical run

The first complete live run after the post-S10 observability-reader correction used:

- Studio test run: `tr_9394a4d7ff4142ea8790b1ad89727277`
- Studio test session: `ts_96c958f3800747e79f4f947701e30257`
- AG session: `studio-600650304720459faa13bb230954d5c6`
- AG root run: `run-b91386c67b17`
- Engine turn: `f1d9b92a7c74`
- assistant output: `assistant_output_e27ea385a60da5a06f692caadef305ff`
- start: `2026-08-16 22:25:20.626010 UTC`
- finish: `2026-08-16 22:25:30.050860 UTC`

Studio catalog evidence is in:

`C:\Users\zcliu\AppData\Local\AIperture\AGMetalensStorageTest\studio-data-52196b7\studio.db`

The runtime workspace is under the same `studio-data-52196b7/test-runtime/local`
tree and has a valid provider manifest selecting `local.sqlite`, format version 1.
The observed `clean_shutdown=false` value was expected because Studio and its worker
were still live during read-only inspection; it was not evidence of corruption.

### 2.2 Causal sequence

The observed causal sequence was:

1. LM Studio chat completed successfully using provider `lmstudio` and model
   `gemma-4-e4b-it`.
2. Engine produced the full assistant text and appended
   `agent_engine.assistant_output` with the output identity above.
3. AG durably committed that Engine event to the authoritative EventStore.
4. AG then attempted the derived Memory search projection through the configured
   default embedding profile.
5. That profile used provider `openai`, model `text-embedding-3-small`, endpoint
   `openai_embeddings`.
6. OpenAI returned a no-credits error.
7. AG surfaced the search projection exception after the authoritative commit.
8. Engine caught the exception, returned an empty `EngineEventAppendReceipt`, and
   raised `RuntimeError: Assistant output ... could not be persisted.`
9. Engine emitted a failed `agent_outcome` with code `unhandled_runtime_error`.
10. The enclosing AG graph function returned that outcome normally, so AG recorded
    the root run as `succeeded`.
11. Studio copied the AG transport status, stored the failed Engine outcome, and
    emitted the lifecycle label `Test succeeded`.

The durable Engine event proves that the error text was false: authoritative
persistence succeeded; the derived search projection failed.

### 2.3 Engine event evidence

The root run contained, in order:

1. `agent_engine.user_request`
2. `agent_engine.agent_entered`
3. `agent_engine.assistant_output`
4. `agent_engine.runtime_error`
5. `agent_engine.agent_exited`
6. `agent_engine.run_outcome`

The assistant-output event contained the successful LM Studio response. The final
outcome was failed. AG and Studio nevertheless stored their transport-level run
statuses as succeeded.

### 2.4 Exact package used

The corrected AG wheel built from `092a4ee` had SHA-256:

`310B51B062724671BC099F5EAF472AC327E6773FEE41AC62D8B9C6D3368DD47C`

Do not confuse it with the earlier wheel built at the same distribution version.
Studio's environment inspector hashes wheel `RECORD` evidence, so exact content—not
only the version string—must be used for qualification.

## 3. Required invariants

Every implementation choice must preserve all of these invariants.

### 3.1 Authoritative write invariants

1. A write is authoritative when its canonical record and idempotency identity are
   committed by the selected provider.
2. A derived search/index projection is not evidence of authoritative durability.
3. A caller must never be told that an authoritative record was absent when it was
   committed.
4. Exact retry must not duplicate events, artifacts, occurrences, counters, semantic
   delivery, or user-visible output.
5. Search freshness remains explicit through an opaque indexed cursor or equivalent
   covering position.
6. Projection failure remains visible and diagnosable; it is not swallowed.
7. Projection retry uses the same explicitly selected provider and operation binding.
   It is not a fallback.
8. A provider that cannot support the required atomic intent/receipt semantics must
   fail readiness before publication.
9. Every retryable Engine-authored event uses a stable caller-owned idempotency
   identity. AG must not generate a new identity for an exact retry.
10. API success, retry, and recovery use the same authoritative implementation; no
    alternate append or recovery path may bypass the selected provider contract.

### 3.2 Engine invariants

1. ReAct v3 remains the single current Engine implementation.
2. An assistant message may be published only after authoritative event commit.
3. A failed derived projection does not retract or misreport an authoritative event.
4. Engine must retain the underlying safe error classification and phase.
5. Cursor/state advancement that depends on durable existence must use authoritative
   durability, not search-index completion.
6. No private AG store, provider path, SQLite table, or container field may cross the
   public Engine boundary.
7. Crash/restart duplicate prevention must be durable. A process-local published set
   may optimize a turn but is not proof of exact-once behavior.

### 3.3 Studio invariants

1. Studio remains behind `StudioAetherGraphRuntime` and the completed execution
   worker/runtime bridge.
2. Studio must not learn provider paths, tables, store handles, or continuation
   tokens.
3. AG transport completion and Engine semantic outcome remain separately inspectable.
4. The user-facing overall status must not say succeeded when the Engine outcome is
   a runtime failure.
5. Existing durable raw output evidence remains readable even when Studio cannot
   project a newer Engine outcome schema.
6. Every operation binding used by the worker is pinned or derived from one verified
   immutable settings snapshot; the v2 path must not trust self-consistent command
   digests without rechecking the selected settings source.

### 3.4 Migration invariants

1. No legacy workspace opener or history migration.
2. No local fallback when an external or explicitly selected provider fails.
3. No dual read, dual write, shadow index, compatibility provider, or temporary
   legacy facade.
4. `app_id` remains only explicitly deprecated optional compatibility metadata. It
   must not become provider scope, an index key, authorization, or a sentinel value.
5. `client_id` must not reappear as canonical Memory or artifact ownership.
6. Local providers and every actual qualified external provider must pass the same
   semantics. The deterministic external fake must pass the same contract
   conformance suite but is not real-service qualification.
7. A deterministic external conformance fake proves contract shape only. It must not
   be described as qualification evidence for a real external service.
8. No milestone may introduce parallel helpers, scripts, receipt types, or status
   projectors when an existing owner can be extended without semantic ambiguity.

## 4. Confirmed mismatches and source map

Line numbers are evidence anchors only. Re-read current source before editing.

### 4.1 Memory commit versus search projection

AG `CanonicalMemoryFacade.append_many()` currently commits authoritative events,
updates the hot cache, and then performs `SearchBackend.upsert_many()`. Search failure
is deliberately visible, but the raised exception carries no committed receipt to
the public caller.

Relevant AG sources:

- `src/aethergraph/services/memory/canonical_facade.py`
- `src/aethergraph/services/memory/canonical_public.py`
- `src/aethergraph/services/memory/canonical_factory.py`
- `src/aethergraph/storage/contracts/stores.py`
- `src/aethergraph/storage/providers/local_sqlite/event_store.py`
- `src/aethergraph/storage/providers/local_sqlite/search_backend.py`
- `tests/test_canonical_memory_facade.py`

The existing test
`test_canonical_memory_search_failure_is_visible_after_authoritative_commit` proves
the partial state but does not test any consumer's interpretation of it.

### 4.2 Engine error erasure

Engine `append_agent_engine_event()` catches every exception from
`memory.append_event()` and returns an empty receipt. `append_react_event()` catches
again. `ResponseOutputCoordinator.publish_response()` interprets the empty receipt as
authoritative failure and raises the generic output-persistence error.

Relevant Engine sources:

- `src/aethergraph_engine/_internal/events/append.py`
- `src/aethergraph_engine/_internal/events/models.py`
- `src/aethergraph_engine/_internal/engines/react_loop_v3/events.py`
- `src/aethergraph_engine/_internal/engines/react_loop_v3/core/response_output.py`
- `src/aethergraph_engine/_internal/engines/react_loop_v3/core/external_observations.py`
- `src/aethergraph_engine/_internal/foundation/memory_utils.py`

The same false-negative receipt can affect external-observation cursor advancement,
Tool-discovery evidence, causal links, and other call sites that gate state mutation
on `bool(receipt)`.

### 4.3 Historical Engine-event envelope

Public Memory writes a canonical storage payload shaped like:

```json
{
  "data": {
    "turn_id": "...",
    "agent_instance_id": "...",
    "dispatch_token": "..."
  }
}
```

AG's historical observability facade currently returns the entire stored payload as
Engine `data`, yielding:

```json
{
  "data": {
    "data": {
      "turn_id": "...",
      "agent_instance_id": "...",
      "dispatch_token": "..."
    }
  }
}
```

Engine expects those fields directly under its `data` mapping.

Relevant sources:

- AG `src/aethergraph/observability/workspace.py`
- AG `src/aethergraph/services/memory/canonical_public.py`
- AG `tests/test_observability_workspace.py`
- Engine `src/aethergraph_engine/observability/source.py`
- Engine `src/aethergraph_engine/observability/membership.py`
- Engine `src/aethergraph_engine/observability/timeline.py`
- Engine `src/aethergraph_engine/observability/service.py`
- Engine `tests/test_observability_source.py`

The current AG test inserts a flat `EventDraft` directly into the provider, bypassing
public Memory's envelope. The current Engine test uses a hand-built flat fake. Both
tests pass while the real path is incompatible.

This mismatch can hide or corrupt:

- Agent instance identity;
- child dispatch and return membership;
- dispatch tokens;
- causal parent links;
- Tool call/result correlation;
- prompt-manifest and LLM-call correlation;
- resource-slot projections;
- multi-Agent timelines.

### 4.4 Layered status mismatch

AG run status represents graph transport/execution. Engine `agent_outcome` represents
semantic Agent completion. Studio currently maps AG `succeeded` directly to Studio
`succeeded` and labels it `Test succeeded`, even when the stored Engine outcome is
failed or a runtime error. Simulation uses the Studio status to count successful
items.

Relevant Studio sources:

- `backend/src/ag_studio_backend/services/test_runtime_service.py`
- `backend/src/ag_studio_backend/domain/test_runtime_schemas.py`
- `backend/src/ag_studio_backend/services/simulation_service.py`
- `backend/src/ag_studio_backend/domain/simulation_schemas.py`
- `backend/tests/test_test_runtime_outcome_contract.py`
- `backend/tests/test_simulation_api.py`

Raw output preservation is correct and must remain. The missing piece is a total,
explicit projection of transport status, Agent outcome, and user-visible overall
status.

### 4.5 Incomplete operation-profile snapshot

Studio resolves and pins one chat profile for the test runtime. It hashes the complete
application settings file, but the execution snapshot does not present the effective
embedding or image-generation operation bindings. The worker replaces
`config.llm.default` with the selected chat profile and leaves `config.embed.default`
independent. AG's default embedding profile is OpenAI
`text-embedding-3-small` unless explicitly changed.

Relevant sources:

- Studio `backend/src/ag_studio_backend/services/execution_config_service.py`
- Studio `backend/src/ag_studio_backend/domain/execution_config_schemas.py`
- Studio `backend/src/ag_studio_backend/execution/worker/runtime.py`
- Studio `backend/src/ag_studio_backend/services/application_settings_service.py`
- AG `src/aethergraph/config/llm.py`
- AG `src/aethergraph/services/container/default_container.py`

Static profile/capability validity can be checked before execution. Live quota or
credit availability is temporally unstable and cannot replace the authoritative
commit fix.

### 4.6 Artifact partial commit and bookkeeping

`CanonicalArtifactFacade` stores the blob, artifact record, and occurrence before
search indexing, but run/session artifact counters are updated after the search
projection. A search failure therefore leaves durable artifact state and an error,
with counter advancement deferred until an exact retry.

Relevant AG sources:

- `src/aethergraph/services/artifacts/canonical_facade.py`
- `src/aethergraph/services/artifacts/canonical_public.py`
- `src/aethergraph/services/artifacts/canonical_factory.py`
- `tests/test_canonical_artifact_facade.py`

The existing
`test_canonical_artifact_search_failure_is_visible_after_durable_records` confirms
the partial state. It must be extended to cover authoritative receipts, counters,
exact retry, and consumer-visible behavior.

### 4.7 Deprecated identity sentinel leakage

AG's text formatter supplies `"-"` for missing context fields. The observation log
handler persists every truthy scope value. Handler ordering can therefore persist
`app_id="-"` or `client_id="-"`, contradicting the explicit optional-compatibility
metadata rule.

Relevant AG sources:

- `src/aethergraph/observability/logger/formatters.py`
- `src/aethergraph/observability/logging.py`
- `src/aethergraph/observability/canonical_service.py`
- `src/aethergraph/observability/canonical_inspection.py`

### 4.8 Missing stable Engine-event idempotency identity

Public Memory currently creates a new event ID inside `append_event()`. Engine has a
stable assistant `output_id`, but does not supply a stable event ID to AG. Engine's
`ResponseOutputCoordinator` also keeps published output identities only in process
memory. A crash after authoritative commit but before semantic publication can
therefore retry with a new event ID and duplicate the durable event or output.

Relevant sources:

- AG `src/aethergraph/services/memory/canonical_public.py`
- Engine `src/aethergraph_engine/_internal/events/append.py`
- Engine `src/aethergraph_engine/_internal/engines/react_loop_v3/core/response_output.py`

The final contract must accept one stable caller-owned event identity for retryable
Engine events. Engine must derive it deterministically from the existing canonical
output/event identity rather than creating a second idempotency registry.

### 4.9 Provider transaction boundary is not yet sufficient

Current AG composition exposes `EventStore`, `ArtifactRepository`, and
`SearchBackend` as separate focused repositories. It does not expose a provider-owned
operation that atomically commits an authoritative record and its projection intent.
The preferred A1 design is therefore not implementable by merely rearranging facade
calls.

Relevant sources:

- `src/aethergraph/storage/contracts/provider.py`
- `src/aethergraph/storage/contracts/stores.py`
- `src/aethergraph/storage/providers/local_sqlite/provider.py`
- `tests/storage_conformance/external_provider.py`

A0 must choose a minimal provider-owned composite operation or explicitly revise the
atomic-intent invariant. Do not add a generic transaction framework, expose raw
connections, or simulate atomicity with unrelated writes.

### 4.10 Studio v2 operation snapshot can drift

Studio's v2 worker path verifies that the command identity and runtime-profile
snapshot contain the same application-settings digest, but it does not recompute the
current settings-file digest on that path. The worker then loads live settings,
materializes the pinned chat profile, and leaves embedding and image-generation
profiles from the live file. Non-chat operation bindings can therefore change after
session pinning without being detected.

Relevant Studio sources:

- `backend/src/ag_studio_backend/services/execution_config_service.py`
- `backend/src/ag_studio_backend/domain/execution_config_schemas.py`
- `backend/src/ag_studio_backend/execution/worker/runtime.py`

The corrected worker must verify the source digest for every supported snapshot
version and materialize the complete required operation set through one centralized
snapshot/resolution path.

## 5. Target architecture

### 5.1 Authoritative commit receipt

Define one provider-neutral result model that distinguishes at least:

- authoritative identity and cursor;
- whether the authoritative record was newly committed or already present;
- derived projection intent identity or target cursor;
- projection state: pending, indexed, or failed;
- sanitized typed projection diagnostic when failed;
- enough information for exact idempotent retry without exposing a provider path or
  physical row identity.

The final name should follow the current contract vocabulary. Possible names in this
plan, such as `AuthoritativeCommitReceipt` or `ProjectionCommitReceipt`, are
descriptive placeholders, not an instruction to add duplicate receipt types.

Inventory existing receipt and outbox records first. Reuse or generalize an existing
canonical primitive when its semantics are exact. Do not create parallel Memory-only
and Artifact-only implementations when one focused projection-intent contract can
serve both safely.

The inventory must begin with `MemoryCommitReceipt`, `ArtifactCommitReceipt`,
`EngineEventAppendReceipt`, existing producer outboxes, leases, and provider
transaction helpers. Extend one of them when its ownership and semantics remain
exact. A new shared receipt is justified only if both Memory and Artifact use the
same state machine in production; shared field names alone are insufficient.

The public API decision must also be explicit. Public Memory currently returns an
`Event`, so returning a different type is a reviewed breaking contract, not an
incidental internal refactor. Inventory all suite consumers—including Engine,
Metalens, simulations, examples, docs, and unowned consumers discoverable in the
workspace—before choosing an additive committed-result method, a versioned breaking
return contract, or a public typed exception carrying the committed receipt.

### 5.2 Projection intent and retry

Preferred end state:

1. the authoritative record and a durable projection intent are committed as one
   provider-owned operation;
2. the write returns authoritative success once that operation commits;
3. synchronous projection may complete the intent immediately;
4. projection failure updates the same intent with a typed diagnostic;
5. bounded retry targets only the same selected SearchBackend and exact document;
6. a dead or repeatedly failing intent remains visible in health/inspection;
7. exact-search callers can require a covering indexed cursor and fail closed when
   it is not reached;
8. ordinary authoritative event consumers do not misclassify the write as absent.

This is not fallback. There is one selected provider bundle and one selected search
projection. Retrying the exact projection intent is continuation of the same write.

If current provider contracts cannot atomically persist the authoritative record and
projection intent, stop in milestone A0 and document the transaction-boundary gap.
Do not simulate atomicity with two unrelated writes and call it complete.

If a provider-owned composite operation is required, add the smallest focused
operation to the existing storage contract and bundle. Do not introduce a generic
unit-of-work layer, raw transaction callback API, facade-owned transaction manager,
or parallel provider path. The local implementation and external conformance fake
must exercise the exact same public operation. A real external provider, when one is
available, requires separate qualification evidence beyond the fake.

### 5.3 Public Memory behavior

Public `Memory.append_event()` and `append_chat_turn()` must retain their documented
event identity and idempotency behavior. Retryable writes must also accept a stable
caller-owned event identity rather than generating a new identity on every attempt.
Their final contract must make these facts
unambiguous:

- authoritative event durability;
- derived search readiness;
- projection degradation.
- caller idempotency identity and whether the record was newly committed or already
  present.

Do not silently discard projection errors. Expose them through a typed receipt,
explicit diagnostic channel, or typed exception carrying the committed receipt,
depending on the final public API decision. Whichever form is chosen, Engine must be
able to distinguish `committed + projection failed` from `not committed` without
inspecting private AG types.

### 5.4 Engine publication behavior

`ResponseOutputCoordinator` must:

1. require authoritative event durability;
2. derive and pass one stable AG event identity from the canonical Engine output
   identity;
3. use durable identity/receipt evidence as the crash-safe duplicate boundary;
4. add the output identity to its local published set only as a turn-local
   optimization after authoritative success;
5. emit the semantic assistant message exactly once;
6. retain a visible projection-degraded diagnostic when indexing failed;
7. raise only when authoritative persistence failed;
8. never republish or duplicate after exact retry/resumption.

Other Engine receipt-gated call sites must be audited and classified. Do not
mechanically replace every truth check. For each caller, state whether it requires:

- authoritative existence;
- indexed search readiness;
- resource-link indexing;
- best-effort diagnostic persistence.

Review `resource_links_indexed` specifically. It currently reflects the presence of
resource links on a successful append rather than proof of search indexing. Rename
or redefine it through the existing receipt model so recorded resource relations and
search readiness cannot be confused.

### 5.5 Historical observability projection

The AG observability facade is the correct normalization boundary. It must project a
stable Engine event mapping from canonical storage without exposing the public
Memory storage envelope.

For Engine-tagged events written through public Memory:

- validate the exact canonical envelope;
- project the inner authored `data` mapping as Engine `data`;
- retain top-level event identity, time, run/session scope, kind, text, and tags;
- reject malformed canonical envelopes directly;
- do not add a legacy flat/nested fallback heuristic.

Tests must write through the real public Memory facade, close/reopen the manifested
workspace, and then pass the result through the real Engine reader.

### 5.6 Layered status model

Use explicit names rather than overloading `succeeded`:

- transport/execution status: whether AG admitted and executed the graph;
- Agent outcome: Engine's completed, paused, cancelled, failed, budget-exhausted,
  or runtime-error classification;
- workflow outcome: domain workflow completion where supplied;
- overall/user status: a deterministic projection of the layers above.

The exact Studio schema migration must be decided in its adoption milestone. At
minimum:

- transport failure always yields overall failure;
- transport success plus failed/runtime-error Agent outcome yields Agent failure;
- transport success plus paused outcome yields a resumable/waiting state;
- transport success plus completed Agent outcome yields success;
- absent or unprojectable Agent outcome remains explicit and does not invent success;
- lifecycle labels say `completed`, `agent failed`, or an equally accurate phrase,
  never `Test succeeded` for the observed failure;
- simulations do not count an Agent failure as a successful item.

Do not make Engine throw solely to force AG run status to failed. AG currently returns
durable output only for successful graph runs; converting semantic failure into an
unhandled graph exception would lose the typed outcome and recreate ambiguity.

### 5.7 Operation-profile identity

Studio execution snapshots must include credential-free identities for every
operation binding required by the runtime:

- selected chat binding;
- `embedding_default` binding when canonical search projection requires embeddings;
- image-generation binding when the authored Agent or enabled Tools require it.

Include effective endpoint, provider, model, adapter/catalog revision, capability
facts, and stable digests without secrets. Preserve the complete application-settings
digest as the immutable file identity.

The worker must recompute and compare that file identity for every supported snapshot
version before loading any operation. It must then materialize all required operation
bindings from the pinned snapshot through one centralized resolver. Do not pin chat
while reading embedding or image generation from a later live file.

Preflight should validate configuration and advertised capability. Runtime failures
must identify the exact operation/profile/provider/model safely. Do not perform an
implicit provider substitution when a binding is unavailable.

LLM qualification must enable AG's existing LLM logging/observability whenever the
selected provider and endpoint advertise that capability. Use the strongest supported
credential-safe capture mode needed by the scenario, record the effective capture
mode, and preserve provider request/response correlation without logging secrets. If
the provider does not support the required logging capability, record that fact as an
explicit capability result; do not switch providers or create an alternate logging
path.

### 5.8 Artifact authority

Artifact authoritative success includes the content/blob identity, artifact record,
and occurrence. Run/session bookkeeping must either:

- commit atomically with that authoritative occurrence; or
- be a deterministic provider-owned projection whose pending/failed state is
  represented and retried exactly.

Search indexing remains a separate derived projection. A failed index must not make
the system claim that the artifact does not exist.

Failure semantics must be decided for every current boundary: blob stage, artifact
record, occurrence, retention, search intent/index, run counter, and session counter.
For blob stores that cannot share the metadata transaction, use the existing orphan
reconciliation ownership or one focused staged-blob protocol; do not claim impossible
cross-service atomicity or add a second artifact write path.

## 6. Implementation milestones

Milestone labels use `A` to distinguish this follow-up from storage migration S0-S10.

### A0 — Reverification and contract decision

Status: complete

Deliverables:

1. verified branches, heads, dirty state, and isolated Engine/Studio worktrees;
2. an updated Studio base decision for current clean head `695ea64`, including the
   Observability Timeline tests added after the original plan base;
3. a suite-wide call-site inventory for every AG Memory/Artifact public write and
   receipt consumer, including Engine, Studio, Metalens, simulations, examples, docs,
   and discoverable unowned consumers;
4. a transaction-capability audit for local providers, the deterministic external
   conformance fake, and each actual qualified external provider as distinct evidence
   classes;
5. an inventory of existing outbox/lease/receipt records, idempotency helpers,
   transaction helpers, profile resolvers, digest helpers, boundary fixtures, and
   scripts that could implement the projection-intent contract without duplication;
6. a stable caller-owned Engine-event identity design covering crash/restart;
7. updated correctness baselines for all confirmed mismatch areas;
8. exact interpreter, dependency-extra, distribution `RECORD`, and wheel-hash
   evidence for each baseline;
9. an architecture decision recorded in this document before implementation.

Required decision questions:

- Can authoritative record plus projection intent be committed in one provider
  transaction for every qualified provider?
- Does the provider contract need one focused composite operation, and if so, what is
  the smallest extension that avoids a generic transaction abstraction?
- Is one generalized projection-intent contract appropriate, or do Memory and
  Artifact transactions require focused typed operations over a shared model?
- Which existing receipt or outbox type can be extended without overlapping
  semantics, and what exact evidence would justify a new shared type?
- Which public method returns the discriminated receipt, and is that additive or an
  explicitly versioned breaking change from the current `Event` return?
- How does Engine deterministically derive the AG event ID from its existing output
  or event identity?
- How are pending/failed projection intents drained and inspected?
- Which Engine call sites require only authoritative durability and which require
  index readiness?
- What exact schema version will Studio use for layered status and operation-profile
  snapshots?
- How does the Studio worker verify and materialize the complete chat, embedding, and
  image operation set from one immutable source?

Stop and discuss if atomic projection intent cannot be expressed through the current
provider contract without weakening external-provider qualification.

Implemented decision (2026-08-16):

- Memory uses one focused `EventStore.append_many_with_search_intents` operation.
  The local provider commits the authoritative event and one deterministic
  `memory-search:<event-id>` intent in the same events-database transaction; ordinary
  runtime event appends remain unchanged.
- The existing EventStore is extended instead of introducing a provider-wide
  transaction coordinator, generic outbox framework, second Memory facade, or
  fallback SearchBackend.
- `CanonicalPublicMemoryFacade.append_event_commit()` is additive and returns an
  explicit authoritative/projection receipt. Existing `append_event()` remains the
  simple Event-returning API and raises `MemoryProjectionError` with that receipt when
  search degrades.
- Engine event IDs are deterministic SHA-256 identities of the normalized canonical
  Engine payload. Assistant semantic delivery continues to use the existing stable
  `output_id`/channel upsert key; no process-local recovery route was added.
- Artifact authority remains a separate focused decision because ArtifactRepository,
  RunRepository, and SessionRepository need one control-database production commit;
  it is not forced through the Memory intent operation.
- Studio schema keeps persisted transport status authoritative while response `status`
  becomes the centralized overall projection and `transport_status` exposes the
  supervision layer explicitly. Runtime profile snapshot v3 pins chat, embedding, and
  image-generation identities without credentials.

### A1 — AG authoritative Memory commit semantics

Status: complete

Scope:

- canonical storage records/protocols required for projection intent;
- local provider transaction and exact external-provider conformance fake;
- canonical Memory facade and public Memory projection;
- bounded retry/health/inspection for pending or failed intents;
- no Engine or Studio production change yet.

Acceptance:

1. authoritative event plus projection intent is atomic;
2. injected embedding failure leaves one authoritative event and one visible failed
   or pending projection intent;
3. caller can distinguish authoritative success from projection status;
4. exact retry indexes the same event without duplication;
5. `require_indexed_cursor` remains strict;
6. provider close/reopen preserves pending work and idempotency;
7. local and deterministic-external contract conformance pass, with real external
   provider qualification recorded separately when available;
8. stable caller-owned event identity survives close/reopen and exact retry;
9. two retries with the same identity create one event, one intent, one semantic
   delivery identity, and one covering cursor progression;
10. the implementation extends existing focused contracts where possible and adds no
    fallback, duplicate append path, generic transaction framework, or legacy path.

### A2 — Existing Engine receipt adoption and error fidelity

Status: complete

Create/use an isolated Engine worktree from the verified current Engine head.

Scope:

- existing ReAct v3 event append path;
- `EngineEventAppendReceipt` or its reviewed replacement;
- assistant output publication;
- external-observation cursors and every other receipt-gated call site;
- typed sanitized diagnostic projection;
- no new Engine implementation.

Acceptance:

1. committed event plus failed search projection publishes the assistant response
   exactly once and returns a projection-degraded diagnostic;
2. uncommitted authoritative event prevents publication and produces the real typed
   persistence error;
3. no blanket `except Exception: return empty receipt` remains on the critical path;
4. exact retry/resumption does not duplicate semantic delivery or output;
5. Tool, decision, external-observation, and run-outcome call sites use the correct
   durability level;
6. assistant and other retryable Engine events pass a deterministic stable AG event
   identity derived from an existing canonical Engine identity;
7. crash after AG commit but before semantic publication resumes without a duplicate
   event or assistant message;
8. recorded resource links and search-index readiness use distinct, accurately named
   receipt fields;
9. ReAct v3 remains the sole implementation and no parallel publication/recovery
   path exists.

### A3 — AG historical Engine-event projection

Status: complete

This AG phase may proceed before or after A2 once the public Memory envelope decision
is fixed, but it must be qualified with the exact Engine reader.

Acceptance:

1. write Engine events through `CanonicalPublicMemoryFacade.append_event()`;
2. close and reopen a manifested local workspace;
3. `ObservabilityFacade.list_engine_events()` returns flat Engine-authored `data`;
4. real Engine reader restores `agent_instance_id`, turn ID, dispatch token, causal
   links, prompt manifest, and Tool fields;
5. multi-run dispatch/return membership and timeline projection pass;
6. malformed nested storage envelope fails explicitly;
7. no flat legacy payload fallback is introduced.

### A4 — AG artifact authoritative commit alignment

Status: complete

Acceptance:

1. injected failures at blob, artifact-record, occurrence, retention, search,
   run-counter, and session-counter boundaries produce the explicitly decided
   authoritative/pending state;
2. injected search failure leaves exactly one authoritative artifact and occurrence;
3. run/session artifact accounting is complete or represented as one exact pending
   provider projection;
4. caller receives authoritative identity and projection state;
5. retry after close/reopen creates no duplicate content, occurrence, retention
   revision, or counter;
6. orphan or staged-blob handling uses the existing centralized ownership or one
   focused reviewed extension;
7. pinned retention behavior is unchanged;
8. local and deterministic-external contract conformance pass without a second
   artifact path, with real external provider qualification recorded separately when
   available.

### A5 — AG deprecated identity and log-scope hygiene

Status: complete

Acceptance:

1. missing `app_id` and `client_id` remain `None` in canonical observations;
2. formatter order cannot turn absence into `"-"` storage metadata;
3. explicitly supplied deprecated `app_id` remains marked optional compatibility
   metadata only;
4. no canonical scope, provider query, index, authorization, or workspace identity
   accepts `app_id`;
5. existing presentation formatting may still display a local placeholder without
   mutating the record seen by persistence handlers.

### A6 — Studio layered status adoption

Status: complete

Create/use an isolated Studio worktree from the verified current Studio head. Keep
all AG access inside the completed integration boundary.

Scope:

- Test Runtime status/output projection;
- lifecycle labels and API schemas;
- simulation item/batch status projection;
- Studio control/AI result projection and TypeScript API types;
- Design bottom panel, sandbox, simulation, and relevant Observability Timeline
  status rendering;
- persisted schema migration if required;
- no provider-store access and no Studio Agent implementation change.

Acceptance:

1. observed failure shape projects as transport completed plus Agent failed/runtime
   error;
2. overall/user status is not success;
3. raw Engine output remains preserved;
4. unprojectable future Engine outcome remains readable and explicitly unknown;
5. simulation does not count the failed Agent outcome as success;
6. cancellation, interruption, awaiting input, and resumable pause semantics remain
   correct;
7. UI badges, lifecycle labels, control projections, and simulation counts use the
   same centralized overall-status projection;
8. Studio's frozen AG runtime boundary audit and the current `695ea64` Timeline tests
   remain green.

### A7 — Studio operation-profile snapshot adoption

Status: complete

Scope:

- credential-free embedding/image binding snapshots;
- execution-config digest and schema migration;
- worker verification of the pinned complete operation set;
- centralized reuse of existing profile resolution, binding resolution, catalog, and
  stable-digest helpers;
- user-facing diagnostics and preflight;
- provider-capability-aware LLM logging/observability for qualification;
- no secrets in persisted/API payloads.

Acceptance:

1. the live LM Studio chat plus OpenAI embedding combination is visible before run;
2. a missing or statically incapable embedding binding fails preflight explicitly;
3. runtime quota/credit errors identify the embedding binding safely;
4. changing any required operation binding invalidates the pinned settings digest;
5. the worker recomputes the settings-source digest for every snapshot version before
   loading operations;
6. chat, embedding, and required image generation are all materialized from the same
   pinned operation snapshot rather than a later live file;
7. LLM logging/observability is enabled whenever the effective provider/endpoint
   capability supports it, with capture mode and correlation evidence recorded and
   secrets excluded;
8. an unsupported logging capability is recorded explicitly and never causes a
   provider fallback or alternate logging implementation;
9. no provider fallback is selected;
10. existing named chat-profile and Engine-role selection behavior remains intact.

### A8 — Cross-package exact-wheel qualification

Status: in progress

Build exact AG and Engine wheels from the reviewed commits. Use a fresh isolated test
environment; do not rely on editable imports or source overlays for release evidence.

Required scenarios:

1. successful Metalens chat with a valid embedding binding;
2. successful LM Studio chat with injected/real embedding projection failure after
   authoritative event commit;
3. authoritative event-store failure before commit;
4. one Tool call and Tool result;
5. multi-Agent child dispatch, return, and resumption;
6. prompt-manifest hydration and causal timeline;
7. artifact production with search projection failure and exact retry;
8. Studio interaction pause/resume;
9. Studio cancellation/interruption;
10. simulation item with failed Agent outcome;
11. absence and explicit deprecated presence of `app_id`;
12. local provider plus deterministic external contract conformance, with any real
    external-provider qualification recorded separately;
13. crash/restart after authoritative assistant-event commit and before semantic
    publication, using the same stable caller event identity;
14. pinned settings changed between snapshot and worker load for chat, embedding, and
    image-generation bindings;
15. provider-capability-aware LLM logging enabled for every live model operation that
    supports it, with an explicit unsupported-capability result otherwise.

For each scenario, record:

- exact commits and wheel hashes;
- Python and package versions;
- Studio environment snapshot and distribution digest;
- AG run/session, Engine turn, and Studio run/session identities;
- transport, Agent, workflow, and overall statuses;
- authoritative and indexed cursors/intent state where applicable;
- stable caller event identity and whether the append was new or idempotently
  existing;
- effective chat, embedding, and image provider/model/endpoint identities;
- LLM logging capability, selected capture mode, correlation IDs, and a secret-
  negative assertion;
- tests and expected skips;
- any deviation.

Metalens source and its existing dirty files remain unchanged. Only disposable test
runtime data created specifically for qualification may be removed, and exact paths
must be verified before cleanup.

### A9 — Final cleanup, documentation, and merge readiness

Status: pending

Acceptance:

1. no compatibility shim, old receipt behavior, fallback, dual route, or stale test
   fixture remains;
2. public docs explain authoritative durability versus search freshness;
3. Engine docs explain receipt semantics and semantic outcome layering;
4. Studio docs/UI terminology distinguishes transport and Agent result;
5. the original storage migration plan links this completed follow-up and records all
   boundary commits;
6. this ledger contains final commit hashes, wheel hashes, and complete test evidence;
7. all involved worktrees are clean except explicitly preserved user changes;
8. the user reviews the combined boundary result before any worktree is merged.

## 7. Failure-injection matrix

| Injection point | Authoritative state expected | Derived state expected | Engine behavior | Studio behavior |
|---|---|---|---|---|
| EventStore fails before commit | No event | No intent/index | No assistant publication; typed persistence failure | Transport/Agent failure shown accurately |
| Event commits; projection-intent commit fails in same transaction | Neither may commit | None | Typed authoritative failure | Failure, no output claim |
| Event + intent commit; embedding fails | One event | Failed/pending intent, no covering index cursor | Publish once; report degraded projection | Agent may complete; diagnostic identifies embedding projection |
| Projection retry succeeds | Same one event | Intent indexed exactly once | No duplicate output/event | Status remains stable; diagnostic may resolve visibly |
| Engine process stops after commit before semantic emit | One event | Any explicit projection state | Resume/recovery emits at most once using output identity | One assistant message |
| Artifact blob stage fails | No authoritative artifact occurrence | No search intent | Tool failure with exact phase | Accurate failure |
| Blob succeeds; artifact record fails | No authoritative artifact occurrence; staged/orphan blob is reconciled by the selected policy | No search intent | Tool failure with exact phase | Accurate failure |
| Artifact record succeeds; occurrence fails | State follows the explicit A0 authority decision; it is never reported as a complete occurrence | No search intent | Typed partial-stage failure | Accurate failure |
| Occurrence succeeds; retention or counters fail | One authoritative artifact occurrence plus one exact pending bookkeeping state | Search state remains explicit | Tool sees authoritative artifact plus typed degradation | Artifact remains visible; accounting is not invented |
| Artifact commits; search fails | One artifact + occurrence | Failed/pending search intent | Tool result reflects authoritative artifact plus degradation | Artifact remains visible; no duplicate on retry |
| Historical event envelope malformed | Existing malformed record | Not applicable | Projection fails explicitly | Inspection shows attributed boundary error |
| Agent outcome failed while AG run succeeded | AG transport succeeded | Not applicable | Typed failed outcome retained | Overall Agent failure, never `Test succeeded` |
| Embedding profile missing capability | No run if preflight can prove invalid | None | Not invoked | Explicit configuration failure, no fallback |

## 8. Test corrections required

### 8.1 AG tests

Add or revise tests so they exercise public paths rather than provider-private setup:

- public Memory append followed by historical observability reopen/read;
- authoritative receipt after search failure;
- projection intent persistence across close/reopen;
- exact projection retry;
- stable caller-owned event identity through public Memory, close/reopen, and exact
  retry;
- event and artifact duplicate prevention;
- artifact boundary failure injection and accounting under search failure;
- handler-order-independent absence of App/client compatibility identity;
- local/external provider conformance for every new capability, while labeling the
  deterministic external provider as contract conformance rather than real-service
  qualification.

Do not delete the useful existing tests that prove search failure is visible after
durable commit. Refine their expected contract so visibility no longer means false
authoritative failure.

### 8.2 Engine tests

Replace hand-built flat-only fakes with at least one shared boundary fixture produced
by AG's public Memory and observability facade. Add:

- committed + projection-failed receipt behavior;
- uncommitted failure behavior;
- exact-once assistant semantic publication;
- deterministic Engine-to-AG event identity and crash/restart after commit but before
  semantic publication;
- external-observation cursor correctness;
- child membership and causal timeline from real event shapes;
- preservation of safe typed AG diagnostics;
- accurate separation of recorded resource links from search readiness.

### 8.3 Studio tests

Add the missing cross-product assertions:

- AG transport `succeeded` + Engine outcome `failed`;
- AG transport `succeeded` + Engine runtime error;
- AG transport `succeeded` + paused/resumable outcome;
- unknown future Engine outcome key;
- simulation aggregation for each combination;
- operation snapshot includes chat, embedding, and required image identities;
- every snapshot version rejects a settings-source digest change before worker load;
- provider-supported LLM logging is enabled and correlated for live qualification;
- unsupported logging capability is explicit and does not trigger fallback;
- current `695ea64` Observability Timeline and status-tone behavior;
- no secret appears in snapshot, diagnostic, receipt, or API response.

### 8.4 AG LLM test policy and reviewed inventory

The only current provider-live AG tests are:

| Test | Provider/model | Required opt-in |
|---|---|---|
| `tests/live/test_openai_tool_search_cache_raw.py` | OpenAI `gpt-5.4` by default | `OPENAI_API_KEY` and `AG_RUN_OPENAI_CACHE_SMOKE=1` |
| `tests/live/test_openai_tool_search_cache_client.py` | OpenAI `gpt-5.4` by default | `OPENAI_API_KEY` and `AG_RUN_OPENAI_CACHE_SMOKE=1` |

`AG_OPENAI_CACHE_SMOKE_MODEL` may override the live model. An API key alone must not
activate a billable/network test. The current non-live LLM, embedding, image,
catalog, profile, transport, observability, metering, and runtime-registration tests
use fake clients, mocked transports, static catalogs, or synthetic model identities.
They must remain deterministic even when provider keys are present in the environment.

For every provider-live test added or changed by this plan:

1. require an explicit scenario-specific opt-in in addition to credentials;
2. record provider, model, endpoint, operation, and expected number of live calls;
3. enable the existing AG LLM logging/observability capability when the effective
   provider/endpoint supports it;
4. assert request/result correlation and the selected credential-safe capture mode;
5. assert that no API key, credential, sensitive header, or secret-bearing provider
   body appears in logs, observations, diagnostics, snapshots, or receipts;
6. record unsupported logging capability explicitly without fallback or a duplicate
   logging adapter;
7. keep live tests in the centralized `tests/live` scenario/fixture structure rather
   than adding per-module scripts.

## 9. Baseline and qualification commands

Resolve exact environments first. These commands are starting points, not permission
to use an interpreter whose package hashes do not match the intended commits. Before
each baseline, record `sys.executable`, Python version, installed AG/Engine/Studio
distribution versions, wheel `RECORD` availability, required extras such as Pillow,
and source/wheel hashes. Do not use `PYTHONPATH`, editable imports, or source overlays
as release qualification evidence.

### AG focused baseline

```powershell
Set-Location "C:\Users\zcliu\Documents\Github\aethergraph-suite\.worktrees\aethergraph-storage-provider-migration-20260814"
$AgPython = "<resolved absolute AG test interpreter>"
& $AgPython -m pytest `
  tests/test_canonical_memory_facade.py `
  tests/test_canonical_artifact_facade.py `
  tests/test_observability_workspace.py `
  tests/test_local_storage_provider.py `
  tests/storage_conformance -q
```

### Engine focused baseline

```powershell
Set-Location "C:\Users\zcliu\Documents\Github\aethergraph-suite\ag-engine"
$EnginePython = "<resolved absolute Engine test interpreter>"
& $EnginePython -m pytest `
  tests/test_observability_source.py `
  tests/test_react_loop_v3_events.py `
  tests/test_react_loop_v3_external_observations.py `
  tests/test_react_loop_v3_run_exit.py -q
```

Resolve the exact current response-output test module during A0; do not assume this
plan's filename inventory is complete.

### Studio focused baseline

```powershell
Set-Location "C:\Users\zcliu\Documents\Github\aethergraph-suite\ag-studio"
$StudioPython = "<resolved absolute Studio test interpreter>"
& $StudioPython -m pytest `
  backend/tests/test_test_runtime_outcome_contract.py `
  backend/tests/test_studio_ai_semantics.py `
  backend/tests/test_simulation_api.py `
  backend/tests/execution/test_supervisor.py -q
```

### Metalens acceptance baseline

Run from `others/ag-metalens` using its exact documented environment:

```powershell
$MetalensPython = "<resolved absolute Metalens test interpreter>"
& $MetalensPython -m unittest agent.v5.tests.test_chat_agent -v
```

Discover and run the complete current Metalens suite rather than assuming this one
module is sufficient. At the preceding migration qualification, the complete suite
was `112 passed, 1 skipped`; reverify the current count.

### AG LLM baseline and opt-in live tests

Run the deterministic LLM/model suite with provider keys present and live opt-in
flags absent. It must remain offline and skip the live tests. Use the existing
centralized test files rather than creating a standalone script.

For the current OpenAI cache smoke, live execution requires both
`OPENAI_API_KEY` and `AG_RUN_OPENAI_CACHE_SMOKE=1`; the key alone is insufficient.
The default model is `gpt-5.4`, optionally overridden by
`AG_OPENAI_CACHE_SMOKE_MODEL`. Before enabling the flag:

1. record the expected six live Responses API calls when both current smoke tests run;
2. configure AG LLM observability/logging through the existing settings/profile path
   when the effective endpoint capability supports it;
3. record the selected capture mode and correlation evidence;
4. assert that credentials and sensitive provider bodies are absent;
5. unset the explicit opt-in after the isolated live run.

Do not add a second live runner, infer live execution from credential presence, or
substitute a different provider when logging or the requested operation is
unsupported.

### Planning/review-time audit tests

The read-only audit reran seven relevant existing tests:

- AG: 2 passed;
- Engine: 1 passed;
- Studio: 4 passed.

They demonstrated that current package-local tests pass despite the live mismatch,
which is why the cross-package fixtures in section 8 are mandatory.

The follow-up review also ran the AG LLM/model-focused suites with an
`OPENAI_API_KEY` present and the explicit live flag absent:

- focused core LLM/model suite: `325 passed, 2 skipped`;
- media suite under an interpreter with Pillow: `9 passed`;
- broader LLM/embedding/image/configuration/observability audit:
  `477 passed, 2 skipped, 1 environment-only failure`.

The two skips were the provider-live OpenAI cache tests. The environment-only failure
was the host-manifest exact-distribution test because the available installed AG
distribution lacked wheel `RECORD` metadata. This is not product qualification and
must be corrected by the exact-wheel A8 environment.

### Implementation validation checkpoint

The implementation checkpoint on 2026-08-17 produced the following evidence before
exact-wheel qualification:

- Engine full suite with the selected AG source: `774 passed`;
- AG full source suite: all functional tests passed; the two host release tests failed
  only because a source overlay has no installed wheel `RECORD` metadata and are
  reserved for A8;
- focused canonical Studio observability fixtures: `4 passed` after replacing deleted
  direct SQLite fixture writers with one shared manifested-provider fixture;
- focused Studio contract repairs (authoritative diagnostic receipts, layered frozen
  status, current unversioned migration stamping, composition preview, and Tiny Agent
  release path): `19 passed` after the final baseline amendment;
- Studio UI clean install, production build, and complete test suite: `125` files and
  `800` tests passed;
- `npm audit --json`: `0` vulnerabilities after pinning `dompurify 3.4.13`,
  `nanoid 3.3.18`, `postcss 8.5.26`, and `react-router-dom 7.18.2` through the existing
  package dependency/override ownership; no alternate install or fallback path was
  added.

The Studio backend full source-overlay run originally reported 23 failures. Ten were
fixed through the centralized current contracts above. The 13 remaining cases are
classified before A8 rather than hidden:

- nine `test_trace_projection.py` cases and one Assistant trace API case use the
  checked-in pre-manifest SQLite observability fixture; production historical opening
  correctly rejects it and no legacy reader/fallback will be added;
- deployment and local-host-supervisor cases require installed AG/Engine distributions
  and belong to the exact-wheel environment;
- one release-audit assertion about `validateControlProject` predates and is unrelated
  to authoritative outcome alignment; it is recorded as an out-of-plan Studio cleanup
  finding for A9/user review.

The normal Vite chunk-size warning remains non-security build advice. It is not an npm
audit finding and is not addressed by this plan.

## 10. Stop conditions

Stop implementation and discuss with the user if any of these occurs:

1. authoritative record plus projection intent cannot be made atomic for every
   qualified provider;
2. the proposed fix requires a legacy store, fallback provider, dual read/write, or
   history migration;
3. the fix would make `app_id` or `client_id` canonical provider scope;
4. Studio would need direct provider/store/container access outside
   `StudioAetherGraphRuntime` and its worker bridge;
5. a new Engine implementation appears necessary rather than a focused ReAct v3
   adoption patch;
6. changing AG's public Memory return contract would silently break an unowned
   consumer not inventoried in A0;
7. local and external providers cannot satisfy the same semantics;
8. exact-once assistant publication cannot be proven across crash/retry;
9. a current dirty change overlaps a required file and ownership is unclear;
10. a test would require modifying Metalens user changes or Studio code outside its
    completed integration boundary;
11. preserving a failed Engine outcome would require discarding raw output evidence;
12. exact-wheel qualification cannot reproduce the source-level result;
13. stable caller event identity cannot be carried through public AG Memory without a
    duplicate append/recovery path;
14. the proposed provider transaction fix requires a generic transaction framework,
    raw provider connection, or facade-owned transaction manager rather than a
    focused existing-contract extension;
15. a deterministic external conformance fake would be the only evidence offered for
    a claimed real external-provider qualification;
16. Studio cannot verify and materialize all required operation bindings from one
    immutable snapshot without reading a later live settings path;
17. provider-supported LLM logging would require a fallback provider, duplicate
    adapter, or secret-bearing capture;
18. a new helper, receipt, projector, or script duplicates an existing owner and
    cannot be removed or centralized before merge.

## 11. Implementation ledger

Update this table immediately after each reviewed commit. Include commit hashes from
all repositories, exact test commands/counts, wheel hashes, and deviations. Do not
mark a milestone complete based only on source inspection or mocked tests.

| Milestone | Status | AG commit | Engine commit | Studio commit | Evidence and deviations |
|---|---|---|---|---|---|
| Plan preparation | Complete | documentation-only working tree change | — | — | Live Studio/Engine/AG evidence recorded; no production code changed; seven existing audit tests passed. |
| Plan review amendment | Complete | documentation-only working tree change | — | `d92a4cb` review base; implementation base advanced to `695ea64` | Added stable idempotency, minimal/reuse-first contract rules, provider-conformance distinction, complete Studio snapshot verification, granular artifact failures, UI scope, and provider-capability-aware LLM logging policy. AG LLM audit: 325 passed/2 live skipped; media 9 passed; broader audit 477 passed/2 skipped/1 environment-only failure. |
| A0 reverification and contract decision | Complete | `092a4ee` base + working tree | `85c19c9` base + isolated worktree | `695ea64` base + isolated worktree | Worktrees and heads verified; focused EventStore composite, deterministic Engine event identity, layered Studio status, and v3 operation snapshot decisions recorded above. |
| A1 AG authoritative Memory semantics | Complete | working tree, not committed | — | — | Event + deterministic search intent atomic locally; public discriminated receipt; sanitized degradation; exact retry and close/reopen persistence. Focused AG/provider/conformance set: 50 passed. Real external service qualification remains A8 and is not claimed. |
| A2 existing Engine adoption | Complete | paired AG working tree | working tree, not committed | — | ReAct v3 uses only `append_event_commit`; blanket empty-receipt catches removed; stable payload-derived event IDs; receipts distinguish newly created from idempotently reused events; recorded links are separate from search status; assistant projection degradation/error/restart tests pass. Complete modified Engine regression set: 188 passed; final receipt amendment recheck: 39 passed. |
| A3 historical Engine-event projection | Complete | working tree, not committed | reader boundary retained | — | Public Memory write → local workspace close/reopen → flat Engine data projection covered in the 20-test observability set. Malformed envelopes fail without a flat fallback. |
| A4 Artifact alignment | Complete | working tree, not committed | — | — | One focused `ArtifactRepository.commit_production` transaction owns metadata, occurrence, initial retention, run/session accounting, and deterministic search intent. Blob staging retains the existing orphan reconciler; search degradation returns an authoritative receipt and retries the same intent after close/reopen without duplicate occurrence, retention, or counters. Focused artifact/control/API set: 50 passed; local plus deterministic-external conformance set: 26 passed. Real external service qualification remains A8. |
| A5 deprecated identity hygiene | Complete | working tree, not committed | — | — | SafeFormatter now formats a copy; persisted absent App/client identity remains absent regardless of handler order. Included in the 20-test observability set. |
| A6 Studio layered status | Complete | — | — | working tree, not committed | Central projector exposes overall `status` plus explicit `transport_status`; unknown/paused/simulation behavior updated; raw Engine output retained. Studio outcome/control/simulation qualification included in 28 passing backend tests; UI status/Timeline set: 13 passed; TypeScript check passed. |
| A7 Studio operation profiles | Complete | endpoint capability registry working tree | — | working tree, not committed | Runtime snapshot v3 pins credential-free chat, embedding, and image-generation profiles plus exact resolved bindings and endpoint-owned observability evidence. Required embedding/image capabilities fail preflight; the worker recomputes the settings-source digest before materialization and enables manifest capture whenever any pinned endpoint supports it. Unsupported capture is explicit and does not select a fallback. Final AG registry set: 30 passed; Studio snapshot/worker set: 15 passed. |
| A8 exact-wheel qualification | In progress | commit pending | commit pending | commit pending | Engine full source suite: 774 passed. AG functional source suite passed with two expected wheel-metadata release failures. Studio UI: 800 tests, build passed, npm audit 0. Exact commits, wheel hashes, isolated distribution tests, live IDs, and expected skips remain to record. |
| A9 cleanup and merge readiness | Pending | — | — | — | User review required before merge. |

## 12. Commit and worktree discipline

1. Keep AG implementation in the existing storage-provider migration worktree.
2. Create isolated Engine and Studio worktrees only after A0 verifies their bases.
3. Use focused commits whose messages identify the boundary being corrected.
4. Keep shared fixtures, retry/drain utilities, model-binding resolvers, digest
   helpers, and qualification scripts in their existing centralized ownership. Group
   new related helpers together; do not add package-local copies.
5. Update this plan in the same repository after each external commit so the AG
   worktree remains the authoritative cross-package ledger.
6. Never commit generated test environments, runtime workspaces, databases, WAL/SHM
   files, secrets, or Metalens `.data`.
7. Build wheels only from clean reviewed commits and record SHA-256 hashes.
8. Do not merge or delete any worktree until A8 is complete and the user has reviewed
   A9 evidence.

## 13. Completion definition

This follow-up is complete only when all of the following are true:

- authoritative AG event and artifact durability cannot be confused with derived
  search readiness;
- projection failure is visible, typed, retryable, and never selects a fallback;
- stable caller-owned Engine event identity proves exact-once behavior across
  crash/restart without a duplicated publication or append path;
- existing Engine ReAct v3 publishes a committed assistant output exactly once even
  when its search projection is degraded;
- true authoritative failure prevents publication and retains its exact safe cause;
- historical Engine events round-trip through public Memory and AG observability with
  the shape Engine expects;
- multi-Agent membership, causal timelines, Tool activity, and prompt hydration pass
  against real boundary data;
- Studio and simulation distinguish AG transport completion from Engine Agent
  failure;
- Studio pins and presents every required model-operation binding;
- Studio verifies the immutable settings source for every snapshot version and never
  mixes pinned chat with later live embedding/image settings;
- provider-supported LLM logging is enabled for live qualification, correlated, and
  proven secret-free; unsupported capability is explicit without fallback;
- artifact bookkeeping remains coherent across projection failure and retry;
- absent `app_id`/`client_id` never becomes `"-"` canonical metadata;
- local providers and the deterministic external fake pass the same contract
  conformance, and any claimed real external provider has separate qualification;
- similar functions, fixtures, resolvers, and scripts are centralized under one
  owner, with no bloated generic abstraction or duplicate path remaining;
- exact AG/Engine wheels pass Studio + Engine + Metalens qualification;
- the original migration plan links the final evidence;
- no source worktree is merged until the user approves the complete boundary set.
