---
name: aethergraph-codex-graph-builder
description: Build and iteratively repair AetherGraph `@tool + @graphify` workflows for Codex users from plain intent, existing scripts, docs, or an empty workspace. Use when a user wants a runnable workflow generated into their workspace, persisted as versioned files, validated with current `graphify` rules, registered as an app for AG UI, and optionally run locally with `run_async` or served with `start_server`.
---

# AetherGraph Codex Graph Builder

Generate a runnable AetherGraph workflow in the user's workspace and carry it through validation, repair, and app registration.

## Preflight

- Do not assume the user has the AetherGraph repository checked out.
- The minimum requirement is an environment where `aethergraph` can be imported and run.
- If the workspace does not contain the AG repo, still generate the workflow into the user's project, but explicitly tell the user they need an installed `aethergraph` package and a runnable Python environment before validation, registration, or UI serving will work.
- If neither the repo nor an installed package is available, stop and say that execution cannot proceed yet.

## Core Contract

- Default to `@tool + @graphify` for AG UI workflows.
- Treat repository source as truth for API details.
- Keep graph inputs and outputs string-only unless the user explicitly requires otherwise.
- Persist important large or structured intermediates as artifacts, then pass artifact ids/uris as strings between nodes.
- Prefer artifact-backed final outputs too; return artifact ids/uris as the main graph outputs unless the user explicitly asks for filesystem output or inline payloads.
- Do not use `send_phase`; the run channel pattern is no longer reliable for this workflow.
- For long-running steps, prefer artifact checkpoints and durable saved outputs over streaming status.

## Default Workspace Layout

- Create workflows under `aethergraph_graphs/<workflow_slug>/`.
- Put the current runnable graph in `aethergraph_graphs/<workflow_slug>/workflow.py`.
- Save supporting notes in `aethergraph_graphs/<workflow_slug>/README.md` only if the user asks.
- When modifying an existing workflow, archive the previous implementation under `aethergraph_graphs/<workflow_slug>/versions/<timestamp>__<change_slug>.py` before replacing `workflow.py`.

## Build Flow

1. Inspect the user's workspace for reusable scripts, docs, schemas, or sample data.
2. Derive a workflow contract:
- Choose a stable workflow slug.
- Define graph `name`, string `inputs`, and string `outputs`.
- Decide which data must move through artifacts instead of inline payloads.
3. Generate the graph module:
- Use explicit `@tool(...)` and `@graphify(name=..., inputs=..., outputs=..., as_app=...)`.
- Keep `@graphify` as sync `def`.
- Make tool signatures explicit, adding `*, context: NodeContext` only when runtime services are used.
- Add a reasonable `as_app.input_schema` with typed widgets, labels, descriptions, and useful defaults so AG UI forms are usable without manual editing.
4. Validate and repair iteratively:
- First fix registration-facing issues with `validate_graphify_source(...)` or registration errors.
- Then import the module and call `<graph_fn>.build()` to surface graph construction errors.
- Repeat until validation and build succeed.
5. Register the workflow as an app.
6. Ask the user what to do next:
- Run locally with `run_async`
- Start a local AG UI server with `start_server` and approval if needed
- Cancel after generation/registration

## Output Rules

- Keep generated modules self-contained and runnable with explicit imports.
- Ensure every declared graph output is returned by exact key.
- Use `inputs=[...]` when all inputs are required.
- Use `inputs={...}` only when defaults are intentional and fully understood.
- Keep `as_app` minimal but valid, with at least `id`.
- Prefer including `as_app.input_schema` for general-purpose workflows. Infer field types from the graph contract and add practical defaults whenever the workflow can reasonably start from them.
- Use user-facing app metadata that matches the workflow purpose.

## Repair Loop

- Treat validation failures as required fixes, not advisory notes.
- Check current fail-fast graphify rules in `references/validation_and_registration.md`.
- After syntax and decorator validation pass, call `.build()` on the graph function and fix builder/runtime graph wiring errors.
- If registration prints app or graph metadata issues, update `as_app`, IO declarations, or returned keys and retry.
- Stop only when the generated module can be validated, built, and registered cleanly.

## Long-Run And Artifact Policy

- Keep graph edges string-only.
- Save blobs, JSON payloads, reports, generated code, media, and other large outputs to artifacts.
- Pass artifact identifiers or uris into later nodes as strings.
- Default to saving the final deliverable into artifacts as well so the result is visible in the centralized artifact store.
- For expensive steps, add checkpoint save/load tools so reruns can skip repeated work.
- For long iterative nodes, save progress artifacts from inside the node and resume from the latest saved state if the node retries or restarts.
- Prefer durable artifact saves over progress chatter.

## Resource Map

- Current `graphify` constraints: `references/graphify_rules.md`
- Validation, registration, and `.build()` loop: `references/validation_and_registration.md`
- Artifact and checkpoint patterns: `references/artifact_patterns.md`
- Local run and AG UI server flow: `references/run_and_register.md`

## Finalization

- Register the workflow as an app after code generation succeeds.
- If the user asked to revise an existing workflow, preserve version history before overwriting the main file.
- After registration, ask whether to run it locally, start AG UI, or stop.
