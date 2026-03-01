---
name: aethergraph-agent-graph-fastbuild
description: Build new AetherGraph agents and graphs quickly from user intent using @tool, @graph_fn, @graphify, run_async, start_server, and NodeContext service APIs. Use for fast creation, API-first coding, and producing runnable scaffolds or feature slices without over-constraining exact architecture patterns.
---

# AetherGraph Agent/Graph Fastbuild

Create working AetherGraph code quickly. Optimize for shipping new functionality and using the real API correctly.

## Core Approach

- Prefer the shortest path to runnable code.
- Treat this repository source as truth when docs conflict.
- Start with clear IO (`inputs`, `outputs`) and return matching dict keys.
- Use `@graph_fn` for fast async orchestration.
- Use `@tool + @graphify` when workflow steps should be explicit/reusable.
- Add `context: NodeContext` only when runtime services are needed.

## Build Flow

1. Pick primitive:
- `@graph_fn` for quick feature delivery and dynamic control flow.
- `@tool + @graphify` for explicit DAG composition and reusable steps.

2. Define contract:
- Declare `name`, `inputs`, `outputs` on decorators.
- Ensure code returns exactly declared outputs.

3. Add runtime features only when needed:
- Channel messaging, artifacts, memory, child runs, LLM, viz, triggers, skills.
- Use NodeContext methods from `references/node_context_services.md`.

4. Execute quickly:
- Local: `await run_async(graph_or_fn, inputs={...})`
- Server/UI: `start_server(...)` then keep process alive if needed.

5. Tighten only after first run:
- Improve metadata (`as_agent`, `as_app`), validation, and guardrails.

## Resource Map

- Primitives and signatures: `references/primitives.md`
- Run and server startup: `references/run_and_server.md`
- NodeContext service APIs: `references/node_context_services.md`
- Copy-ready examples: `references/examples.md`

## Output Rules for Generated Code

- Keep snippets runnable with explicit imports.
- Return dict outputs that match decorator `outputs`.
- If using context services, include `*, context: NodeContext` in signature.
- Avoid inventing APIs; use only methods shown in references.
- Prefer minimal viable implementation first, then iterate.

## External Tutorial

- Secondary reference: https://aiperture.github.io/aethergraph-docs/tutorials/t1-build-your-first-graph-fn/
- Treat repository source code as the final authority when examples differ.

