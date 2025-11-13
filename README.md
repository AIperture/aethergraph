# AetherGraph

**AetherGraph** is a **Python‑first agentic DAG execution framework** for building and orchestrating AI‑powered workflows. It pairs a clean, function‑oriented developer experience with a resilient runtime—event‑driven waits, resumable runs, and pluggable services (LLM, memory, artifacts, RAG)—so you can start simple and scale to complex R&D pipelines.

Use AetherGraph to prototype interactive assistants, simulation/optimization loops, data transforms, or multi‑step automations without boilerplate. It works **with or without LLMs**—bring your own tools and services, and compose them into repeatable, observable graphs.

---

## Requirements

* Python **3.10+**
* macOS, Linux, or Windows
* *(Optional)* An LLM API key (OpenAI, Anthropic, Google, etc.)

---

## Install

### Option A — From Git (early testing)

Pin to a **tag** or **commit** (avoid `@main` for reproducibility):

```bash
pip install "aethergraph @ git+https://github.com/AIperture/aethergraph.git@v0.1.0a1"
# or an exact commit
pip install "aethergraph @ git+https://github.com/AIperture/aethergraph.git@<commit_sha>"
```

### Option B — From PyPI (pre‑release or stable)

When available on PyPI:

```bash
# Stable channel
pip install aethergraph

# Pre‑releases (alpha/beta/rc)
pip install --pre aethergraph
```

---

## Configure (optional)

Most examples run without an LLM, but for LLM‑backed flows set keys via environment variables or a local secrets file.

Minimal example (OpenAI):

```ini
# .env (example)
AETHERGRAPH_LLM__ENABLED=true
AETHERGRAPH_LLM__DEFAULT__PROVIDER=openai
AETHERGRAPH_LLM__DEFAULT__MODEL=gpt-4o-mini
AETHERGRAPH_LLM__DEFAULT__API_KEY=sk-...your-key...
```

Or inline in a script at runtime (for on‑demand key setting):

```python
import os
from aethergraph.runtime import register_llm_client

open_ai_client = register_llm_client(
    profile="my_llm",
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...your-key...",
)
```
---

## Quickstart (60 seconds)

1. Verify install:

```bash
python -c "import aethergraph; print('AetherGraph OK, version:', getattr(aethergraph, '__version__', 'dev'))"
```

2. Run a minimal example:

```bash
python - <<'PY'
from aethergraph import graph_fn

@graph_fn()
async def hello(ctx):
    print("Hello from AetherGraph!")

import asyncio
asyncio.run(hello())
PY
```

---

## Examples

* Local examples live under `examples/` in this repo.
* A growing gallery of standalone examples will be published at:

  * **Repo:** [https://github.com/AIperture/aethergraph-examples](https://github.com/AIperture/aethergraph-examples)
  * **Path:** `examples/`

Run an example:

```bash
cd examples
python 0_hello_world.py
```

---

## CLI (if enabled)

If a CLI entry point is provided:

```bash
aethergraph --help
```

---

## Troubleshooting

* **`ModuleNotFoundError`**: ensure you installed into the active venv and that your shell is using it.
* **LLM/API errors**: confirm provider/model/key configuration (env vars or your local secrets file).
* **Windows path quirks**: clear any local cache folders (e.g., `.rag/`) and re‑run; verify write permissions.

---

## Contributing (early phase)

* Use feature branches and open a PR against `main`.
* Keep public examples free of real secrets.
* Run tests locally before pushing.

Dev install:

```bash
pip install -e .[dev]
pytest -q
```

---

## Versioning & Release Policy

* **Pre‑releases:** `0.1.0aN`, `0.1.0bN`, `0.1.0rcN` for canaries.
* **Stable:** `0.1.0`.
* **Hotfix/patch:** `0.1.1`, etc.
* **Docs/metadata‑only updates:** post‑releases like `0.1.0.post1`.
* **Bad release?** Yank it on PyPI so it’s skipped by default.

---

## Project Links

* **Source:** [https://github.com/AIperture/aethergraph](https://github.com/AIperture/aethergraph)
* **Issues:** [https://github.com/AIperture/aethergraph/issues](https://github.com/AIperture/aethergraph/issues)
* **Examples:** [https://github.com/AIperture/aethergraph-examples](https://github.com/AIperture/aethergraph-examples)
* **Docs (preview):** [https://aiperture.io/docs](https://aiperture.github.io/aethergraph-docs/)

---

## License

**Apache‑2.0** — see `LICENSE`.
