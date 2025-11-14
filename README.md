<p align="center">
  <img src="assets/logo.png" alt="AetherGraph" width="360"/>
</p>

# AetherGraph

**AetherGraph** is a **Python‑first agentic DAG execution framework** for building and orchestrating AI‑powered workflows. It pairs a clean, function‑oriented developer experience with a resilient runtime—event‑driven waits, resumable runs, and pluggable services (LLM, memory, artifacts, RAG)—so you can start simple and scale to complex R&D pipelines.

Use AetherGraph to prototype interactive assistants, simulation/optimization loops, data transforms, or multi‑step automations without boilerplate. It works **with or without LLMs**—bring your own tools and services, and compose them into repeatable, observable graphs.

---

## Requirements

* Python **3.10+**
* macOS, Linux, or Windows
* *(Optional)* LLM API keys (OpenAI, Anthropic, Google, etc.)
* *(Optional extras)* `slack` adapter

---

## Install

### Option A — PyPI (recommended)

```bash
pip install aethergraph
```

Optional extras:

```bash
# Slack adapter
pip install "aethergraph[slack]"

# Dev tooling (linting, tests, types)
pip install "aethergraph[dev]"
```


### Option B — From source (editable dev mode)

```bash
git clone https://github.com/AIperture/aethergraph.git
cd aethergraph

# Base
pip install -e .

# With extras
echo "(optional)" && pip install -e ".[slack,dev]"
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
from aethergraph.runtime import register_llm_client

open_ai_client = register_llm_client(
    profile="my_llm",
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...your-key...",
)
```

See our docs for setup of **external channel** methods for real-time interaction. 


> **Where should `.env` live?** In your **project root** (the directory where you run your Python entry point). You can override with `AETHERGRAPH_ENV_FILE=/path/to/.env` if needed.

---

## Quickstart (60 seconds)

1. Verify install:

```bash
python -c "import aethergraph; print('AetherGraph OK, version:', getattr(aethergraph, '__version__', 'dev'))"
```

2. Run a minimal graph:

```bash
python - <<'PY'
from aethergraph import graph_fn, NodeContext
from aethergraph.runner import run 

@graph_fn(name="hello_world")
async def hello_world(context: NodeContext):
    print("Hello from AetherGraph!")
    return {"ok": True}

run(hello_world)
PY
```

---

## Examples

Quick‑start scripts live under `examples/` in this repo. A growing gallery of standalone examples will be published at:

* **Repo:** [https://github.com/AIperture/aethergraph-examples](https://github.com/AIperture/aethergraph-examples)
* **Path:** `examples/`

Run an example:

```bash
cd examples
python hello_world.py
```

---

## Troubleshooting

* **`ModuleNotFoundError`**: ensure you installed into the active venv and that your shell is using it.
* **LLM/API errors**: confirm provider/model/key configuration (env vars or your local secrets file).
* **Windows path quirks**: clear any local cache folders (e.g., `.rag/`) and re‑run; verify write permissions.
* **Slack extra**: install with `pip install "aethergraph[slack]"` if you need Slack channel integration.

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

## Project Links

* **Source:** [https://github.com/AIperture/aethergraph](https://github.com/AIperture/aethergraph)
* **Issues:** [https://github.com/AIperture/aethergraph/issues](https://github.com/AIperture/aethergraph/issues)
* **Examples:** [https://github.com/AIperture/aethergraph-examples](https://github.com/AIperture/aethergraph-examples)
* **Docs (preview):** [https://aiperture.github.io/aethergraph-docs/](https://aiperture.github.io/aethergraph-docs/)

---

## License

**Apache‑2.0** — see `LICENSE`.
