<p align="center">
  <img src="assets/logo.png" alt="AetherGraph" width="360"/>
</p>

# AetherGraph

**AetherGraph** is a **Python-first agentic DAG execution framework** for building and orchestrating AI-powered workflows. It pairs a clean, function-oriented developer experience with a resilient runtime—event-driven waits, resumable runs, and pluggable services (LLM, memory, artifacts, RAG)—so you can start simple and scale to complex R&D pipelines.

Use AetherGraph to prototype interactive assistants, simulation/optimization loops, data transforms, or multi-step automations without boilerplate. It works **with or without LLMs**—bring your own tools and services, and compose them into repeatable, observable graphs.

* **[Introduction](https://aiperture.io)**
* **[Docs](https://aiperture.github.io/aethergraph-docs/)**
* **[Examples](https://github.com/AIperture/aethergraph-examples)**

---

## Requirements

* Python **3.10+**
* macOS, Linux, or Windows
* *(Suggested)* LLM API keys (OpenAI, Anthropic, Google, etc.)
* *(Optional)* `slack` or `telegram` token. See [Channel Setup](https://aiperture.github.io/aethergraph-docs/channel-setup/introduction/)
* *(Optional UI)* A modern browser to use the built-in AetherGraph web UI

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

> The PyPI package ships with a prebuilt UI bundle, so `/ui` works out of the box
> when you run the server locally.

### Option B — From source (editable dev mode)

```bash
git clone https://github.com/AIperture/aethergraph.git
cd aethergraph

# Base
pip install -e .

# With extras
echo "(optional)" && pip install -e ".[slack,dev]"
```

> When running from source, the backend still works without the frontend bundle.
> If you want the **built-in UI** from source, you’ll need to build the frontend
> and copy the static bundle into `aethergraph/server/ui_static/`
> (see the “UI Guide” in the docs).

---

## Configure

Aethergraph can run without an LLM, but for many LLM-backed flows in [examples](https://github.com/AIperture/aethergraph-examples), set keys via environment variables or a local secrets file.

Minimal example (OpenAI):

```ini
# .env (example)
AETHERGRAPH_LLM__ENABLED=true
AETHERGRAPH_LLM__DEFAULT__PROVIDER=openai
AETHERGRAPH_LLM__DEFAULT__MODEL=gpt-4o-mini
AETHERGRAPH_LLM__DEFAULT__API_KEY=sk-...your-key...
```

Or inline in a script at runtime (for on-demand key setting):

```python
from aethergraph.runtime import register_llm_client

open_ai_client = register_llm_client(
    profile="my_llm",
    provider="openai",
    model="gpt-4o-mini",
    api_key="sk-...your-key...",
)
```

See the docs for setup of **external channel** methods (Slack, Telegram, etc.) for real-time interaction.

> **Where should `.env` live?**
> In your **project root** (the directory where you run your Python entry point).
> You can override with `AETHERGRAPH_ENV_FILE=/path/to/.env` if needed.

---

## Verify install

```bash
python -c "import aethergraph; print('AetherGraph OK, version:', getattr(aethergraph, '__version__', 'dev'))"
```

---

## Run the built-in UI

AetherGraph ships with a small web UI that lets you:

* Browse and launch **apps** (click-to-run graphs)
* Chat with **agents** (graph-backed chat endpoints)
* Inspect **runs**, **sessions**, and **artifacts**

### 1. Define a simple project module

From your project root:

```text
my_project/
  demos/
    __init__.py
    chat_demo.py
  aethergraph_data/   # workspace (created automatically as needed)
```

Example `chat_demo.py`:

```python
# demos/chat_demo.py
from aethergraph import graphify

@graphify(
    name="chat_with_memory_demo",
    inputs=[],
    outputs=["turns", "summary"],
    as_app={
        "id": "chat_with_memory_demo",
        "name": "Chat with Memory",
    },
)
def chat_with_memory_demo():
    # Your graph implementation here – tools, nodes, etc.
    ...
```

> `as_app={...}` tells AetherGraph to expose this graph in the **App Gallery** of the UI.
> You can also define `graph_fn`-based **agents** with `as_agent={...}` to appear in the
> **Agent Gallery**.

### 2. Start the server with UI from the terminal (recommended)

From `my_project/`:

```bash
aethergraph serve \
  --project-root . \
  --load-module demos \
  --reload
```

This will:

* Add `.` to `sys.path` so `demos` can be imported.
* Load any graphs/apps/agents defined in the `demos` module.
* Start the API + UI server on `http://127.0.0.1:8745`.
* Enable **auto-reload**: editing your graph files triggers a restart and reload.

You should see log lines like:

```text
[AetherGraph] 🚀  Server started at:  http://127.0.0.1:8745
[AetherGraph] 🖥️  UI:                 http://127.0.0.1:8745/ui
[AetherGraph] 📡  API:                http://127.0.0.1:8745/api/v1/
[AetherGraph] 📂  Workspace:          ./aethergraph_data
[AetherGraph] ♻️  Auto-reload:        enabled
```

Then open in your browser:

* **UI:** `http://127.0.0.1:8745/ui` – App Gallery, Agent Gallery, runs, sessions, artifacts.
* **API:** `http://127.0.0.1:8745/api/v1/` – for direct HTTP calls.

### 3. (Optional) Start the server from a Python script

If you prefer to embed the server in your own launcher:

```python
# start_server.py
from aethergraph import start_server

if __name__ == "__main__":
    start_server(
        workspace="./aethergraph_data",
        project_root=".",
        load_module=["demos"],
        host="127.0.0.1",
        port=8745,
    )
```

```bash
python start_server.py
```

> For **active development**, the CLI with `--reload` is recommended.
> `start_server(...)` is better when you want a simple “ship a server in my app” story.

For more details, see the **UI Guide** section in the docs (server setup, agents/apps, and common pitfalls).

---

## Examples

Quick-start scripts live under `examples/` in this repo.

Run an example:

```bash
cd examples
python hello_world.py
```

A growing gallery of standalone examples and recipes lives under:

* **Repo:** [https://github.com/AIperture/aethergraph-examples](https://github.com/AIperture/aethergraph-examples)

---

## Troubleshooting

* **`ModuleNotFoundError`**: ensure you installed into the active venv and that your shell is using it.
* **LLM/API errors**: confirm provider/model/key configuration (env vars or your local secrets file).
* **Windows path quirks**: clear any local cache folders (e.g., `.rag/`) and re-run; verify write permissions.
* **Slack extra**: install with `pip install "aethergraph[slack]"` if you need Slack channel integration.
* **UI shows no apps/agents**:

  * Make sure your module (e.g. `demos`) is importable under `--project-root`.
  * Ensure at least one graph has `as_app={...}` or `graph_fn` has `as_agent={...}`.

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

**Apache-2.0** — see `LICENSE`.
