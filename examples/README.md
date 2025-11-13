# Quickstart · Hello World + `graph_fn` & `NodeContext` intro

> Run an agent in under 90 seconds, understand what each line does, then jump straight to context methods.

---

## TL;DR — Run it

```bash
python examples/0_quick_start.py
```

**Expected output**

* `AetherGraph sidecar server started at: http://127.0.0.1:...`
* Two channel messages (hello + LLM reply)
* Final print: `Result: {'final_output': 'HELLO WORLD'}`

---

## What this example shows

* **Start the sidecar**: `start()` — brings up default services (channel, LLM, memory, logging) so you don’t wire anything manually.
* **Turn a function into an agent**: `@graph_fn` — a normal async Python function becomes a first‑class agent step.
* **Use built‑in services via context**: `context.channel()`, `context.llm()`, `context.logger()` — consistent API across local/remote providers.
* **Return structured outputs**: return a `dict` so steps compose cleanly.
* **Run conveniently**: `run(my_fn, inputs={...})` — a helper that executes the step with a fresh context.

---

## Code anatomy (line‑by‑line mental model)

```python
from aethergraph import graph_fn, NodeContext
from aethergraph.runner import run 

@graph_fn(name="hello_world")
async def hello_world(input_text: str, *, context: NodeContext) -> str:
    # log the function call by context's logger
    context.logger().info("hello_world function called")

    # send a greeting message to the default channel (Console). More channels can be configured. 
    await context.channel().send_text(f"👋 Hello! You sent: {input_text}")

    # use llm service to generate a response with context.llm()
    llm_text, _usage = await context.llm().chat(
        messages=[
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": f"Say hi back to: {input_text}"},
        ],
    )

    await context.channel().send_text(f"🤖 LLM responded: {llm_text}")

    context.logger().info("hello_world function completed")
    return {"response": llm_text}

result = run(hello_world, inputs={"input_text": "hello world"}) # one-line runner
print("Result:", result)
```

**Why it matters**

* You write ordinary Python; AetherGraph provides the **ambient runtime** (channel/LLM/memory/logging) through `NodeContext`.
* The return `dict` keeps composition explicit (great for fan‑in/fan‑out later).

---

## `graph_fn` in one minute

* **What it is**: a decorator that turns an async function into an **agent step**.
* **Signature**: your parameters are your inputs; you also receive `context: NodeContext`.
* **Contract**: return a `dict` of named outputs.
* **Why**: easy to test, chain, and visualize; no hidden globals.

> Think: *“a function with superpowers”* — it runs with a consistent service set wherever you execute it.

---

## `NodeContext` (essentials you’ll use immediately)

`NodeContext` is your **service hub**. In this example you used:

| Method              | What it gives you                        | Core calls (keep it small)                                             |
| ------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| `context.channel()` | User I/O (console/Slack/GUI via sidecar) | `send_text(text)`, `ask_text(prompt)`, `ask_approval(prompt, options)` |
| `context.llm(name)` | An LLM client by name                    | `chat(messages) -> (text, usage)`                                      |
| `context.logger()`  | Structured logging                       | `info(msg)`, `warning(msg)`, `error(msg)`                              |
| `context.memory()`  | Lightweight recent records               | `record(kind, value)`, `recent(kinds=[...])`                           |

> You don’t need to configure providers here. The sidecar supplies sensible defaults; later you can swap in Slack/Telegram or your preferred LLM.

For more examples and context method introduction, please see [documentation](https://aiperture.github.io/aethergraph-docs/) and [example repo](https://github.com/AIperture/aethergraph-examples)
 
---


