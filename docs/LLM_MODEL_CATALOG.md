# Maintaining the LLM model catalog

The packaged catalog at `src/aethergraph/services/llm/catalog/model_catalog.v1.json`
is the source of provider model facts. Add a new entry only for capabilities backed
by a provider document. Keep unknown capabilities unknown, even when a neighboring
model supports them. Use `model_ids` for a documented list of exact IDs and
`model_pattern` only when the documented family has a stable naming rule.

For each update, record source URLs, `verified_at`, `catalog_revision`, and a
priority that makes the intended capability entry win. A capability can resolve
from a different entry than another capability for the same model. Existing
patterns do not need to be rewritten when a higher-priority exact entry records
a new model or a changed provider contract.

Run these commands from an environment that imports this AetherGraph checkout:

```powershell
python -m aethergraph.services.llm.catalog validate
python -m aethergraph.services.llm.catalog coverage --provider openai --endpoint openai_responses --models gpt-6-luna gpt-6-sol
python -m pytest tests/test_llm_model_catalog.py -q
```

The coverage command uses the same resolver as runtime calls and reports the
winning catalog key, documentation URLs, and verification date for each capability.
It reports missing coverage as `null`. A catalog fact does not by itself qualify
a model for an application's agent workflow; that application should maintain its
own tested binding list. Live provider tests should use the resolved catalog
capability rather than construct a synthetic capability object.
