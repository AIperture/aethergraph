# Storage providers and workspace operations

AetherGraph opens exactly one storage provider for each runtime. That provider owns
one coherent bundle containing state, events, memory, artifacts, search, control
records, observations, integration streams, and runtime output. Runtime services use
the bundle's typed repositories; they do not select databases or inspect
provider-private paths.

The built-in provider is `local.sqlite`. External providers use the same contracts
and must be registered explicitly by an embedding Host. A missing, unhealthy, or
incomplete selected provider fails startup directly. AetherGraph never substitutes
the local provider, performs a second read or write, or probes another workspace
layout.

## Configuration

The default configuration is equivalent to:

```ini
AETHERGRAPH_STORAGE_PROVIDER__PROVIDER=local.sqlite
AETHERGRAPH_STORAGE_PROVIDER__PROFILE=default
AETHERGRAPH_STORAGE_PROVIDER__OPTIONS__BUSY_TIMEOUT_MS=5000
AETHERGRAPH_STORAGE_PROVIDER__OPTIONS__DURABILITY=normal
AETHERGRAPH_STORAGE_PROVIDER__OPTIONS__RUNTIME_OUTPUT_MAX_PENDING_FRAMES=10000
AETHERGRAPH_STORAGE_PROVIDER__OPTIONS__SEARCH_MAX_CANDIDATES=10000
```

`local.sqlite` accepts only these options. `durability` is `normal` or `full`.
The continuation-secret reference is fixed to AG's workspace-bound derivation from
the resolved authentication signing secret and normally should not be set manually.
Unknown options, hidden paths, or `app_id`/`application_id`/`client_id` provider
options fail validation. App identity is deprecated optional compatibility metadata,
not storage scope or provider configuration.

The `profile` field labels trusted configuration resolution. It is not passed to the
provider and does not select a fallback.

## External injection

An embedding Host selects an external provider by exact name and supplies its factory
and secret resolver in the closed runtime request:

```python
from pathlib import Path

from aethergraph.runtime import RuntimeOpenRequest, open_embedded_runtime
from aethergraph.storage.contracts import StorageProviderSelection

request = RuntimeOpenRequest(
    root=Path("deployment/runtime").resolve(),
    settings=settings,
    workspace_id="deployment-runtime-1",
    storage_selection=StorageProviderSelection(
        provider="company.external",
        config={"cluster": "primary"},
    ),
    storage_providers={"company.external": build_company_provider},
    storage_secrets=company_secret_resolver,
)
runtime = open_embedded_runtime(request)
```

Provider names are lowercase exact identifiers. Registration is explicit and
duplicate names are rejected. The provider must return every `StorageBundle`
repository and satisfy the runtime capability set before the bundle is published:
durability, transactions, compare-and-set, ordered append, monotonic and shared
delivery cursors, TTL, leases, blob streaming/ranges, structural and lexical search,
and health. Semantic and hybrid search are optional capabilities; requesting an
unsupported mode raises a typed capability error.

## Workspace format

The local provider owns `workspace.json` at the workspace root. Current local format
version is `1`. The manifest records exact workspace and owner identity, provider,
format, a configuration fingerprint, compatibility metadata, and lifecycle state. It
never stores raw provider options or resolved secrets.

A writable runtime initializes only an absent or empty workspace root. A non-empty
root without a valid current manifest is rejected. Writable reopen validates the
configuration fingerprint. The historical observability reader opens only a current
manifested workspace in read-only mode and validates provider, format, workspace, and
owner identity; it does not probe database filenames.

Workspace formats are exact compatibility boundaries. Unknown fields, unsupported
versions, malformed manifests, symlinked manifests, identity mismatches, and
unmanifested data fail closed.

## Backup and restore

For `local.sqlite`, stop every runtime using the workspace and let provider shutdown
complete before copying data. Back up the entire workspace root as one unit,
including `workspace.json` and every provider-owned file. Do not copy individual
SQLite files while the runtime is active and do not omit WAL or other provider-owned
files.

Restore the complete backup to the same authorized absolute root, or open it with the
same explicit `workspace_id`, owner scope, provider selection, and writable
configuration. A path-derived default workspace identity changes when the root moves,
so relocation requires an explicit stable identity supplied by the Host. Validate the
backup and restore procedure in a disposable environment before relying on it.

External-provider backup, restore, retention, and disaster recovery are owned by that
provider. AG still requires the restored provider to report healthy before publishing
runtime services.

## Clean-cut history warning

Runtime history written by pre-provider AetherGraph builds is not migrated and is not
accepted by the current runtime or historical reader. Back up any old workspace you
must retain before upgrading, keep that backup offline with its matching old software,
and initialize a fresh empty current-format workspace. Copying old database files into
a new workspace does not convert them and will cause startup to fail.

Project source, provider configuration without secrets, and separately managed input
data should be backed up independently. Removing an old runtime workspace permanently
removes its runs, sessions, memory, artifacts, continuations, trigger state,
observations, and search indexes unless an offline backup exists.
