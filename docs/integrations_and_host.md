# AG Host, Agent Endpoint, and integrations

AG Host runs one immutable compiled agent release. Every interactive transport—AG UI,
a bespoke application, Slack, or Telegram—enters the same
`IntegrationIngressCoordinator`. Transports authenticate and translate their native
payloads into an ingress envelope; they do not select an agent, create a run directly,
stage attachments independently, or resolve continuations by guesswork.

```text
AG UI / bespoke UI / Slack / Telegram
    -> authenticated transport edge
    -> IngressEnvelope
    -> IntegrationIngressCoordinator
    -> exact route + durable session binding
    -> exact continuation resume OR one root dispatch
    -> semantic event log
    -> origin-bound delivery
```

## Channel and origin ownership

Agent code normally calls `context.channel()` without a transport address. The runtime
binds an immutable origin to the root turn and preserves it through nested and resumed
execution. Engine semantic policy chooses operations such as text, phase, progress,
interaction, artifact, and structured output. The Host adapter projects those operations
for the originating transport.

Do not put Slack channel IDs, Telegram chat IDs, endpoint IDs, or `ui:session` addresses
in agent code. Do not mutate process-global Channel defaults. Deployment routes and the
authenticated ingress envelope own external addressing.

## Immutable Host launch

The blocking Host command consumes only explicit launch handles:

```powershell
python -m aethergraph host `
  --manifest C:\deployment\host-manifest.json `
  --runtime-identity C:\deployment\runtime-identity.json `
  --settings C:\deployment\application.env `
  --workspace C:\deployment\runtime `
  --control-token C:\deployment\control-token.handle `
  --provider-secrets C:\deployment\provider-secrets.json
```

`--provider-secrets` is omitted when no Slack or Telegram route is enabled. The command
listens only on loopback and chooses an ephemeral port. It emits one structured readiness
handshake on stdout. Health, readiness, diagnostics, and shutdown require the per-launch
`X-AG-Host-Control` token. Studio creates and protects these files; manual callers must do
the same.

Before importing generated code, Host verifies:

- the canonical Host manifest digest and compiled manifest checksum;
- exact AetherGraph and Engine versions;
- Python ABI, platform, and architecture;
- installed distribution `RECORD` hashes and their lock digest;
- Host capabilities and services;
- ingress and semantic protocol versions;
- logical output requirements and entrypoint schemas;
- compiler provenance and compiled build identity.

An incompatible release exits with `host.release_incompatible` and an actionable reason.
There is no alternate interpreter, dependency installation, raw module import, or
compatibility execution path.

## Bespoke applications through Agent Endpoint

A deployment route of kind `agent_endpoint` or `ag_ui` supplies an immutable
`endpoint_id`. A client never sends `agent_id` or `graph_id` with a message.

The public route family is:

```text
POST /api/v1/agent-endpoints/{endpoint_id}/authenticate
POST /api/v1/agent-endpoints/{endpoint_id}/sessions
POST /api/v1/agent-endpoints/{endpoint_id}/ingress
GET  /api/v1/agent-endpoints/{endpoint_id}/sessions/{session_id}/events
GET  /api/v1/agent-endpoints/{endpoint_id}/sessions/{session_id}/stream
POST /api/v1/agent-endpoints/{endpoint_id}/sessions/{session_id}/cancel
GET  /api/v1/agent-endpoints/{endpoint_id}/artifacts/{artifact_id}
```

Host creates one random, eight-hour credential for each enabled AG UI or Agent Endpoint
route and transfers it only in the private readiness handshake. A browser launch carries
that bounded credential in the URL fragment, removes the fragment before rendering, and
exchanges it through `authenticate` for an HttpOnly, SameSite-strict cookie scoped to the
exact endpoint path. Host restart issues new credentials. Execution routes never accept a
query token or generic local identity, and a credential for one endpoint cannot authorize
another endpoint.

Create a session with a client-stable idempotency key:

```json
{"idempotency_key":"browser-conversation-42","title":"Support"}
```

Then submit a closed ingress command:

```json
{
  "session_id": "endpoint-session-...",
  "idempotency_key": "message-0001",
  "text": "Summarize the attached report.",
  "attachments": []
}
```

Repeated session or ingress keys replay the same durable identity/receipt; they do not
start a second turn. A choice response supplies the public `interaction_id` and exact
`option_ids`. Free text resumes only one eligible interaction in the bound session;
missing, mismatched, or ambiguous interactions reject instead of selecting the newest
wait.

History and Server-Sent Events use the same ordered semantic cursor. Persist the last
cursor and reconnect with `after_cursor`; do not merge this stream with legacy session
chat or run polling. Cancellation requires the exact `turn_id` owned by the endpoint
session.

### Attachments

JSON ingress may reference an already authenticated artifact. Multipart ingress may carry
uploads plus the same closed body fields. Both reach `ResourceIngress`, which enforces the
shared size/type/source policy and produces the same canonical resource shape. Provider
adapters may fetch protected bytes at their authenticated edge, but they do not create a
second staging or agent-payload path.

## Slack and Telegram local operation

Provider credentials are explicit Host launch secrets attached to immutable routes. They
are not discovered from a global `.env` file.

- Slack local hosting uses Socket Mode with a bot token and app-level token.
- Telegram local hosting uses polling with a bot token.
- Webhook configuration is not part of the local deployment workflow.
- A provider connection must pass its explicit readiness check before Studio can deploy a
  route that references it.

The provider runner extracts authenticated identities and attachments, submits the same
ingress envelope as Agent Endpoint, and projects semantic events back to the verified
origin. There is no provider default agent or implicit destination.

## Removed integration paths

The following interfaces are intentionally unavailable: the Channel API stub,
`/chat/incoming` root dispatch, the UI chat WebSocket/event wrapper, `/ws/channel`
outbox streaming, `ChannelIngress`, provider default-agent routing, process-global Channel
defaults/aliases, and environment-driven provider startup. Migrate applications to Agent
Endpoint or a provider edge over `IntegrationIngressCoordinator`; no shim is retained.
