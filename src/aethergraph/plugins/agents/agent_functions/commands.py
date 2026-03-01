from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from aethergraph.core.runtime.node_context import NodeContext

if TYPE_CHECKING:
    from aethergraph.plugins.agents.types import ClassifiedIntent

CommandHandler = Callable[..., Awaitable[str]]


@dataclass
class CommandSpec:
    name: str
    summary: str
    usage: str
    handler: CommandHandler
    aliases: list[str] | None = None

    def all_names(self) -> list[str]:
        return [self.name] + (self.aliases or [])


# We'll build COMMANDS after defining handlers below.
COMMANDS: dict[str, CommandSpec] = {}


# ----------------------- /help & internal helpers -----------------------


async def _cmd_help(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /help
    /help <command>
    """
    args = args.strip()

    # ----------------------------------------------------------------------
    # 1) Global help: list all commands
    # ----------------------------------------------------------------------
    if not args:
        # Deduplicate by spec object, then sort by primary name
        seen_specs = {id(v): v for v in COMMANDS.values()}.values()
        specs_sorted = sorted(seen_specs, key=lambda s: s.name)

        lines: list[str] = []
        lines.append("### Available commands\n")
        lines.append("Use the following slash commands to inspect and control Aether.\n")

        for spec in specs_sorted:
            names = [spec.name] + (spec.aliases or [])
            primary = names[0]
            aliases = names[1:]

            # `/runs` (aliases: `/r`, `/recent`)
            alias_part = ""
            if aliases:
                alias_str = ", ".join(f"`/{a}`" for a in aliases)
                alias_part = f" (aliases: {alias_str})"

            lines.append(f"- `/{primary}`{alias_part} — {spec.summary}")

        lines.append("\nType `/help <command>` for detailed usage of a specific command.")
        return "\n".join(lines)

    # ----------------------------------------------------------------------
    # 2) Detailed help for a specific command
    # ----------------------------------------------------------------------
    target = args.lstrip("/").strip()
    spec = _find_command_spec(target)
    if not spec:
        return f"Unknown command `/{target}`. Type `\/help` to see all commands."

    names = [spec.name] + (spec.aliases or [])
    alias_str = " | ".join(f"/{n}" for n in names)

    return f"""### Help: `{alias_str}`

{spec.usage}"""


def _find_command_spec(name: str) -> CommandSpec | None:
    name = name.strip().lower()
    for spec in COMMANDS.values():
        if name == spec.name.lower() or name in {a.lower() for a in (spec.aliases or [])}:
            return spec
    return None


# ----------------------- /whoami -----------------------


async def _cmd_whoami(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /whoami
    """
    identity = context.identity()
    # Adjust this depending on your actual identity object shape
    org_id = getattr(identity, "org_id", None)
    user_id = getattr(identity, "user_id", None)
    client_id = getattr(identity, "client_id", None)
    session_id = getattr(identity, "session_id", None)

    lines = [
        "Current Aether identity:",
        f"  org_id:    {org_id}",
        f"  user_id:   {user_id}",
        f"  client_id: {client_id}",
        f"  session_id:{session_id}",
    ]
    return "\n".join(lines)


# ----------------------- /graphs -----------------------


async def _cmd_graphs(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /graphs
    """
    # TODO: replace with your real graph registry query:
    # e.g., graph_svc = services.graph_service
    # graphs = await graph_svc.list_graphs_for_identity(context.identity())
    #
    # For now, placeholder text to show shape:
    # (Better than failing if you haven't wired services yet.)
    placeholder = True
    if placeholder:
        return "TODO: implement /graphs using your graph registry / service."

    # Example of what you might do once you wire it:
    # lines = ["Registered graphs:"]
    # for g in graphs:
    #     lines.append(f"  {g.id:<20} {g.name or ''}  [{g.kind}]")
    # return "\n".join(lines)


# ----------------------- /agents -----------------------


async def _cmd_agents(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /agents
    """
    # TODO: wire this to your AgentRegistry:
    #   agent_reg = services.agent_registry
    #   agents = await agent_reg.list_agents_for_identity(context.identity())
    return "TODO: implement /agents using your AgentRegistry (list installed agents)."


# ----------------------- /runs -----------------------


async def _cmd_runs(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /runs
    /runs <limit>
    """
    # Parse limit
    limit = 10
    if args.strip():
        try:  # noqa: SIM105
            limit = max(1, int(args.strip().split()[0]))
        except Exception:
            pass

    # TODO: wire to your run / execution service
    #   run_svc = services.run_service
    #   runs = await run_svc.list_recent_runs(identity=context.identity(), limit=limit)
    #
    # Placeholder:
    return f"TODO: implement /runs; would list last {limit} runs for this identity."


# ----------------------- /triggers -----------------------


async def _cmd_triggers(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /triggers
    """
    # TODO: wire to your TriggerService:
    #   trig_svc = services.trigger_service
    #   triggers = await trig_svc.list_triggers_for_identity(context.identity())
    return "TODO: implement /triggers using your TriggerService (list triggers)."


# ----------------------- /logs -----------------------


async def _cmd_logs(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
) -> str:
    """
    /logs
    /logs <limit>
    """
    # Very simple version: you can later wire this to a log store or event log.
    limit = 20
    if args.strip():
        try:  # noqa: SIM105
            limit = max(1, int(args.strip().split()[0]))
        except Exception:
            pass

    # TODO: replace this with a call to an event log / log store service
    return (
        "TODO: implement /logs to show recent log lines or events.\n"
        f"(Requested last {limit} entries.)"
    )


async def _cmd_use(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
    session_id: str | None,
) -> str:
    """
    /use <agent>

    Set the active agent for this session. Subsequent messages (without / or @)
    will go to that agent until you /exit or /use another agent.
    """
    alias_or_id = args.strip()
    if not alias_or_id:
        return "Usage: `/use <agent>`\n\nExample: `/use deeplens`"

    # # Try treating it as an id first; if you have a registry, you can check.
    # # For now, resolve as alias.
    # agent_id = await resolve_agent_alias(alias_or_id, context=context) or alias_or_id

    # # TODO: you may want to validate that agent_id exists in AgentRegistry here.

    # state = await get_session_agent_state(context, session_id)
    # state.active_agent_id = agent_id
    # await set_session_agent_state(context, session_id, state)

    return (
        f"Switched active agent to `{alias_or_id}` for this session.\n"
        "Messages (without `/` or leading `@`) will now be sent there.\n"
        "Use `/exit` to return to the Aether Agent shell."
    )


async def _cmd_exit(
    *,
    intent: "ClassifiedIntent",
    args: str,
    context: NodeContext,
    session_id: str | None,
) -> str:
    """
    /exit

    Return to the Aether Agent shell (clear active agent).
    """
    # state = await get_session_agent_state(context, session_id)
    # state.active_agent_id = "aether_agent"
    # await set_session_agent_state(context, session_id, state)

    return "Exited back to the Aether Agent shell. Messages will be handled here again."


def _init_command_registry() -> dict[str, CommandSpec]:
    # Primary specs
    specs: list[CommandSpec] = [
        CommandSpec(
            name="help",
            aliases=[],
            summary="Show available commands or usage for a specific command.",
            usage="  /help\n  /help <command>",
            handler=_cmd_help,
        ),
        CommandSpec(
            name="whoami",
            aliases=[],
            summary="Show current Aether identity (org_id, user_id, etc.).",
            usage="  /whoami",
            handler=_cmd_whoami,
        ),
        CommandSpec(
            name="graphs",
            aliases=[],
            summary="List registered graphs (workflows) visible to you.",
            usage="  /graphs",
            handler=_cmd_graphs,
        ),
        CommandSpec(
            name="agents",
            aliases=[],
            summary="List installed agents (graph entrypoints with as_agent metadata).",
            usage="  /agents",
            handler=_cmd_agents,
        ),
        CommandSpec(
            name="runs",
            aliases=[],
            summary="Show recent graph runs.",
            usage="  /runs\n  /runs <limit>",
            handler=_cmd_runs,
        ),
        CommandSpec(
            name="triggers",
            aliases=[],
            summary="List triggers associated with your identity.",
            usage="  /triggers",
            handler=_cmd_triggers,
        ),
        CommandSpec(
            name="logs",
            aliases=[],
            summary="Show recent log lines or events.",
            usage="  /logs\n  /logs <limit>",
            handler=_cmd_logs,
        ),
        CommandSpec(
            name="use",
            aliases=[],
            summary="Set the active agent for this session.",
            usage="  /use <agent_alias_or_id>",
            handler=_cmd_use,
        ),
        CommandSpec(
            name="exit",
            aliases=[],
            summary="Exit the active agent and return to the Aether Agent shell.",
            usage="  /exit",
            handler=_cmd_exit,
        ),
    ]

    registry: dict[str, CommandSpec] = {}
    for spec in specs:
        for name in spec.all_names():
            key = name.lower()
            if key in registry:
                # You can log a warning if aliases collide; for now just keep first
                continue
            registry[key] = spec
    return registry


# Initialize global registry
COMMANDS = _init_command_registry()
