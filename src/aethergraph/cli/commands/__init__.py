"""
Current top-level commands: serve, run, register.

Planned commands with existing backend seams:
- registry list: backed by RegistryFacade.list/list_agents/list_apps.
- registry validate: backed by RegistryFacade.validate_graphify_source.
- registry replay: backed by RegistryFacade.replay_registered_sources.
- registry delete-app: backed by RegistryFacade.delete_registered_app.
- registry delete-agent: backed by RegistryFacade.delete_registered_agent.
- register-folder: backed by RegistryFacade.register_by_folder.
- run status: backed by GET /api/v1/runs/{run_id}.
- run watch: backed by existing run polling flow; stream/log UX still needs shape.
- server status: backed by workspace server.json read/validation.
- server stop: blocked on safe PID ownership and shutdown semantics.
"""
