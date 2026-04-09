from aethergraph.contracts.services.metering import MeteringService


class NoopMeteringService(MeteringService):
    async def record_llm(self, **kwargs):
        return {}

    async def record_embedding(self, **kwargs): ...
    async def record_run(self, **kwargs): ...
    async def record_artifact(self, **kwargs): ...
    async def record_event(self, **kwargs): ...

    async def get_overview(self, **kwargs) -> dict[str, int]:
        return {}

    async def get_llm_stats(self, **kwargs) -> dict[str, dict[str, int]]:
        return {}

    async def get_embedding_stats(self, **kwargs) -> dict[str, dict[str, int]]:
        return {}

    async def get_graph_stats(self, **kwargs) -> dict[str, dict[str, int]]:
        return {}

    async def get_artifact_stats(self, **kwargs) -> dict[str, dict[str, int]]:
        return {}

    async def get_memory_stats(self, **kwargs) -> dict[str, dict[str, int]]:
        return {}
