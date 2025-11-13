from typing import Any, Dict, List, Optional, Tuple, Protocol

class LLMClientProtocol(Protocol):
    async def chat(self, messages: List[Dict[str, Any]], **kw) -> Tuple[str, Dict[str, int]]: ...
    async def embed(self, texts: List[str], **kw) -> List[List[float]]: ...
    async def raw(
        self,
        *,
        method: str = "POST",
        path: Optional[str] = None,
        url: Optional[str] = None,
        json: Any | None = None,
        params: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        return_response: bool = False,
    ) -> Any: ...
