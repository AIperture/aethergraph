"""
Compatibility shim.

`MemoryFacadeInterface` has moved to contracts as `MemoryFacadeProtocol`.
Keep this alias to avoid breaking older imports.
"""

from aethergraph.contracts.services.memory import MemoryFacadeProtocol as MemoryFacadeInterface

__all__ = ["MemoryFacadeInterface"]
