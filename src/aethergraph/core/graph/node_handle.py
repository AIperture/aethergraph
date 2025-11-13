from dataclasses import dataclass 
from typing import Dict, List 
from .graph_refs import ref as _ref 

@dataclass
class NodeHandle:
    """ A handle to a node's outputs in the graph. 
     Allows access to outputs via attribute or key.
    """
    node_id: str 
    output_keys: List[str]

    def __getattr__(self, name: str) -> Dict[str, str]:
        """ Access output by attribute, e.g. node.output_key """
        # Allow handle.path or handle.analysis -> Ref
        if name in self.output_keys:
            return _ref(self.node_id, name)
        raise AttributeError(f"NodeHandle has no output '{name}'")
    
    def __getitem__(self, key: str) -> Dict[str, str]:
        """ Access output by key, e.g. node["output_key"] """
        # Allow handle["path"] or handle["analysis"] -> Ref
        if key in self.output_keys:
            return _ref(self.node_id, key)
        raise KeyError(f"NodeHandle has no output '{key}'")
    
    def tuple(self, n: int):
        return [
            _ref(self.node_id, f"out{i}") for i in range(n)
        ]
    
    def unpack(self, *names: str):
        return [
            _ref(self.node_id, name) for name in names
        ]
    
    