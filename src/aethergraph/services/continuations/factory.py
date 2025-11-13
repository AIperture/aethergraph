import os 
from aethergraph.contracts.services.continuations import AsyncContinuationStore
from stores.inmem_store import InMemoryContinuationStore
from stores.fs_store import FSContinuationStore 

def make_continuation_store() -> AsyncContinuationStore:
    """ Factory to create a continuation store based on environment configuration. 
     Returns:
        An instance of AsyncContinuationStore.
    
    Currently supports:
    - InMemoryContinuationStore (CONT_STORE="inmem")
    - FSContinuationStore (CONT_STORE="fs")

    We need env vars:
    - CONT_STORE: "inmem" or "fs" (default: "fs")
    - CONT_SECRET: optional secret for HMAC token generation
    - CONT_ROOT: for FS store, root directory to store continuations (default: "./artifacts/continuations")
    """
    kind = (os.getenv("CONT_STORE", "fs")).lower()
    secret = os.getenv("CONT_SECRET")
    secret_bytes = secret.encode("utf-8") if secret else os.urandom(32)

    if kind == "inmem":
        return InMemoryContinuationStore(secret=secret_bytes)
    elif kind == "fs":
        return FSContinuationStore(root=os.getenv("CONT_ROOT", "./artifacts/continuations"), secret=secret_bytes)
    else:
        raise ValueError(f"Unknown continuation store kind: {kind}")