from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMCallCorrelation:
    llm_call_id: str
    prompt_manifest_id: str | None = None


_current_llm_call: ContextVar[LLMCallCorrelation | None] = ContextVar(
    "aethergraph_llm_call_correlation",
    default=None,
)


def begin_llm_call_correlation(llm_call_id: str) -> None:
    _current_llm_call.set(LLMCallCorrelation(llm_call_id=llm_call_id))


def complete_llm_call_correlation(
    llm_call_id: str,
    *,
    prompt_manifest_id: str | None,
) -> None:
    _current_llm_call.set(
        LLMCallCorrelation(
            llm_call_id=llm_call_id,
            prompt_manifest_id=prompt_manifest_id,
        )
    )


def current_llm_call_correlation() -> LLMCallCorrelation | None:
    return _current_llm_call.get()
