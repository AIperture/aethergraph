from __future__ import annotations

from collections.abc import Sequence
import json
from typing import TYPE_CHECKING, Any, Literal

from .utils import short_hash, slug

if TYPE_CHECKING:
    from aethergraph.contracts.services.memory import Event

    from .types import MemoryFacadeInterface


class RAGMixin:
    """Methods for interacting with RAG services."""

    # ----------- RAG: DX helpers (key-based) -----------
    async def rag_remember_events(
        self,
        *,
        key: str = "default",
        where: dict | None = None,
        policy: dict | None = None,
    ) -> dict:
        """
        High-level: bind a RAG corpus by logical key and promote events into it.

        Example:
          await mem.rag_remember_events(
              key="session",
              where={"kinds": ["tool_result"], "limit": 200},
              policy={"min_signal": 0.25},
          )
        """
        corpus_id = await self.rag_bind(key=key, create_if_missing=True)
        return await self.rag_promote_events(
            corpus_id=corpus_id,
            events=None,
            where=where,
            policy=policy,
        )

    async def rag_remember_docs(
        self,
        docs: Sequence[dict[str, Any]],
        *,
        key: str = "default",
        labels: dict | None = None,
    ) -> dict[str, Any]:
        """
        High-level: bind a RAG corpus by key and upsert docs into it.
        """
        corpus_id = await self.rag_bind(key=key, create_if_missing=True, labels=labels)
        return await self.rag_upsert(corpus_id=corpus_id, docs=list(docs))

    async def rag_search_by_key(
        self,
        *,
        key: str = "default",
        query: str,
        k: int = 8,
        filters: dict | None = None,
        mode: Literal["hybrid", "dense"] = "hybrid",
    ) -> list[dict]:
        """
        High-level: resolve corpus by logical key and run rag_search() on it.
        """
        corpus_id = await self.rag_bind(key=key, create_if_missing=False)
        return await self.rag_search(
            corpus_id=corpus_id,
            query=query,
            k=k,
            filters=filters,
            mode=mode,
        )

    async def rag_answer_by_key(
        self,
        *,
        key: str = "default",
        question: str,
        style: Literal["concise", "detailed"] = "concise",
        with_citations: bool = True,
        k: int = 6,
    ) -> dict:
        """
        High-level: RAG QA over a corpus referenced by logical key.

        Internally calls rag_bind(..., create_if_missing=False) and rag_answer().
        """
        corpus_id = await self.rag_bind(key=key, create_if_missing=False)
        return await self.rag_answer(
            corpus_id=corpus_id,
            question=question,
            style=style,
            with_citations=with_citations,
            k=k,
        )

    async def rag_upsert(
        self: MemoryFacadeInterface,
        *,
        corpus_id: str,
        docs: Sequence[dict[str, Any]],
        topic: str | None = None,
    ) -> dict[str, Any]:
        if not self.rag:
            raise RuntimeError("RAG facade not configured")
        return await self.rag.upsert_docs(corpus_id=corpus_id, docs=list(docs))

    async def rag_bind(
        self: MemoryFacadeInterface,
        *,
        corpus_id: str | None = None,
        key: str | None = None,
        create_if_missing: bool = True,
        labels: dict | None = None,
    ) -> str:
        if not self.rag:
            raise RuntimeError("RAG facade not configured")

        mem_scope = self.memory_scope_id
        if corpus_id:
            cid = corpus_id
        else:
            logical_key = key or "default"
            base = f"{mem_scope}:{logical_key}"
            cid = f"mem:{slug(mem_scope)}:{slug(logical_key)}-{short_hash(base, 8)}"

        scope_labels = {}
        if self.scope:
            scope_labels = self.scope.rag_labels(scope_id=mem_scope)

        meta = {"scope": scope_labels, **(labels or {})}
        if create_if_missing:
            await self.rag.add_corpus(cid, meta=meta, scope_labels=scope_labels)
        return cid

    async def rag_promote_events(
        self: MemoryFacadeInterface,
        *,
        corpus_id: str,
        events: list[Event] | None = None,
        where: dict | None = None,
        policy: dict | None = None,
    ) -> dict:
        if not self.rag:
            raise RuntimeError("RAG facade not configured")
        policy = policy or {}
        min_signal = policy.get("min_signal", self.default_signal_threshold)

        if events is None:
            # We use RetrievalMixin's .recent here
            kinds = (where or {}).get("kinds")
            limit = int((where or {}).get("limit", 200))
            recent = await self.recent(kinds=kinds, limit=limit)  # type: ignore
            events = [e for e in recent if (getattr(e, "signal", 0.0) or 0.0) >= float(min_signal)]

        docs: list[dict] = []
        for e in events:
            title = f"{e.kind}:{(e.tool or e.stage or 'n/a')}:{e.ts}"
            scope_labels = (
                self.scope.rag_labels(scope_id=self.memory_scope_id) if self.scope else {}
            )
            labels = {
                **scope_labels,
                "kind": e.kind,
                "tool": e.tool,
                "stage": e.stage,
                "severity": e.severity,
                "tags": list(e.tags or []),
            }
            body = e.text
            if not body:
                body = json.dumps(
                    {"inputs": e.inputs, "outputs": e.outputs, "metrics": e.metrics},
                    ensure_ascii=False,
                )
            docs.append({"text": body, "title": title, "labels": labels})

        if not docs:
            return {"added": 0}

        stats = await self.rag.upsert_docs(corpus_id=corpus_id, docs=docs)

        # Log result
        await self.write_result(
            tool=f"rag.promote.{corpus_id}",
            outputs=[{"name": "added_docs", "kind": "number", "value": stats.get("added", 0)}],
            tags=["rag", "ingest"],
            message=f"Promoted {stats.get('added', 0)} events",
            severity=2,
        )
        return stats

    async def rag_answer(
        self: MemoryFacadeInterface,
        *,
        corpus_id: str,
        question: str,
        style: Literal["concise", "detailed"] = "concise",
        with_citations: bool = True,
        k: int = 6,
    ) -> dict:
        if not self.rag:
            raise RuntimeError("RAG facade not configured")

        ans = await self.rag.answer(
            corpus_id=corpus_id,
            question=question,
            llm=self.llm,
            style=style,
            with_citations=with_citations,
            k=k,
        )

        outs = [{"name": "answer", "kind": "text", "value": ans.get("answer", "")}]
        for i, rc in enumerate(ans.get("resolved_citations", []), start=1):
            outs.append({"name": f"cite_{i}", "kind": "json", "value": rc})

        await self.write_result(
            tool=f"rag.answer.{corpus_id}",
            outputs=outs,
            tags=["rag", "qa"],
            message=f"Q: {question}",
            metrics=ans.get("usage", {}),
            severity=2,
        )
        return ans
